"""问 3：附件 3 预报插值、固定时点滚动重优化与调整购电结算。

每个决策日：
  1. 0:00 用附件 3 的正式预报与三日历史光伏均值组合，负载采用分解式预测；完整交付后
     才可选用的同发布时刻配对路径构成场景，SAA 两阶段 LP 冻结计划购电量 $G^0$；
  2. 在 6:00 / 12:00 / 18:00 用新预报与当前执行 SOC 求解重优化 LP，决定调整购电量
     $G^a = G^0 + \\Delta^+ - \\Delta^-$，**只提交到下一个预报时刻**；
  3. 储能由精确凸分段线性 DP 的未来费用函数逐段因果执行；固定购电量为已提交段的
     $G^a$ 与未提交段的 $G^0$ 共同组成的数组；
  4. 按 $p\\min(G^0,G^a) + 0.5p(G^0-G^a)^+ + 1.5p(G^a-G^0)^+ + 5pE$ 结算。

`--issues` 可选预报时刻子集：集合 $S$ 中的时刻 $h_0$ 提交到 $S$ 中下一个时刻（或 24:00），
不在 $S$ 中的时刻既不重优化也不刷新点预测曲线，用于回答「是否需要引入其他时刻预报」。

常规运行中 1 月只生成当时发布的预测与完整路径残差，SOC 保持待机的 6000 kWh；
2–12 月连续执行。只有显式校准运行才在一月执行，且其状态不传入正式回测。
参考基线在每个发布时刻使用完整 24h 前瞻与线性续存费用；48h + 次日点预测
价值函数为显式增量（`horizon_days=2`）。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path

import numpy as np

from .data import (
    ETA,
    SOC_INIT,
    SOC_MAX,
    SOC_MIN,
    T,
    Attachment1,
    DailySeries,
    load_attachment1,
    load_attachment2,
    load_attachment3,
)
from .checkpoint import identity as checkpoint_identity, load as load_checkpoint, save as save_checkpoint
from .executor import DayExecution, run_day_dp, DPValueExecutor
from .value_dp import ConvexPiecewiseLinear, future_cost, next_day_value
from .parallel_eval import CandidateEvaluator
from .forecast import PointForecaster
from .forecast_price import ConstantPriceSource, PriceSource
from .forecast_pv import ISSUE_HOURS, PVIssueForecaster, issue_segment
from .optimizer import solve_readjust, solve_saa, solve_saa_point
from .results import print_day_summary_q3, summarize_day_q3, variant_table, write_result3
from .run_q2 import PERIOD_END, RECORD_START, WARMUP_START, daterange
from .scenarios import IssuedPathLibrary, M_SCEN
from .settlement import cost_q3

REPORT_DAYS = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]
# 问 2 在 1 月计分窗口内选定的参数，问 3 直接沿用，不再调参。
K_LOAD = 6
K_PV = 7
VARIANTS = [(0,), (0, 6), (0, 6, 12), (0, 6, 12, 18)]


@dataclass
class Params:
    """一次回测的全部超参数。issues 为可用的预报时刻集合（必须含 0）。"""

    k_load: int = K_LOAD
    k_pv: int = K_PV  # 只用于次日均值曲线与预报未覆盖的段
    m_scen: int = M_SCEN
    # Kept for legacy StorageRollingSolver experiments only; DP is always
    # executed at each native 10-minute segment.
    step_minutes: int = 10
    issues: tuple[int, ...] = ISSUE_HOURS
    candidate_workers: int = 1  # independent plan candidates; 1 keeps serial execution
    candidate_reeval: bool = False  # 凸组合重评为显式增量，不是论文基线
    horizon_days: int = 1
    reference_baseline: bool = True
    scenario_same_type: bool = False
    calibration: bool = False


@dataclass
class DayResult:
    """一天的计划购电、调整购电、执行与四项费用。"""

    day: date
    G0: np.ndarray
    Ga: np.ndarray
    C: np.ndarray
    D: np.ndarray
    S: np.ndarray
    E: np.ndarray
    W: np.ndarray
    soc_start: float
    plan_cost: float
    curtail_cost: float
    extra_cost: float
    emergency_cost: float
    R: np.ndarray | None = None
    plan_only_cost: float = 0.0  # Σ p·G⁰，「不调整」的假想购电费
    price: np.ndarray | None = None  # 该日结算电价（None = 附件 1 常数电价）

    @property
    def total_cost(self) -> float:
        return self.plan_cost + self.curtail_cost + self.extra_cost + self.emergency_cost

    @property
    def soc_end(self) -> float:
        return float(self.S[-1])

    @property
    def adjust_kwh(self) -> float:
        return float(np.abs(self.Ga - self.G0).sum())


@dataclass
class PeriodResult:
    """一段日期区间的回测汇总。"""

    days: list[DayResult] = field(default_factory=list)
    soc_end: float = SOC_INIT
    runtime_s: float = 0.0
    label: str = ""

    def _sum(self, attr: str) -> float:
        return float(sum(getattr(d, attr) for d in self.days))

    @property
    def plan_cost(self) -> float:
        return self._sum("plan_cost")

    @property
    def curtail_cost(self) -> float:
        return self._sum("curtail_cost")

    @property
    def extra_cost(self) -> float:
        return self._sum("extra_cost")

    @property
    def emergency_cost(self) -> float:
        return self._sum("emergency_cost")

    @property
    def total_cost(self) -> float:
        return self._sum("total_cost")

    @property
    def plan_kwh(self) -> float:
        return float(sum(float(d.G0.sum()) for d in self.days))

    @property
    def plan_only_cost(self) -> float:
        """假想的「不调整」购电费 Σ p·G⁰，用作调整净费用的基准。"""
        return self._sum("plan_only_cost")

    @property
    def adjust_net_cost(self) -> float:
        """调整相关净费用 =（计划费 + 减购违约费 + 增购费）− Σ p·G⁰。

        减购时每 kWh 净省 0.5p、增购时每 kWh 净付 1.5p；储能充裕时前者常占优，
        因此这个数可能为负（是预期结果，不是错误）。
        """
        return self.plan_cost + self.curtail_cost + self.extra_cost - self.plan_only_cost

    @property
    def purchase_kwh(self) -> float:
        return float(sum(float(d.Ga.sum()) for d in self.days))

    @property
    def emergency_kwh(self) -> float:
        return float(sum(float(d.E.sum()) for d in self.days))

    @property
    def emergency_days(self) -> int:
        return sum(1 for d in self.days if d.E.max() > 1e-6)

    @property
    def mean_adjust_kwh(self) -> float:
        """平均每段调整量 Σ|G^a − G^0| / 段数。"""
        n = len(self.days) * T
        return self._sum("adjust_kwh") / n if n else 0.0

    def by_date(self, d: date) -> DayResult | None:
        for day in self.days:
            if day.day == d:
                return day
        return None

    def row(self) -> dict:
        """变体对比表的一行。"""
        return {
            "label": self.label,
            "total_cost": self.total_cost,
            "plan_cost": self.plan_cost,
            "curtail_cost": self.curtail_cost,
            "extra_cost": self.extra_cost,
            "emergency_cost": self.emergency_cost,
            "emergency_kwh": self.emergency_kwh,
            "mean_adjust_kwh": self.mean_adjust_kwh,
            "adjust_net_cost": self.adjust_net_cost,
        }


@dataclass
class Bundle:
    """只读输入数据（附件 1、2、3），供多个变体复用。"""

    att1: Attachment1
    daily: DailySeries
    att3: dict[tuple[date, int], np.ndarray]
    _row: dict[date, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._row = {d: i for i, d in enumerate(self.daily.dates)}

    def truth(self, d: date) -> tuple[np.ndarray, np.ndarray]:
        """日期 d 的负载、光伏真值（只在执行与结算时调用，不进入预测路径）。"""
        i = self._row[d]
        return self.daily.load_kwh[i], self.daily.pv_kwh[i]


def load_bundle(
    path1: str = "data/附件1.xlsx",
    path2: str = "data/附件2.xlsx",
    path3: str = "data/附件3.xlsx",
) -> Bundle:
    return Bundle(
        att1=load_attachment1(path1),
        daily=load_attachment2(path2),
        att3=load_attachment3(path3),
    )


def block_bounds_of(issues: tuple[int, ...], n_seg: int = T) -> list[tuple[int, int, int]]:
    """预报时刻集合 → [(发布时刻, 起段, 止段)]，止段为下一个预报时刻（或一天末）。"""
    edges = [issue_segment(h) for h in issues] + [n_seg]
    return [(h, edges[j], edges[j + 1]) for j, h in enumerate(issues)]


def run_period(
    start: date,
    end: date,
    params: Params,
    bundle: Bundle,
    *,
    record_from: date | None = None,
    soc_init: float = SOC_INIT,
    progress: int = 0,
    price_source: PriceSource | None = None,
    checkpoint_path: str | None = None,
    resume: bool = False,
    checkpoint_tag: str | None = None,
) -> PeriodResult:
    """Publication-time walk-forward orchestration with complete residual paths.

    The baseline uses a 24h linear continuation.  The explicit 48h increment
    uses current-path uncertainty plus a point-predicted, free-purchase next
    day value.  January is standby unless ``calibration`` is enabled.
    """
    record_from = start if record_from is None else record_from
    issues = tuple(sorted(set(int(h) for h in params.issues)))
    assert issues and issues[0] == 0
    blocks = block_bounds_of(issues)
    ps: PriceSource = price_source or ConstantPriceSource(bundle.att1.price)
    if params.horizon_days not in (1, 2):
        raise ValueError("horizon_days must be 1 (24h) or 2 (48h + V)")
    forecaster = PointForecaster(
        bundle.daily, bundle.att1, k_load=params.k_load, k_pv=params.k_pv,
        reference_baseline=params.reference_baseline,
    )
    pv_fc = PVIssueForecaster(bundle.daily, bundle.att3)
    paths = IssuedPathLibrary(same_type=params.scenario_same_type)
    pending: list[tuple[date, int, np.ndarray, np.ndarray, np.ndarray | None]] = []

    def release(asof: date, hour: int) -> None:
        now = datetime.combine(asof, clock_time(hour))
        keep = []
        for issued, issued_hour, lp, vp, pp in pending:
            delivered = datetime.combine(issued, clock_time(issued_hour)) + timedelta(hours=24)
            if delivered <= now:
                if issued != WARMUP_START:
                    t = issue_segment(issued_hour)
                    l0, v0 = bundle.truth(issued)
                    l1, v1 = bundle.truth(issued + timedelta(days=1))
                    actual_l = np.concatenate([l0[t:], l1[:t]])
                    actual_v = np.concatenate([v0[t:], v1[:t]])
                    actual_p = None
                    if pp is not None:
                        p0, p1 = ps.truth(issued), ps.truth(issued + timedelta(days=1))
                        actual_p = np.concatenate([p0[t:], p1[:t]])
                    paths.update(issued, issued_hour, actual_l - lp, actual_v - vp,
                                 None if actual_p is None else actual_p - pp)
            else:
                keep.append((issued, issued_hour, lp, vp, pp))
        pending[:] = keep

    checkpoint_file = None if checkpoint_path is None else Path(checkpoint_path)
    run_identity = checkpoint_identity(
        "q3", start, end, record_from, params, varies_price=ps.varies,
        tag=checkpoint_tag,
    )
    out = PeriodResult(soc_end=float(soc_init), label=','.join(map(str, issues)))
    soc = float(soc_init)
    last_completed: date | None = None
    prior_elapsed = 0.0
    if resume:
        if checkpoint_file is None:
            raise ValueError("resume requires checkpoint_path")
        state = load_checkpoint(checkpoint_file, run_identity)
        out = state["result"]
        soc = float(state["soc"])
        paths = state["paths"]
        pending = state["pending"]
        last_completed = state["last_day"]
        prior_elapsed = float(state.get("elapsed_s", 0.0))
    wall = time.perf_counter()
    evaluator = CandidateEvaluator(params.candidate_workers) if params.candidate_reeval else None
    run_start = min(start, WARMUP_START)
    for day_no, d in enumerate(daterange(run_start, end)):
        if last_completed is not None and d <= last_completed:
            continue
        executing = d >= RECORD_START or params.calibration
        execution = DayExecution.empty(d, soc) if executing else None
        G0 = Ga = None
        soc_block = soc
        for h, t0, t1 in blocks:
            release(d, h)
            load_today, pv3_today = forecaster.predict(d)
            load_next, pv3_next = forecaster.predict_next(d)
            pv_today, pv_next = pv_fc.combined_curves(d, h, pv3_today)
            pv_path = pv_fc.path(d, h, pv3_today)
            load_path = np.concatenate([load_today[t0:], load_next[:t0]])
            try:
                price_today, price_next = ps.predict(
                    d, t0,
                    net_load_pred=float((load_path - pv_path).sum()),
                    net_load_next=float((load_next - pv3_next).sum()),
                )
            except TypeError:
                # Compatibility with instrumentation wrappers that expose the
                # original positional interface only.
                price_today, price_next = ps.predict(d, t0)
            price_today = np.asarray(price_today, float)
            price_next = np.asarray(price_next, float)
            price_path = np.concatenate([price_today[t0:], price_next[:t0]])
            pending.append((d, h, load_path.copy(), pv_path.copy(),
                            price_path.copy() if ps.varies else None))
            if not executing:
                continue

            L, PV, P = paths.scenarios(
                datetime.combine(d, clock_time(h)), h,
                load_path, pv_path, price_path if ps.varies else None,
                params.m_scen,
            )
            q = T - t0
            use_48h = params.horizon_days == 2
            terminal_rate = 0.481548
            if ps.varies:
                price_for_v = price_path[None, :] if P is None else P
                q_v = T - t0
                terminal_rate = float(np.mean([row[:30].mean() / .9 if t0 == 0
                                               else row[q_v:q_v + 30].mean() / .9
                                               for row in price_for_v]))
            terminal = ConvexPiecewiseLinear.constant(0).add_linear(-terminal_rate)
            price48 = np.concatenate([price_today, price_next[:t0]])
            # The 48h increment keeps uncertainty only in the current
            # remaining q segments.  Tomorrow is a common point forecast
            # with free purchase; its exact continuous DP V is the terminal
            # value used by the causal executor and candidate score.
            if use_48h:
                terminal = next_day_value(price_next, load_next, pv_next)
                price48 = np.concatenate([price_today, price_next])
            p48_scen = None
            if P is not None:
                p48_scen = np.concatenate([np.tile(price_today[:t0], (P.shape[0], 1)), P], axis=1)
            if use_48h and P is not None:
                p48_scen = np.concatenate([
                    np.tile(price_today[:t0], (P.shape[0], 1)),
                    P[:, :q],
                    np.tile(price_next, (P.shape[0], 1)),
                ], axis=1)
            if h == 0:
                future_l = np.tile(load_next[None, :], (L.shape[0], 1)) if use_48h else L[:, q:]
                future_pv = np.tile(pv_next[None, :], (PV.shape[0], 1)) if use_48h else PV[:, q:]
                saa = solve_saa(price48, L[:, :q], PV[:, :q], future_l, future_pv, soc_block,
                                n_today=q, price_scen=p48_scen,
                                terminal_value=0.0 if use_48h else terminal_rate)
                g_s = np.maximum(saa.G0, 0)
                if evaluator is not None:
                    point = solve_saa_point(
                        price48, load_path[:q], pv_path[:q],
                        load_next if use_48h else load_path[q:],
                        pv_next if use_48h else pv_path[q:], soc_block,
                        n_today=q, terminal_value=0.0 if use_48h else terminal_rate,
                    )
                    g_d = np.maximum(point.G[:q], 0)
                    candidates = [(1-a)*g_s+a*g_d for a in (0., .25, .5, .75, 1.)]
                    scored = evaluator.evaluate(candidates, L[:, :q] if use_48h else L,
                                                 PV[:, :q] if use_48h else PV,
                                                 price_path[:q] if use_48h else price_path,
                                                 soc_block, terminal,
                                                 price_scen=P[:, :q] if use_48h and P is not None else P, n_fixed=q)
                    chosen = candidates[int(np.argmin([x[0] for x in scored]))]
                    hbar = scored[int(np.argmin([x[0] for x in scored]))][1]
                else:
                    chosen = g_s
                    hbar = future_cost(
                        P[:, :q] if use_48h and P is not None else (P if P is not None else price_path[:q] if use_48h else price_path),
                        L[:, :q] if use_48h else L,
                        PV[:, :q] if use_48h else PV,
                        chosen, terminal, n_fixed=q,
                    )
                G0 = chosen.copy()
                Ga = G0.copy()
                g_execute = G0
            else:
                assert G0 is not None and Ga is not None
                future_l = np.tile(load_next[None, :], (L.shape[0], 1)) if use_48h else L[:, q:]
                future_pv = np.tile(pv_next[None, :], (PV.shape[0], 1)) if use_48h else PV[:, q:]
                rj = solve_readjust(t0, price48, G0, L[:, :q], PV[:, :q], future_l, future_pv,
                                    soc_block, price_scen=p48_scen,
                                    terminal_value=0.0 if use_48h else terminal_rate)
                g_s = np.maximum(rj.Ga[t0:], 0)
                if evaluator is not None:
                    rj_point = solve_readjust(
                        t0, price48, G0, load_path[:q][None, :], pv_path[:q][None, :],
                        load_next[None, :] if use_48h else load_path[q:][None, :],
                        pv_next[None, :] if use_48h else pv_path[q:][None, :], soc_block,
                        terminal_value=0.0 if use_48h else terminal_rate,
                    )
                    g_d = np.maximum(rj_point.Ga[t0:], 0)
                    candidates = [(1-a)*g_s+a*g_d for a in (0., .25, .5, .75, 1.)]
                    scored = evaluator.evaluate(candidates, L[:, :q] if use_48h else L,
                                                 PV[:, :q] if use_48h else PV,
                                                 price_path[:q] if use_48h else price_path,
                                                 soc_block, terminal,
                                                 price_scen=P[:, :q] if use_48h and P is not None else P,
                                                 base_plan=G0[t0:], n_fixed=q)
                    winner = int(np.argmin([x[0] for x in scored]))
                    chosen, hbar = candidates[winner], scored[winner][1]
                else:
                    chosen = g_s
                    hbar = future_cost(
                        P[:, :q] if use_48h and P is not None else (P if P is not None else price_path[:q] if use_48h else price_path),
                        L[:, :q] if use_48h else L,
                        PV[:, :q] if use_48h else PV,
                        chosen, terminal, n_fixed=q,
                    )
                # Only this block is committed.  The later part remains a temporary decision.
                Ga[t0:t1] = chosen[:t1-t0]
                g_execute = Ga
            dp = DPValueExecutor(hbar, price=price_today[t0:]).prepare(t0)
            load_true, pv_true = bundle.truth(d)
            run_day_dp(d, g_execute, soc_block, load_true, pv_true, ps.truth(d), dp,
                       t_start=t0, t_end=t1, out=execution,
                       price_fn=(lambda j, p=ps.truth(d): float(p[j])) if ps.varies else None)
            soc_block = float(execution.S[t1-1])

        if not executing:
            if checkpoint_file is not None:
                save_checkpoint(
                    checkpoint_file, run_identity=run_identity, last_day=d,
                    soc=soc, result=out, pending=pending, paths=paths,
                    elapsed_s=prior_elapsed + time.perf_counter() - wall,
                )
            if progress and (day_no + 1) % progress == 0:
                print(f'  [{d.isoformat()}] 已完成 {day_no + 1} 天，'
                      f'累计运行 {prior_elapsed + time.perf_counter() - wall:.1f}s，', flush=True)
            continue
        assert execution is not None and G0 is not None and Ga is not None
        price_true = ps.truth(d)
        cost = cost_q3(price_true, G0, Ga, execution.E)
        if d >= record_from:
            out.days.append(DayResult(d, G0, Ga, execution.C, execution.D, execution.S, execution.E,
                execution.W, execution.soc_start, cost['plan'], cost['curtail_penalty'], cost['extra'],
                cost['emergency'], execution.R, float(price_true @ G0), price_true if ps.varies else None))
        soc = execution.soc_end
        if checkpoint_file is not None:
            save_checkpoint(
                checkpoint_file, run_identity=run_identity, last_day=d,
                soc=soc, result=out, pending=pending, paths=paths,
                elapsed_s=prior_elapsed + time.perf_counter() - wall,
            )
        if progress and (day_no + 1) % progress == 0:
            print(f'  [{d.isoformat()}] 已完成 {day_no + 1} 天，'
                  f'累计运行 {prior_elapsed + time.perf_counter() - wall:.1f}s，', flush=True)
    if evaluator is not None:
        evaluator.close()
    out.soc_end, out.runtime_s = soc, prior_elapsed + time.perf_counter() - wall
    return out


def validate(res: PeriodResult, bundle: Bundle, issues: tuple[int, ...]) -> None:
    """复核全部不变量：提交规则、平衡、SOC 界与跨日连续、四项费用分解。"""
    default_price = np.asarray(bundle.att1.price, dtype=float)
    first_block_end = block_bounds_of(tuple(sorted(set(issues))))[0][2]
    for prev, cur in zip(res.days, res.days[1:]):
        assert (cur.day - prev.day).days != 1 or abs(cur.soc_start - prev.soc_end) < 1e-9, (
            f"{cur.day} 的 0:00 储电量与前一日 24:00 不连续"
        )
    for day in res.days:
        price = default_price if day.price is None else np.asarray(day.price, dtype=float)
        assert day.Ga.shape == (T,) and day.G0.shape == (T,)
        assert day.Ga.min() >= -1e-9, f"{day.day} 出现负的调整购电量"
        assert np.abs(day.Ga[:first_block_end] - day.G0[:first_block_end]).max() < 1e-9, (
            f"{day.day} 首个提交区间之前的段被改动"
        )
        assert np.minimum(day.E, day.W).max() < 1e-9, f"{day.day} 存在 E_t·W_t ≠ 0"
        assert np.minimum(day.C, day.D).max() < 1e-6, f"{day.day} 存在同时充放电"
        assert day.S.min() >= SOC_MIN - 1e-6 and day.S.max() <= SOC_MAX + 1e-6
        prev_soc = np.concatenate([[day.soc_start], day.S[:-1]])
        assert np.abs(day.S - prev_soc - ETA * day.C + day.D / ETA).max() < 1e-6
        cost = cost_q3(price, day.G0, day.Ga, day.E)
        assert abs(cost["total"] - day.total_cost) < 1e-6, f"{day.day} 费用分解不闭合"
        assert abs(cost["plan"] - day.plan_cost) < 1e-6
        assert abs(cost["curtail_penalty"] - day.curtail_cost) < 1e-6
        assert abs(cost["extra"] - day.extra_cost) < 1e-6
        assert abs(cost["emergency"] - 5.0 * float(price @ day.E)) < 1e-6
        load_true, pv_true = bundle.truth(day.day)
        # 执行层用的是提交值 G^a：结算与执行必须对同一个购电量
        balance = day.Ga + day.E + pv_true + day.D - load_true - day.C - day.W
        assert np.abs(balance).max() < 1e-6, f"{day.day} 平衡式残差过大"
    total = sum(d.total_cost for d in res.days)
    assert abs(res.total_cost - total) < 1e-6


def print_period_summary(res: PeriodResult, label: str = "全年") -> None:
    share = 100.0 * res.emergency_kwh / res.plan_kwh if res.plan_kwh else 0.0
    print(f"—— {label}汇总（{len(res.days)} 天，预报时刻 {res.label}）——")
    print(f"  总费用        {res.total_cost:14.2f} 元")
    print(f"    计划费      {res.plan_cost:14.2f} 元  = Σ p·min(G⁰,Gᵃ)")
    print(f"    减购违约费  {res.curtail_cost:14.2f} 元  = Σ 0.5p·(G⁰−Gᵃ)⁺")
    print(f"    增购费      {res.extra_cost:14.2f} 元  = Σ 1.5p·(Gᵃ−G⁰)⁺")
    print(f"    紧急购电费  {res.emergency_cost:14.2f} 元  = Σ 5p·E")
    print(f"  计划购电量    {res.plan_kwh:14.2f} kWh")
    print(f"  实际购电量    {res.purchase_kwh:14.2f} kWh（Σ Gᵃ）")
    print(f"  紧急购电量    {res.emergency_kwh:14.2f} kWh（占计划购电量 {share:.3f}%）")
    print(f"  平均每段调整量{res.mean_adjust_kwh:14.2f} kWh")
    print(
        f"  调整相关净费用{res.adjust_net_cost:14.2f} 元"
        f"（购电侧 {res.plan_cost + res.curtail_cost + res.extra_cost:.2f}"
        f" − 不调整基准 Σp·G⁰ {res.plan_only_cost:.2f}）"
    )
    print(f"  发生紧急购电的天数  {res.emergency_days} / {len(res.days)}")
    print(f"  24:00 储电量均值    {np.mean([d.soc_end for d in res.days]):.2f} kWh")


def _parse_issues(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(p) for p in text.replace("，", ",").split(",") if p.strip()}))


def main(argv: list[str] | None = None) -> PeriodResult:
    parser = argparse.ArgumentParser(
        description="问 3：附件 3 预报 + 固定时点重优化 + 调整购电结算"
    )
    parser.add_argument("--tune", action="store_true", help="显式运行独立的一月增量校准")
    parser.add_argument("--skip-tuning", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--k-load", type=int, default=None)
    parser.add_argument("--k-pv", type=int, default=None)
    parser.add_argument("--m-scen", type=int, default=M_SCEN)
    parser.add_argument("--step-minutes", type=int, default=10,
                        help="仅 legacy 滚动 LP 实验使用；DP 主流程固定 10 分钟")
    parser.add_argument("--candidate-workers", type=int, default=1,
                        help="并行重评独立候选的进程数；单次回测最多有效使用 5 个")
    parser.add_argument("--candidate-reeval", action="store_true")
    parser.add_argument("--horizon-days", choices=(1, 2), type=int, default=1)
    parser.add_argument("--legacy-means", action="store_true")
    parser.add_argument("--same-type-scenarios", action="store_true")
    parser.add_argument("--issues", type=_parse_issues, default=ISSUE_HOURS,
                        help="可用预报时刻子集，如 0,6,12（必须含 0）")
    parser.add_argument("--variants", action="store_true",
                        help="依次跑 0 / 0,6 / 0,6,12 / 0,6,12,18 四个变体并打印对比表")
    parser.add_argument("--start", type=date.fromisoformat, default=RECORD_START)
    parser.add_argument("--end", type=date.fromisoformat, default=PERIOD_END)
    parser.add_argument("--out", default="results/result3.xlsx")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--progress", type=int, default=30)
    args = parser.parse_args(argv)

    wall = time.perf_counter()
    bundle = load_bundle()
    params = Params(
        m_scen=args.m_scen,
        step_minutes=args.step_minutes,
        issues=args.issues,
        candidate_workers=args.candidate_workers,
        candidate_reeval=args.candidate_reeval,
        horizon_days=args.horizon_days,
        reference_baseline=not args.legacy_means,
        scenario_same_type=args.same_type_scenarios,
        calibration=False,
    )

    if args.k_load is not None:
        params.k_load = args.k_load
    if args.k_pv is not None:
        params.k_pv = args.k_pv
    if args.tune and args.k_load is None and args.k_pv is None:
        from .run_q2 import Params as Q2Params
        from .run_q2 import load_bundle as load_bundle_q2
        from .tuning import select_k

        print("在 1 月按问 2 流程（实际购电费）选择 K_L, K_P …", flush=True)
        picked = select_k(
            load_bundle_q2(),
            base=Q2Params(
                m_scen=params.m_scen,
                candidate_workers=params.candidate_workers,
                candidate_reeval=params.candidate_reeval,
                horizon_days=params.horizon_days,
                reference_baseline=False,
                scenario_same_type=params.scenario_same_type,
            ),
        )
        params.k_load, params.k_pv = picked.k_load, picked.k_pv
        params.reference_baseline = False
        params.calibration = False
        print(
            f"选定 K_L={params.k_load}, K_P={params.k_pv}，"
            f"{picked.window} 总费用 {picked.cost:.2f} 元\n"
        )
    else:
        source = "命令行指定" if (args.k_load or args.k_pv) else "论文基线默认"
        print(f"使用参数 K_L={params.k_load}, K_P={params.k_pv}（{source}）\n")

    variants = VARIANTS if args.variants else [tuple(params.issues)]
    results: list[PeriodResult] = []
    main_issues = tuple(sorted(set(params.issues)))
    for issues in variants:
        print(
            f"回测 {WARMUP_START.isoformat()} → {args.end.isoformat()}"
            f"（{args.start.isoformat()} 起计入结果），预报时刻 {issues}",
            flush=True,
        )
        res = run_period(
            WARMUP_START,
            args.end,
            Params(
                k_load=params.k_load,
                k_pv=params.k_pv,
                m_scen=params.m_scen,
                step_minutes=params.step_minutes,
                issues=issues,
                candidate_workers=params.candidate_workers,
                candidate_reeval=params.candidate_reeval,
                horizon_days=params.horizon_days,
                reference_baseline=params.reference_baseline,
                scenario_same_type=params.scenario_same_type,
                calibration=params.calibration,
            ),
            bundle,
            record_from=args.start,
            soc_init=SOC_INIT,
            progress=args.progress,
        )
        validate(res, bundle, issues)
        results.append(res)
        print(f"  用时 {res.runtime_s:.1f}s\n", flush=True)

    chosen = next(
        (r for r in results if r.label == ",".join(str(h) for h in main_issues)), results[-1]
    )

    if not args.no_write:
        path = write_result3(chosen.days, args.out)
        print(f"已写出 {path}（{len(chosen.days)} 行）\n")

    for d in REPORT_DAYS:
        day = chosen.by_date(d)
        if day is not None:
            print_day_summary_q3(summarize_day_q3(day, bundle.att1.price))

    print_period_summary(chosen, label=f"{args.start.isoformat()}→{args.end.isoformat()} ")
    if len(results) > 1:
        print("\n—— 变体对比（是否引入其他时刻预报）——")
        print(variant_table([r.row() for r in results]))
    print(f"\n总运行时间 {time.perf_counter() - wall:.1f}s")
    return chosen


if __name__ == "__main__":
    main()
