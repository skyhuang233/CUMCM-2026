"""问 2：逐日 walk-forward 回测（SAA 购电 + 储能滚动控制 + 真值结算）。

每个决策日 0:00：
  1. 预测/场景层用日期 < d 的附件 2 数据给出当天点预测、次日点预测与 M 个等权场景；
  2. SAA 两阶段 LP 冻结当天计划购电量 G^0；
  3. 执行层用附件 2 真值逐段跑储能滚动控制，得到执行的充放电、储电量、紧急购电；
  4. 结算层按 p_t G^0_t + 5 p_t E_t 计费，残差入库，24:00 储电量结转次日。

1 月默认只重建因果预测与完整路径残差；2–12 月连续执行。显式
``calibration=True`` 仅供独立的一月增量参数评估，末库存不带入二月基线。
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
)
from .checkpoint import identity as checkpoint_identity, load as load_checkpoint, save as save_checkpoint
from .executor import DayExecution, run_day_dp, DPValueExecutor
from .value_dp import ConvexPiecewiseLinear, future_cost, next_day_value
from .parallel_eval import CandidateEvaluator
from .forecast import PointForecaster
from .forecast_price import ConstantPriceSource, PriceSource, horizon_price
from .optimizer import solve_saa, solve_saa_point
from .results import print_day_summary, summarize_day, write_result2
from .scenarios import IssuedPathLibrary, M_SCEN
from .settlement import cost_q2

WARMUP_START = date(2025, 1, 1)
RECORD_START = date(2025, 2, 1)
PERIOD_END = date(2025, 12, 31)
REPORT_DAYS = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]


@dataclass
class Params:
    """一次回测的全部超参数。"""

    k_load: int = 35
    k_pv: int = 7
    m_scen: int = M_SCEN
    # Kept for legacy StorageRollingSolver experiments only.  DP execution is
    # causal at the native 10-minute cadence and deliberately ignores it.
    step_minutes: int = 10
    horizon_days: int = 1  # 1=24h 线性续存基线；2=48h+V(e) 增量
    candidate_workers: int = 1
    candidate_reeval: bool = False
    reference_baseline: bool = True
    scenario_same_type: bool = False
    calibration: bool = False


@dataclass
class DayResult:
    """一天的计划、执行与费用。"""

    day: date
    G0: np.ndarray
    C: np.ndarray
    D: np.ndarray
    S: np.ndarray
    E: np.ndarray
    W: np.ndarray
    soc_start: float
    plan_cost: float
    emergency_cost: float
    R: np.ndarray | None = None
    point_solution: object | None = None  # 点预测解，仅供论文对照
    price: np.ndarray | None = None  # 该日结算电价（None = 附件 1 常数电价）

    @property
    def total_cost(self) -> float:
        return self.plan_cost + self.emergency_cost

    @property
    def soc_end(self) -> float:
        return float(self.S[-1])


@dataclass
class PeriodResult:
    """一段日期区间的回测汇总。"""

    days: list[DayResult] = field(default_factory=list)
    soc_end: float = SOC_INIT
    runtime_s: float = 0.0

    @property
    def plan_cost(self) -> float:
        return sum(d.plan_cost for d in self.days)

    @property
    def emergency_cost(self) -> float:
        return sum(d.emergency_cost for d in self.days)

    @property
    def total_cost(self) -> float:
        return self.plan_cost + self.emergency_cost

    @property
    def plan_kwh(self) -> float:
        return float(sum(float(d.G0.sum()) for d in self.days))

    @property
    def emergency_kwh(self) -> float:
        return float(sum(float(d.E.sum()) for d in self.days))

    @property
    def emergency_days(self) -> int:
        return sum(1 for d in self.days if d.E.max() > 1e-6)

    def by_date(self, d: date) -> DayResult | None:
        for day in self.days:
            if day.day == d:
                return day
        return None


@dataclass
class Bundle:
    """只读输入数据，供多次回测复用（调参时避免重复读盘）。"""

    att1: Attachment1
    daily: DailySeries
    _row: dict[date, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._row = {d: i for i, d in enumerate(self.daily.dates)}

    def truth(self, d: date) -> tuple[np.ndarray, np.ndarray]:
        """日期 d 的负载、光伏真值（只在执行与结算时调用，不进入预测路径）。"""
        i = self._row[d]
        return self.daily.load_kwh[i], self.daily.pv_kwh[i]


def load_bundle(
    path1: str = "data/附件1.xlsx", path2: str = "data/附件2.xlsx"
) -> Bundle:
    return Bundle(att1=load_attachment1(path1), daily=load_attachment2(path2))


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def run_period(
    start: date,
    end: date,
    params: Params,
    bundle: Bundle,
    *,
    record_from: date | None = None,
    soc_init: float = SOC_INIT,
    progress: int = 0,
    point_days: set[date] | None = None,
    price_source: PriceSource | None = None,
    checkpoint_path: str | None = None,
    resume: bool = False,
    checkpoint_tag: str | None = None,
) -> PeriodResult:
    """从 start 到 end 逐日 walk-forward 回测；只记录 record_from 起的日子。

    `point_days` 中的日期额外求一次点预测解（M=1、残差为零、与当前配置同一时域），
    只挂在结果上供论文对照，不参与执行。

    `price_source` 缺省为附件 1 的常数电价（问 2 本体）；问 4 传入附件 4 的电价预测器，
    此时每天的电价曲线来自决策时刻的点预测、场景带电价残差、结算与执行用真值。
    """
    record_from = start if record_from is None else record_from
    if params.horizon_days not in (1, 2):
        raise ValueError("horizon_days must be 1 (baseline) or 2 (48h increment)")
    point_days = point_days or set()
    ps: PriceSource = price_source or ConstantPriceSource(bundle.att1.price)
    forecaster = PointForecaster(
        bundle.daily, bundle.att1, k_load=params.k_load, k_pv=params.k_pv,
        reference_baseline=params.reference_baseline,
    )
    if ps.varies and getattr(ps, "net_load", None) is None:
        # Direct callers of run_period need the same causal feature as the
        # CLI Q4 factory.  Historical rows are reconstructed by the selected
        # forecaster, never by a realised full-day net load.
        def _historical_net(day: date, segment: int = 0) -> float:
            hist_l, hist_pv = forecaster.predict(day)
            segment = int(segment)
            if segment and day in bundle._row:
                actual_l, actual_pv = bundle.truth(day)
                return float(np.sum(actual_l[:segment] - actual_pv[:segment]) +
                             np.sum(hist_l[segment:] - hist_pv[segment:]))
            return float(np.sum(hist_l - hist_pv))
        ps.net_load = _historical_net
    # Every residual is a complete, paired 24-hour issued path.  Keep the
    # publication timestamp, rather than a date, so an 18:00 publication does
    # not leak into the following day's 0:00/6:00 decisions.
    pending: list[tuple[datetime, np.ndarray, np.ndarray, np.ndarray | None]] = []
    library = IssuedPathLibrary(same_type=params.scenario_same_type)

    def release(asof: datetime) -> None:
        """Release only paths whose complete 24-hour delivery has happened."""
        keep = []
        for issued, lp, vp, pp in pending:
            if issued + timedelta(days=1) <= asof:
                if issued.date() != WARMUP_START and issued.date() + timedelta(days=1) in bundle._row:
                    t = issued.hour * 6
                    l0, v0 = bundle.truth(issued.date())
                    l1, v1 = bundle.truth(issued.date() + timedelta(days=1))
                    actual_l = np.r_[l0[t:], l1[:t]]
                    actual_v = np.r_[v0[t:], v1[:t]]
                    if pp is None:
                        actual_p = None
                    else:
                        actual_p = np.r_[ps.truth(issued.date())[t:], ps.truth(issued.date() + timedelta(days=1))[:t]]
                    library.update(issued.date(), issued.hour, actual_l - lp, actual_v - vp,
                                   None if actual_p is None else actual_p - pp)
            else:
                keep.append((issued, lp, vp, pp))
        pending[:] = keep


    checkpoint_file = None if checkpoint_path is None else Path(checkpoint_path)
    run_identity = checkpoint_identity(
        "q2", start, end, record_from, params, varies_price=ps.varies,
        tag=checkpoint_tag,
    )
    out = PeriodResult(soc_end=float(soc_init))
    soc = float(soc_init)
    last_completed: date | None = None
    prior_elapsed = 0.0
    if resume:
        if checkpoint_file is None:
            raise ValueError("resume requires checkpoint_path")
        state = load_checkpoint(checkpoint_file, run_identity)
        out = state["result"]
        soc = float(state["soc"])
        library = state["paths"]
        pending = state["pending"]
        last_completed = state["last_day"]
        prior_elapsed = float(state.get("elapsed_s", 0.0))
    t_start = time.perf_counter()
    candidate_evaluator = CandidateEvaluator(params.candidate_workers) if params.candidate_reeval else None

    def issued_path(d: date, seg: int):
        """Causal 24h point path from a publication at ``d, seg``."""
        load_today, pv_today = forecaster.predict(d)
        load_next, pv_next = forecaster.predict(d + timedelta(days=1), asof=d)
        load = np.r_[load_today[seg:], load_next[:seg]]
        pv = np.r_[pv_today[seg:], pv_next[:seg]]
        observed_net = 0.0
        if seg:
            actual_l, actual_v = bundle.truth(d)
            observed_net = float(np.sum(actual_l[:seg] - actual_v[:seg]))
        next_net = float(np.sum(load_next - pv_next))
        today_p, next_p = ps.predict(
            d, seg, net_load_pred=observed_net + float(np.sum(load_today[seg:] - pv_today[seg:])),
            net_load_next=next_net,
        )
        return load, pv, np.r_[np.asarray(today_p)[seg:], np.asarray(next_p)[:seg]]

    # Callers may request a February-only result window; January's issued
    # predictions must nevertheless be rebuilt before the first decision.
    run_start = min(start, WARMUP_START)
    for i, d in enumerate(daterange(run_start, end)):
        if last_completed is not None and d <= last_completed:
            continue
        release(datetime.combine(d, clock_time.min))
        load_pred, pv_pred, price_today = issued_path(d, 0)
        pending.append((datetime.combine(d, clock_time.min), load_pred.copy(), pv_pred.copy(),
                        price_today.copy() if ps.varies else None))

        # January is a prediction/residual warm-up only.  It nevertheless
        # publishes all four issued paths so Q4-2's February intraday pool is
        # not silently reduced to a single point path.
        if d < RECORD_START and not params.calibration:
            if ps.varies:
                for segment in (36, 72, 108):
                    issued = datetime.combine(d, clock_time(segment // 6))
                    l_path, pv_path, p_path = issued_path(d, segment)
                    pending.append((issued, l_path, pv_path, p_path))
            if checkpoint_file is not None:
                save_checkpoint(
                    checkpoint_file, run_identity=run_identity, last_day=d,
                    soc=soc, result=out, pending=pending, paths=library,
                    elapsed_s=prior_elapsed + time.perf_counter() - t_start,
                )
            if progress and (i + 1) % progress == 0:
                print(
                    f"  [{d.isoformat()}] 已完成 {i + 1} 天，"
                    f"用时 {prior_elapsed + time.perf_counter() - t_start:.1f}s",
                    flush=True,
                )
            continue

        L_scen, PV_scen, price_scen = library.scenarios(
            datetime.combine(d, clock_time.min), 0, load_pred, pv_pred,
            price_today if ps.varies else None, params.m_scen,
        )
        terminal_rate = 0.481548
        if ps.varies:
            price_for_v = price_today[None, :] if price_scen is None else price_scen
            terminal_rate = float(np.mean([np.mean(row[:30]) / .9 for row in price_for_v]))
        terminal = ConvexPiecewiseLinear.constant(0.0).add_linear(-terminal_rate)
        load_next, pv_next = forecaster.predict(d + timedelta(days=1), asof=d)
        _, price_next = ps.predict(d, 0, net_load_next=float(np.sum(load_next - pv_next)))
        if params.horizon_days == 2:
            price_opt = np.r_[price_today, price_next]
            future_l, future_pv = load_next, pv_next
            terminal_for_eval = next_day_value(price_next, load_next, pv_next)
            price_scen_opt = None if price_scen is None else np.concatenate(
                [price_scen, np.tile(np.asarray(price_next)[None, :], (price_scen.shape[0], 1))], axis=1
            )
        else:
            price_opt = price_today
            future_l = future_pv = np.zeros(0)
            terminal_for_eval = terminal
            price_scen_opt = price_scen
        saa = solve_saa(
            price_opt,
            L_scen,
            PV_scen,
            future_l,
            future_pv,
            soc,
            n_today=T,
            price_scen=price_scen_opt,
            terminal_value=0.0 if params.horizon_days == 2 else terminal_rate,
        )
        point = None
        if params.candidate_reeval or d in point_days:
            point = solve_saa_point(
                price_opt, load_pred, pv_pred, future_l, future_pv, soc,
                terminal_value=0.0 if params.horizon_days == 2 else terminal_rate,
            )
        g_s = np.maximum(saa.G0, 0.0)
        p_eval_scen = price_scen
        if params.candidate_reeval:
            assert point is not None
            g_d = np.maximum(point.G[:T], 0.0)
            candidates = [((1-a)*g_s+a*g_d) for a in (0., .25, .5, .75, 1.)]
            scored = candidate_evaluator.evaluate(candidates, L_scen, PV_scen, price_today, soc, terminal_for_eval, price_scen=p_eval_scen)
            winner = int(np.argmin([score for score, _ in scored]))
            g0, hbar = candidates[winner], scored[winner][1]
        else:
            g0 = g_s
            hbar = future_cost(
                price_today if p_eval_scen is None else p_eval_scen,
                L_scen, PV_scen, g0, terminal_for_eval,
            )

        load_true, pv_true = bundle.truth(d)
        price_true = ps.truth(d)
        price_fn = (lambda t, p=price_true: float(p[t])) if ps.varies else None
        # Q4-2 has no adjustment authority, but its price information and
        # future-cost function are refreshed at all four publication times.
        updates = (0, 36, 72, 108, T) if ps.varies else (0, T)
        execution = DayExecution.empty(d, soc)
        soc_block = soc
        for b0, b1 in zip(updates[:-1], updates[1:]):
            p_value = price_today
            if b0:
                now = datetime.combine(d, clock_time(b0 // 6))
                release(now)
                path_l, path_pv, path_price = issued_path(d, b0)
                pending.append((now, path_l.copy(), path_pv.copy(), path_price.copy()))
                L_update, PV_update, P_update = library.scenarios(now, b0 // 6, path_l, path_pv, path_price, params.m_scen)
                p_value, _ = ps.predict(d, b0)
                p_value = np.asarray(p_value, float)
                # The 24h issued path crosses midnight at q segments.  Its
                # next-day 0:00--5:00 price gives Q4's refreshed v_tau.
                q = T - b0
                rates = P_update[:, q:q + 30].mean(axis=1) / .9 if P_update is not None else np.array([path_price[q:q + 30].mean() / .9])
                if params.horizon_days == 2:
                    # The approved 48h increment prices midnight inventory by
                    # a free-purchase next-day value function, not by -vS.
                    next_l, next_pv = forecaster.predict(d + timedelta(days=1), asof=d)
                    _, next_price = ps.predict(d, b0, net_load_next=float(np.sum(next_l - next_pv)))
                    remainder_terminal = next_day_value(next_price, next_l, next_pv)
                    eval_l = L_update[:, :q]
                    eval_pv = PV_update[:, :q]
                    eval_price = path_price[:q]
                    eval_price_scen = None if P_update is None else P_update[:, :q]
                else:
                    remainder_terminal = ConvexPiecewiseLinear.constant(0.0).add_linear(-float(rates.mean()))
                    eval_l = L_update
                    eval_pv = PV_update
                    eval_price = path_price
                    eval_price_scen = P_update
                hbar = future_cost(
                    eval_price if eval_price_scen is None else eval_price_scen,
                    eval_l, eval_pv, g0[b0:], remainder_terminal, n_fixed=q,
                )
            dp_executor = DPValueExecutor(hbar, price=p_value[b0:]).prepare(b0)
            run_day_dp(d, g0, soc_block, load_true, pv_true, price_true, dp_executor,
                       t_start=b0, t_end=b1, out=execution, price_fn=price_fn)
            soc_block = float(execution.S[b1 - 1])
        plan_cost, emergency_cost = cost_q2(price_true, g0, execution.E)
        if d >= record_from:
            out.days.append(
                DayResult(
                    day=d,
                    G0=g0,
                    C=execution.C,
                    D=execution.D,
                    S=execution.S,
                    E=execution.E,
                    W=execution.W,
                    R=execution.R,
                    soc_start=execution.soc_start,
                    plan_cost=plan_cost,
                    emergency_cost=emergency_cost,
                    point_solution=point if d in point_days else None,
                    price=price_true if ps.varies else None,
                )
            )
        soc = execution.soc_end
        if checkpoint_file is not None:
            save_checkpoint(
                checkpoint_file, run_identity=run_identity, last_day=d,
                soc=soc, result=out, pending=pending, paths=library,
                elapsed_s=prior_elapsed + time.perf_counter() - t_start,
            )
        if progress and (i + 1) % progress == 0:
            print(
                f"  [{d.isoformat()}] 已完成 {i + 1} 天，"
                f"用时 {prior_elapsed + time.perf_counter() - t_start:.1f}s",
                flush=True,
            )

    out.soc_end = soc
    out.runtime_s = prior_elapsed + time.perf_counter() - t_start
    if candidate_evaluator is not None:
        candidate_evaluator.close()
    return out


def validate(res: PeriodResult, bundle: Bundle) -> None:
    """回测结束后复核全部不变量：平衡、无同时充放电、SOC 界与跨日连续、费用分解。"""
    default_price = np.asarray(bundle.att1.price, dtype=float)
    for prev, cur in zip(res.days, res.days[1:]):
        assert (cur.day - prev.day).days != 1 or abs(cur.soc_start - prev.soc_end) < 1e-9, (
            f"{cur.day} 的 0:00 储电量与前一日 24:00 不连续"
        )
    for day in res.days:
        price = default_price if day.price is None else np.asarray(day.price, dtype=float)
        assert np.minimum(day.E, day.W).max() < 1e-9, f"{day.day} 存在 E_t·W_t ≠ 0"
        assert np.minimum(day.C, day.D).max() < 1e-6, f"{day.day} 存在同时充放电"
        assert day.S.min() >= SOC_MIN - 1e-6 and day.S.max() <= SOC_MAX + 1e-6
        assert day.G0.min() >= -1e-9 and day.E.min() >= -1e-9
        prev_soc = np.concatenate([[day.soc_start], day.S[:-1]])
        assert np.abs(day.S - prev_soc - ETA * day.C + day.D / ETA).max() < 1e-6
        assert abs(day.plan_cost - float(price @ day.G0)) < 1e-6
        assert abs(day.emergency_cost - 5.0 * float(price @ day.E)) < 1e-6
        load_true, pv_true = bundle.truth(day.day)
        balance = day.G0 + day.E + pv_true + day.D - load_true - day.C - day.W
        assert np.abs(balance).max() < 1e-6, f"{day.day} 平衡式残差过大"
    total = sum(d.plan_cost for d in res.days) + sum(d.emergency_cost for d in res.days)
    assert abs(res.total_cost - total) < 1e-6


def print_period_summary(res: PeriodResult, label: str = "全年") -> None:
    share = 100.0 * res.emergency_kwh / res.plan_kwh if res.plan_kwh else 0.0
    print(f"—— {label}汇总（{len(res.days)} 天）——")
    print(f"  总费用      {res.total_cost:14.2f} 元")
    print(f"    计划购电费{res.plan_cost:14.2f} 元")
    print(f"    紧急购电费{res.emergency_cost:14.2f} 元")
    print(f"  计划购电量  {res.plan_kwh:14.2f} kWh")
    print(f"  紧急购电量  {res.emergency_kwh:14.2f} kWh（占计划购电量 {share:.3f}%）")
    print(f"  发生紧急购电的天数  {res.emergency_days} / {len(res.days)}")
    print(f"  24:00 储电量均值    {np.mean([d.soc_end for d in res.days]):.2f} kWh")


def main(argv: list[str] | None = None) -> PeriodResult:
    parser = argparse.ArgumentParser(description="问 2：SAA 购电 + 储能滚动控制逐日回测")
    parser.add_argument("--skip-tuning", action="store_true", help="兼容旧参数；参考基线默认不调参")
    parser.add_argument("--tune", action="store_true", help="显式启用 1 月参数选择（消融）")
    parser.add_argument("--k-load", type=int, default=None)
    parser.add_argument("--k-pv", type=int, default=None)
    parser.add_argument("--m-scen", type=int, default=M_SCEN)
    parser.add_argument("--horizon-days", type=int, choices=(1, 2), default=1)
    parser.add_argument("--candidate-reeval", action="store_true", help="启用候选计划重评增量")
    parser.add_argument("--legacy-means", action="store_true", help="使用旧日类型均值预测增量")
    parser.add_argument("--same-type-scenarios", action="store_true", help="使用同类型残差池增量")
    parser.add_argument("--step-minutes", type=int, default=10,
                        help="仅 legacy 滚动 LP 实验使用；DP 主流程固定 10 分钟")
    parser.add_argument("--candidate-workers", type=int, default=1,
                        help="并行重评独立候选的进程数；单次回测最多有效使用 5 个")
    parser.add_argument("--start", type=date.fromisoformat, default=RECORD_START)
    parser.add_argument("--end", type=date.fromisoformat, default=PERIOD_END)
    parser.add_argument("--out", default="results/result2.xlsx")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--progress", type=int, default=30)
    args = parser.parse_args(argv)

    wall = time.perf_counter()
    bundle = load_bundle()
    params = Params(
        m_scen=args.m_scen,
        horizon_days=args.horizon_days,
        step_minutes=args.step_minutes,
        candidate_workers=args.candidate_workers,
        candidate_reeval=args.candidate_reeval,
        reference_baseline=not args.legacy_means,
        scenario_same_type=args.same_type_scenarios,
        # Tuning itself creates an isolated calibration run; the following
        # formal February-onward run must retain the prescribed 6000 kWh
        # inventory and therefore never execute January dispatch.
        calibration=False,
    )

    if args.k_load is not None:
        params.k_load = args.k_load
    if args.k_pv is not None:
        params.k_pv = args.k_pv
    if args.tune and not args.skip_tuning and args.k_load is None and args.k_pv is None:
        from .tuning import select_k

        print("在 1 月（1–7 日预热不计分）按实际购电费选择 K_L, K_P …", flush=True)
        params.reference_baseline = False
        picked = select_k(bundle, base=params)
        params.k_load, params.k_pv = picked.k_load, picked.k_pv
        print(
            f"选定 K_L={params.k_load}, K_P={params.k_pv}，"
            f"{picked.window} 总费用 {picked.cost:.2f} 元\n"
        )
    else:
        source = "命令行指定" if (args.k_load or args.k_pv) else "默认（--skip-tuning）"
        print(f"使用参数 K_L={params.k_load}, K_P={params.k_pv}（{source}）\n")

    print(f"回测 {WARMUP_START.isoformat()} → {args.end.isoformat()}"
          f"（{args.start.isoformat()} 起计入结果）", flush=True)
    res = run_period(
        WARMUP_START,
        args.end,
        params,
        bundle,
        record_from=args.start,
        soc_init=SOC_INIT,
        progress=args.progress,
        point_days=set(REPORT_DAYS),
    )

    validate(res, bundle)

    if not args.no_write:
        path = write_result2(res.days, args.out)
        print(f"\n已写出 {path}（{len(res.days)} 行）\n")

    for d in REPORT_DAYS:
        day = res.by_date(d)
        if day is not None:
            print_day_summary(summarize_day(day, bundle.att1.price))

    print_period_summary(res, label=f"{args.start.isoformat()}→{args.end.isoformat()} ")
    print(f"\n求解用时 {res.runtime_s:.1f}s，总运行时间 {time.perf_counter() - wall:.1f}s")
    return res


if __name__ == "__main__":
    main()
