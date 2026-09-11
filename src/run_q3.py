"""问 3：附件 3 预报插值、固定时点滚动重优化与调整购电结算。

每个决策日：
  1. 0:00 用附件 3 的 0:00 预报（线性插值到 10 分钟）作当天光伏点预测，负载沿用问 2 的
     $K_L$ 同类型日均值；场景取 0:00 残差库，SAA 两阶段 LP 冻结计划购电量 $G^0$；
  2. 在 6:00 / 12:00 / 18:00 用新预报与当前执行 SOC 求解重优化 LP，决定调整购电量
     $G^a = G^0 + \\Delta^+ - \\Delta^-$，**只提交到下一个预报时刻**；
  3. 储能仍由问 2 的滚动 LP 逐段控制，其固定购电量数组 = 已提交段的 $G^a$ + 未提交段的 $G^0$；
  4. 按 $p\\min(G^0,G^a) + 0.5p(G^0-G^a)^+ + 1.5p(G^a-G^0)^+ + 5pE$ 结算。

`--issues` 可选预报时刻子集：集合 $S$ 中的时刻 $h_0$ 提交到 $S$ 中下一个时刻（或 24:00），
不在 $S$ 中的时刻既不重优化也不刷新点预测曲线，用于回答「是否需要引入其他时刻预报」。

1 月为冷启动预热（同样的完整流程，只是不写入结果文件），2–12 月为正式结果。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

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
from .executor import DayExecution, run_day, DPValueExecutor
from .value_dp import next_day_value, future_cost, evaluate_plan
from .forecast import PointForecaster
from .forecast_price import ConstantPriceSource, PriceSource, horizon_price
from .forecast_pv import ISSUE_HOURS, PVIssueForecaster, issue_segment
from .optimizer import StorageRollingSolver, solve_readjust, solve_saa, solve_saa_point
from .results import print_day_summary_q3, summarize_day_q3, variant_table, write_result3
from .run_q2 import PERIOD_END, RECORD_START, WARMUP_START, daterange
from .scenarios import M_SCEN, PVResidualLibraryByIssue, ResidualLibrary
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
    step_minutes: int = 10
    issues: tuple[int, ...] = ISSUE_HOURS


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
) -> PeriodResult:
    """从 start 到 end 逐日 walk-forward 回测；只记录 record_from 起的日子。

    `price_source` 缺省为附件 1 的常数电价（问 3 本体）；问 4 传入附件 4 的电价预测器，
    此时每个预报时刻同时刷新光伏预报与电价的日水平因子 $\\lambda$。
    """
    record_from = start if record_from is None else record_from
    issues = tuple(sorted(set(int(h) for h in params.issues)))
    assert issues and issues[0] == 0, "预报时刻集合必须包含 0:00（计划购电由它决定）"
    assert all(h in ISSUE_HOURS for h in issues), f"预报时刻只能取自 {ISSUE_HOURS}"
    blocks = block_bounds_of(issues)

    ps: PriceSource = price_source or ConstantPriceSource(bundle.att1.price)
    price48_0 = horizon_price(ps, start, 0, 2)
    forecaster = PointForecaster(
        bundle.daily, bundle.att1, k_load=params.k_load, k_pv=params.k_pv
    )
    pv_fc = PVIssueForecaster(bundle.daily, bundle.att3)
    library = ResidualLibrary()
    pv_lib = PVResidualLibraryByIssue(library)
    solver = StorageRollingSolver(price48_0, n_today=T)

    def learn(d: date) -> None:
        """把日期 d 的残差入库：负载沿用问 2 的库，光伏与电价按发布时刻分库。"""
        load_pred, pv_pred = forecaster.predict(d)
        load_true, pv_true = bundle.truth(d)
        library.update(d, load_true, pv_true, load_pred, pv_pred, ps.residual(d, 0))
        for h0 in issues:
            pv_lib.update(
                d, h0, pv_fc.residual(d, h0), ps.residual(d, issue_segment(h0))
            )

    # 起始日之前的历史也要入库，否则场景层要等到区间中段才有残差可用。
    for d in daterange(bundle.daily.dates[0], start - timedelta(days=1)):
        learn(d)

    out = PeriodResult(soc_end=float(soc_init), label=",".join(str(h) for h in issues))
    soc = float(soc_init)
    t_start = time.perf_counter()
    for i, d in enumerate(daterange(start, end)):
        load_pred, _ = forecaster.predict(d)
        load_next, pv_mean_next = forecaster.predict_next(d)
        load_true, pv_true = bundle.truth(d)
        price = ps.truth(d)
        price48 = price48_0 if not ps.varies else horizon_price(ps, d, 0, 2)
        price_fn = None
        if ps.varies:
            price_fn = lambda t, day=d: horizon_price(ps, day, t + 1, 2)  # noqa: E731

        # 0:00：附件 3 的 0:00 预报 + 0:00 残差库 → SAA 冻结计划购电量
        pv_today, pv_next = pv_fc.curves(d, 0, pv_mean_next)
        L_scen, PV_scen = pv_lib.scenarios(d, 0, load_pred, pv_today, params.m_scen)
        price_scen = pv_lib.price_scenarios(
            d, 0, price48[:T], price48[T:], params.m_scen
        )
        saa = solve_saa(
            price48,
            L_scen,
            PV_scen,
            load_next,
            pv_next,
            soc,
            n_today=T,
            price_scen=price_scen,
        )
        # 0:00 与后续重优化块同样从场景解、点预测解的五个凸组合中
        # 选取 DP 回放费用最低的计划，之后才冻结为 G^0。
        point0 = solve_saa_point(
            price48, load_pred, pv_today, load_next, pv_next, soc, n_today=T
        )
        g_s0 = np.maximum(saa.G0, 0.0)
        g_d0 = np.maximum(point0.G[:T], 0.0)
        terminal0 = next_day_value(np.asarray(price48[T:], float), load_next, pv_next)
        p0_scen = None if price_scen is None else np.asarray(price_scen[:, :T], float)
        G0 = min(
            [(1.0 - a) * g_s0 + a * g_d0 for a in (0.0, 0.25, 0.5, 0.75, 1.0)],
            key=lambda g: evaluate_plan(
                g,
                L_scen,
                PV_scen,
                price48[:T],
                soc,
                terminal0,
                price_scen=p0_scen,
            ),
        )
        Ga = G0.copy()  # 每段唯一的提交值；0:00–6:00 段恒等于 G^0
        g_cur = G0.copy()  # 滚动 LP 用：已提交段取 G^a，未提交段取 G^0

        execution = DayExecution.empty(d, soc)
        soc_block = soc
        g_temp = G0.copy()
        for h0, t0, t1 in blocks:
            price_s = price_scen
            if h0:
                pv_today, pv_next = pv_fc.curves(d, h0, pv_mean_next)
                L_s, PV_s = pv_lib.scenarios(
                    d, h0, load_pred[t0:], pv_today[t0:], params.m_scen
                )
                # 重优化同时刷新光伏预报与电价的日水平因子（当天已观测段的最小二乘）
                price48_h = price48 if not ps.varies else horizon_price(ps, d, t0, 2)
                price_s = pv_lib.price_scenarios(
                    d, h0, price48_h[t0:T], price48_h[T:], params.m_scen
                )
                if price_s is not None:  # 场景电价按 price48 的整日约定补齐 $t<t_0$ 段
                    head = np.tile(price48_h[:t0], (price_s.shape[0], 1))
                    price_s = np.concatenate([head, price_s], axis=1)
                rj = solve_readjust(
                    t0,
                    price48_h,
                    G0,
                    L_s,
                    PV_s,
                    load_next,
                    pv_next,
                    soc_block,
                    price_scen=price_s,
                )
                rj_point = solve_readjust(
                    t0,
                    price48_h,
                    G0,
                    load_pred[t0:][None, :],
                    pv_today[t0:][None, :],
                    load_next,
                    pv_next,
                    soc_block,
                )
                vnext = next_day_value(np.asarray(price48[T:], float), load_next, pv_next)
                p_eval = np.asarray(price48_h[t0:T], float)
                ps_eval = None if price_s is None else np.asarray(price_s[:, t0:T], float)
                cand = [
                    (1.0 - a) * rj.Ga[t0:] + a * rj_point.Ga[t0:]
                    for a in (0.0, 0.25, 0.5, 0.75, 1.0)
                ]
                chosen = min(
                    cand,
                    key=lambda g: evaluate_plan(
                        g, L_s, PV_s, p_eval, soc_block, vnext,
                        price_scen=ps_eval, base_plan=G0[t0:]
                    ),
                )
                rj.Ga[t0:] = chosen
                Ga[t0:t1] = chosen[: t1 - t0]  # 只提交到下一预报时刻，其余为临时决策
                g_cur[t0:t1] = Ga[t0:t1]
                g_temp[t0:] = rj.Ga[t0:]
            # 每次重优化后从当前时刻重算未来费用函数，并用 DP 价值执行器逐段执行。
            p_rem = np.asarray(price48_h if h0 else price48, float)[t0:T]
            Lr = np.asarray(L_s if h0 else L_scen, float)
            PVr = np.asarray(PV_s if h0 else PV_scen, float)
            gr = np.asarray(g_temp[t0:T], float)
            p_rem_scen = None if price_s is None else np.asarray(price_s[:, t0:T], float)
            vnext = next_day_value(np.asarray(price48[T:], float), load_next, pv_next)
            hbar = future_cost(p_rem if p_rem_scen is None else p_rem_scen, Lr, PVr, gr, vnext)
            dp_executor = DPValueExecutor(hbar, price=p_rem).prepare(t0)
            run_day(
                d,
                g_cur,
                soc_block,
                load_true,
                pv_true,
                load_pred,
                pv_today,
                load_next,
                pv_next,
                dp_executor,
                step_minutes=params.step_minutes,
                t_start=t0,
                t_end=t1,
                out=execution,
                price_fn=price_fn,
            )
            soc_block = float(execution.S[t1 - 1])

        cost = cost_q3(price, G0, Ga, execution.E)
        if d >= record_from:
            out.days.append(
                DayResult(
                    day=d,
                    G0=G0,
                    Ga=Ga,
                    C=execution.C,
                    D=execution.D,
                    S=execution.S,
                    E=execution.E,
                    W=execution.W,
                    R=execution.R,
                    soc_start=execution.soc_start,
                    plan_cost=cost["plan"],
                    curtail_cost=cost["curtail_penalty"],
                    extra_cost=cost["extra"],
                    emergency_cost=cost["emergency"],
                    plan_only_cost=float(price @ G0),
                    price=price if ps.varies else None,
                )
            )
        learn(d)
        soc = execution.soc_end
        if progress and (i + 1) % progress == 0:
            print(
                f"  [{d.isoformat()}] 已完成 {i + 1} 天，"
                f"用时 {time.perf_counter() - t_start:.1f}s",
                flush=True,
            )

    out.soc_end = soc
    out.runtime_s = time.perf_counter() - t_start
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
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help=f"跳过 1 月参数选择，直接用问 2 选定的 K_L={K_LOAD}, K_P={K_PV}",
    )
    parser.add_argument("--k-load", type=int, default=None)
    parser.add_argument("--k-pv", type=int, default=None)
    parser.add_argument("--m-scen", type=int, default=M_SCEN)
    parser.add_argument("--step-minutes", type=int, default=10)
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
    params = Params(m_scen=args.m_scen, step_minutes=args.step_minutes, issues=args.issues)

    if args.k_load is not None:
        params.k_load = args.k_load
    if args.k_pv is not None:
        params.k_pv = args.k_pv
    if not args.skip_tuning and args.k_load is None and args.k_pv is None:
        from .run_q2 import Params as Q2Params
        from .run_q2 import load_bundle as load_bundle_q2
        from .tuning import select_k

        print("在 1 月按问 2 流程（实际购电费）选择 K_L, K_P …", flush=True)
        picked = select_k(load_bundle_q2(), base=Q2Params())
        params.k_load, params.k_pv = picked.k_load, picked.k_pv
        print(
            f"选定 K_L={params.k_load}, K_P={params.k_pv}，"
            f"{picked.window} 总费用 {picked.cost:.2f} 元\n"
        )
    else:
        source = "命令行指定" if (args.k_load or args.k_pv) else "问 2 选定值（--skip-tuning）"
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
