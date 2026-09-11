"""问 2：逐日 walk-forward 回测（SAA 购电 + 储能滚动控制 + 真值结算）。

每个决策日 0:00：
  1. 预测/场景层用日期 < d 的附件 2 数据给出当天点预测、次日点预测与 M 个等权场景；
  2. SAA 两阶段 LP 冻结当天计划购电量 G^0；
  3. 执行层用附件 2 真值逐段跑储能滚动控制，得到执行的充放电、储电量、紧急购电；
  4. 结算层按 p_t G^0_t + 5 p_t E_t 计费，残差入库，24:00 储电量结转次日。

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
)
from .executor import DayExecution, run_day_dp, DPValueExecutor
from .value_dp import next_day_value
from .parallel_eval import CandidateEvaluator
from .forecast import PointForecaster
from .forecast_price import ConstantPriceSource, PriceSource, horizon_price
from .optimizer import solve_saa, solve_saa_point
from .results import print_day_summary, summarize_day, write_result2
from .scenarios import M_SCEN, ResidualLibrary
from .settlement import cost_q2

WARMUP_START = date(2025, 1, 1)
RECORD_START = date(2025, 2, 1)
PERIOD_END = date(2025, 12, 31)
REPORT_DAYS = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]


@dataclass
class Params:
    """一次回测的全部超参数。"""

    k_load: int = 4
    k_pv: int = 5
    m_scen: int = M_SCEN
    # Kept for legacy StorageRollingSolver experiments only.  DP execution is
    # causal at the native 10-minute cadence and deliberately ignores it.
    step_minutes: int = 10
    horizon_days: int = 2  # 2 = 48h（当天 + 次日）；1 = 24h；3 = 72h
    candidate_workers: int = 1  # independent plan candidates; 1 keeps serial execution


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
) -> PeriodResult:
    """从 start 到 end 逐日 walk-forward 回测；只记录 record_from 起的日子。

    `point_days` 中的日期额外求一次点预测解（M=1、残差为零的 48h 确定性 LP），
    只挂在结果上供论文对照，不参与执行。

    `price_source` 缺省为附件 1 的常数电价（问 2 本体）；问 4 传入附件 4 的电价预测器，
    此时每天的电价曲线来自决策时刻的点预测、场景带电价残差、结算与执行用真值。
    """
    record_from = start if record_from is None else record_from
    point_days = point_days or set()
    ps: PriceSource = price_source or ConstantPriceSource(bundle.att1.price)
    n_days = params.horizon_days
    n_future = T * (n_days - 1)
    price_h0 = horizon_price(ps, start, 0, n_days)

    forecaster = PointForecaster(
        bundle.daily, bundle.att1, k_load=params.k_load, k_pv=params.k_pv
    )
    library = ResidualLibrary()

    # 起始日之前的历史也要入库，否则场景层要等到区间中段才有残差可用。
    for d in daterange(bundle.daily.dates[0], start - timedelta(days=1)):
        load_pred, pv_pred = forecaster.predict(d)
        load_true, pv_true = bundle.truth(d)
        library.update(d, load_true, pv_true, load_pred, pv_pred, ps.residual(d, 0))

    out = PeriodResult(soc_end=float(soc_init))
    soc = float(soc_init)
    t_start = time.perf_counter()
    candidate_evaluator = CandidateEvaluator(params.candidate_workers)
    for i, d in enumerate(daterange(start, end)):
        load_pred, pv_pred = forecaster.predict(d)
        load_future, pv_future = forecaster.predict_future(d, n_days - 1)
        assert load_future.size == n_future

        price_h = price_h0 if not ps.varies else horizon_price(ps, d, 0, n_days)
        L_scen, PV_scen = library.scenarios(d, load_pred, pv_pred, params.m_scen)
        price_scen = library.price_scenarios(
            d, price_h[:T], price_h[T:], params.m_scen
        )
        saa = solve_saa(
            price_h,
            L_scen,
            PV_scen,
            load_future,
            pv_future,
            soc,
            n_today=T,
            price_scen=price_scen,
        )
        point = solve_saa_point(price_h, load_pred, pv_pred, load_future, pv_future, soc)
        g_s, g_d = np.maximum(saa.G0, 0.0), np.maximum(point.G[:T], 0.0)
        # The terminal DP prices every available future day.  This makes the
        # documented 24/48/72-hour horizon variants shape-safe: 24h has an
        # empty future and hence a zero terminal value.
        terminal_for_eval = (
            next_day_value(price_h[T:], load_future, pv_future)
            if n_future else None
        )
        p_eval_scen = None if price_scen is None else np.asarray(price_scen[:, :T], float)
        candidates = [((1-a)*g_s+a*g_d) for a in (0., .25, .5, .75, 1.)]
        scored = candidate_evaluator.evaluate(
            candidates, L_scen, PV_scen, price_h[:T], soc, terminal_for_eval,
            price_scen=p_eval_scen,
        )
        winner = int(np.argmin([score for score, _ in scored]))
        g0, hbar = candidates[winner], scored[winner][1]

        load_true, pv_true = bundle.truth(d)
        # DP 价值执行器：由次日价值与同批场景反推剩余费用函数。
        dp_executor = DPValueExecutor(hbar, price=price_h[:T]).prepare(0)
        price_true = ps.truth(d)
        price_fn = (lambda t, p=price_true: float(p[t])) if ps.varies else None
        execution: DayExecution = run_day_dp(
            d,
            g0,
            soc,
            load_true,
            pv_true,
            price_true,
            dp_executor,
            price_fn=price_fn,
        )
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
        library.update(d, load_true, pv_true, load_pred, pv_pred, ps.residual(d, 0))
        soc = execution.soc_end
        if progress and (i + 1) % progress == 0:
            print(
                f"  [{d.isoformat()}] 已完成 {i + 1} 天，"
                f"用时 {time.perf_counter() - t_start:.1f}s",
                flush=True,
            )

    out.soc_end = soc
    out.runtime_s = time.perf_counter() - t_start
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
    parser.add_argument("--skip-tuning", action="store_true", help="跳过 1 月参数选择，用默认 K_L=4, K_P=5")
    parser.add_argument("--k-load", type=int, default=None)
    parser.add_argument("--k-pv", type=int, default=None)
    parser.add_argument("--m-scen", type=int, default=M_SCEN)
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
        step_minutes=args.step_minutes,
        candidate_workers=args.candidate_workers,
    )

    if args.k_load is not None:
        params.k_load = args.k_load
    if args.k_pv is not None:
        params.k_pv = args.k_pv
    if not args.skip_tuning and args.k_load is None and args.k_pv is None:
        from .tuning import select_k

        print("在 1 月（1–7 日预热不计分）按实际购电费选择 K_L, K_P …", flush=True)
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
