"""参数选择与敏感性实验。

`select_k` 在 1 月按「实际购电费（计划费 + 紧急费）」最小选定 K_L, K_P；
`sensitivity` 提供场景数 M 与优化时域 24/48/72h 的全年 walk-forward 对照入口。
两者都走与主流程完全相同的 walk-forward 回测，只是不写结果文件。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from .data import SOC_INIT
from .run_q2 import (
    PERIOD_END,
    WARMUP_START,
    Bundle,
    Params,
    PeriodResult,
    load_bundle,
    run_period,
)

JAN_END = date(2025, 1, 31)
JAN_SCORE_FROM = date(2025, 1, 8)  # 1 月 1–7 日为冷启动，不计入选参得分
K_LOAD_GRID = (2, 3, 4, 5, 6)
K_PV_GRID = (3, 4, 5, 6, 7)
KP_GRID = (2, 3, 4, 6, 8)  # 问 4 的电价基准窗口 $K_p$
M_GRID = (6, 12, 20)
HORIZON_GRID = (1, 2, 3)  # 24h / 48h / 72h


@dataclass
class Candidate:
    """一个 (K_L, K_P) 候选：cost 等字段是计分窗口内的费用，cost_full 是整个 1 月。"""

    k_load: int
    k_pv: int
    cost: float
    plan_cost: float
    emergency_cost: float
    emergency_kwh: float
    soc_end: float
    cost_full: float = 0.0


@dataclass
class TuningResult:
    """`select_k` 的完整输出：最优候选 + 全部候选在计分窗口内的费用。"""

    k_load: int
    k_pv: int
    cost: float
    score_from: date = JAN_SCORE_FROM
    score_end: date = JAN_END
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def window(self) -> str:
        return f"{self.score_from.isoformat()}→{self.score_end.isoformat()}"

    def table(self) -> str:
        head = f"计分窗口 {self.window}（1 月 1 日起预热，冷启动日不计分）"
        lines = [
            head,
            f"{'K_L':>4} {'K_P':>4} {'窗口总费用':>14} {'计划费':>14} {'紧急费':>12}"
            f" {'紧急购电kWh':>13} {'整月总费用':>14}",
        ]
        for c in sorted(self.candidates, key=lambda c: c.cost):
            lines.append(
                f"{c.k_load:>4} {c.k_pv:>4} {c.cost:>14.2f} {c.plan_cost:>14.2f}"
                f" {c.emergency_cost:>12.2f} {c.emergency_kwh:>13.2f} {c.cost_full:>14.2f}"
            )
        return "\n".join(lines)

    def get(self, k_load: int, k_pv: int) -> Candidate | None:
        for c in self.candidates:
            if c.k_load == k_load and c.k_pv == k_pv:
                return c
        return None


def _january(bundle: Bundle, params: Params, end: date = JAN_END) -> PeriodResult:
    return run_period(WARMUP_START, end, params, bundle, soc_init=SOC_INIT)


def _score(res: PeriodResult, score_from: date) -> tuple[float, float, float]:
    """计分窗口（日期 ≥ score_from）内的 (计划费, 紧急费, 紧急购电量)。"""
    days = [d for d in res.days if d.day >= score_from]
    return (
        sum(d.plan_cost for d in days),
        sum(d.emergency_cost for d in days),
        float(sum(float(d.E.sum()) for d in days)),
    )


def select_k(
    bundle: Bundle | None = None,
    k_loads=K_LOAD_GRID,
    k_pvs=K_PV_GRID,
    *,
    base: Params | None = None,
    end: date = JAN_END,
    score_from: date = JAN_SCORE_FROM,
    verbose: bool = True,
) -> TuningResult:
    """1 月网格搜索 K_L, K_P，取计分窗口内实际总费用（计划费 + 紧急费）最小者。

    每个候选都从 1 月 1 日、SOC=6000 起跑完全相同的预热流程，只是 `score_from`
    之前的日子不计入得分：1 月 1 日无任何历史（只能用附件 1 先验）、1 月 2–7 日的
    残差库里还装着先验预测器留下的巨大残差，这些冷启动日的费用与 K 的关系和常态日
    相反，会主导整月总费用并把 K_P 顶到网格边界。默认 (4, 5) 为回退值。
    """
    bundle = bundle or load_bundle()
    base = base or Params()
    candidates: list[Candidate] = []
    for k_load in k_loads:
        for k_pv in k_pvs:
            t0 = time.perf_counter()
            res = _january(bundle, replace(base, k_load=k_load, k_pv=k_pv), end)
            plan_cost, emergency_cost, emergency_kwh = _score(res, score_from)
            cand = Candidate(
                k_load=k_load,
                k_pv=k_pv,
                cost=plan_cost + emergency_cost,
                plan_cost=plan_cost,
                emergency_cost=emergency_cost,
                emergency_kwh=emergency_kwh,
                soc_end=res.soc_end,
                cost_full=res.total_cost,
            )
            candidates.append(cand)
            if verbose:
                print(
                    f"  K_L={k_load} K_P={k_pv}: 窗口总费用 {cand.cost:12.2f} 元"
                    f"（紧急 {cand.emergency_cost:.2f}；整月 {cand.cost_full:.2f}，"
                    f"{time.perf_counter() - t0:.1f}s）",
                    flush=True,
                )
    best = min(candidates, key=lambda c: c.cost)
    result = TuningResult(
        k_load=best.k_load,
        k_pv=best.k_pv,
        cost=best.cost,
        score_from=score_from,
        score_end=end,
        candidates=candidates,
    )
    if verbose:
        print(f"\n{result.table()}\n", flush=True)
    return result


@dataclass
class KpCandidate:
    """一个 $K_p$ 候选在计分窗口内的费用（用附件 4 真值结算）。"""

    k_price: int
    cost: float
    plan_cost: float
    emergency_cost: float
    emergency_kwh: float
    cost_full: float = 0.0
    runtime_s: float = 0.0


@dataclass
class KpResult:
    """`select_kp` 的输出：最优 $K_p$ + 全部候选。"""

    k_price: int
    cost: float
    score_from: date = JAN_SCORE_FROM
    score_end: date = JAN_END
    candidates: list[KpCandidate] = field(default_factory=list)

    @property
    def window(self) -> str:
        return f"{self.score_from.isoformat()}→{self.score_end.isoformat()}"

    def table(self) -> str:
        lines = [
            f"计分窗口 {self.window}（1 月 1 日起预热，冷启动日不计分）",
            f"{'K_p':>4} {'窗口总费用':>14} {'计划费':>14} {'紧急费':>12}"
            f" {'紧急购电kWh':>13} {'整月总费用':>14}",
        ]
        for c in sorted(self.candidates, key=lambda c: c.cost):
            lines.append(
                f"{c.k_price:>4} {c.cost:>14.2f} {c.plan_cost:>14.2f}"
                f" {c.emergency_cost:>12.2f} {c.emergency_kwh:>13.2f} {c.cost_full:>14.2f}"
            )
        return "\n".join(lines)


def select_kp(
    bundle: Bundle,
    att4: tuple[list[date], np.ndarray],
    k_grid=KP_GRID,
    *,
    base: Params | None = None,
    end: date = JAN_END,
    score_from: date = JAN_SCORE_FROM,
    verbose: bool = True,
) -> KpResult:
    """1 月按问 4-2 管线选电价基准窗口 $K_p$，判据与 `select_k` 相同。

    $K_L, K_P$ 沿用问 2 选定值（由 `base` 传入），只有电价预测器的 $K_p$ 变化；
    得分是计分窗口内用附件 4 真值结算的实际总费用（计划费 + 紧急费）。
    """
    from .forecast_price import PriceForecaster

    base = base or Params()
    dates, prices = att4
    candidates: list[KpCandidate] = []
    for kp in k_grid:
        t0 = time.perf_counter()
        source = PriceForecaster(dates, prices, bundle.att1, k=kp)
        res = run_period(
            WARMUP_START, end, base, bundle, soc_init=SOC_INIT, price_source=source
        )
        plan_cost, emergency_cost, emergency_kwh = _score(res, score_from)
        cand = KpCandidate(
            k_price=kp,
            cost=plan_cost + emergency_cost,
            plan_cost=plan_cost,
            emergency_cost=emergency_cost,
            emergency_kwh=emergency_kwh,
            cost_full=res.total_cost,
            runtime_s=time.perf_counter() - t0,
        )
        candidates.append(cand)
        if verbose:
            print(
                f"  K_p={kp}: 窗口总费用 {cand.cost:12.2f} 元"
                f"（紧急 {cand.emergency_cost:.2f}；整月 {cand.cost_full:.2f}，"
                f"{cand.runtime_s:.1f}s）",
                flush=True,
            )
    best = min(candidates, key=lambda c: c.cost)
    result = KpResult(
        k_price=best.k_price,
        cost=best.cost,
        score_from=score_from,
        score_end=end,
        candidates=candidates,
    )
    if verbose:
        print(f"\n{result.table()}\n", flush=True)
    return result


@dataclass
class SensitivityRow:
    name: str
    m_scen: int
    horizon_days: int
    total_cost: float
    emergency_kwh: float
    mean_soc_end: float
    runtime_s: float


def sensitivity(
    bundle: Bundle | None = None,
    params: Params | None = None,
    *,
    m_grid=M_GRID,
    horizon_grid=HORIZON_GRID,
    start: date = WARMUP_START,
    end: date = PERIOD_END,
    record_from: date = date(2025, 2, 1),
    verbose: bool = True,
) -> list[SensitivityRow]:
    """场景数 M 与时域长度的全年 walk-forward 敏感性；默认不随主流程运行。"""
    bundle = bundle or load_bundle()
    params = params or Params()
    rows: list[SensitivityRow] = []
    configs = [("M", m, params.horizon_days) for m in m_grid]
    configs += [("时域", params.m_scen, h) for h in horizon_grid]
    for tag, m, h in configs:
        cfg = replace(params, m_scen=m, horizon_days=h)
        res = run_period(start, end, cfg, bundle, record_from=record_from, soc_init=SOC_INIT)
        row = SensitivityRow(
            name=f"{tag}: M={m}, 时域={24 * h}h",
            m_scen=m,
            horizon_days=h,
            total_cost=res.total_cost,
            emergency_kwh=res.emergency_kwh,
            mean_soc_end=float(np.mean([d.soc_end for d in res.days])),
            runtime_s=res.runtime_s,
        )
        rows.append(row)
        if verbose:
            print(
                f"  {row.name:<22} 总费用 {row.total_cost:14.2f} 元，"
                f"紧急 {row.emergency_kwh:10.2f} kWh，平均 24:00 SOC {row.mean_soc_end:8.2f}"
                f"（{row.runtime_s:.1f}s）",
                flush=True,
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="问 2 参数选择与敏感性实验")
    parser.add_argument("--select", action="store_true", help="1 月网格选 K_L, K_P")
    parser.add_argument("--sensitivity", action="store_true", help="全年 M / 时域敏感性")
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument(
        "--score-from",
        type=date.fromisoformat,
        default=JAN_SCORE_FROM,
        help="计分窗口起始日（此日之前的冷启动日照跑但不计分）",
    )
    args = parser.parse_args(argv)

    bundle = load_bundle()
    base = Params(step_minutes=args.step_minutes)
    if args.select or not args.sensitivity:
        t0 = time.perf_counter()
        res = select_k(bundle, base=base, score_from=args.score_from)
        default = res.get(Params().k_load, Params().k_pv)
        print(f"选定 K_L={res.k_load}, K_P={res.k_pv}（{res.window} 总费用 {res.cost:.2f} 元）")
        if default is not None:
            print(
                f"默认 K_L={default.k_load}, K_P={default.k_pv} 同窗口 {default.cost:.2f} 元"
                f"（相差 {default.cost - res.cost:+.2f} 元）"
            )
        print(f"用时 {time.perf_counter() - t0:.1f}s")
    if args.sensitivity:
        print("\n全年敏感性：")
        sensitivity(bundle, base)


if __name__ == "__main__":
    main()
