"""问 4：附件 4 波动电价下重跑问 2 与问 3（result4-2、result4-3）。

与问 2 / 问 3 的唯一差别是电价：决策时刻只能看到已发生的电价，用
`forecast_price.PriceForecaster`（近期同类型日基准 × 日水平因子 $\\lambda$）给出当天
剩余段与次日的点预测，电价残差与负载、光伏残差取同一历史日形成联合场景；
储能滚动 LP 每段用「当段真值 + 剩余段点预测」的电价；结算全部用附件 4 真值。

`--perfect-price` 变体让 0:00 的 SAA 与重优化直接看到当天与次日的真实电价（无电价残差
场景），其余完全相同，给出电价不确定性代价的下界。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date

import numpy as np

from .data import SOC_INIT, load_attachment4
from .forecast_price import K_PRICE, PerfectPriceSource, PriceForecaster, PriceSource
from .results import (
    print_day_summary,
    print_day_summary_q3,
    summarize_day,
    summarize_day_q3,
    write_result2,
    write_result3,
)
from .run_q2 import PERIOD_END, RECORD_START, WARMUP_START
from .run_q2 import Params as Q2Params
from .run_q2 import load_bundle as load_bundle_q2
from .run_q2 import print_period_summary as print_period_summary_q2
from .run_q2 import run_period as run_period_q2
from .run_q2 import validate as validate_q2
from .run_q3 import K_LOAD, K_PV
from .run_q3 import Params as Q3Params
from .run_q3 import load_bundle as load_bundle_q3
from .run_q3 import print_period_summary as print_period_summary_q3
from .run_q3 import run_period as run_period_q3
from .run_q3 import validate as validate_q3
from .scenarios import M_SCEN

REPORT_DAYS = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]


@dataclass
class Q4Run:
    """一次问 4 回测的结果与元信息。"""

    which: int
    label: str
    result: object
    runtime_s: float


def make_price_source(
    att4: tuple[list[date], np.ndarray], att1, k_price: int, perfect: bool
) -> PriceSource:
    """构造电价来源：默认为 walk-forward 预测器，`perfect=True` 时为完美电价信息。"""
    dates, prices = att4
    if perfect:
        return PerfectPriceSource(dates, prices)
    return PriceForecaster(dates, prices, att1, k=k_price)


def run_q4_2(
    bundle,
    price_source: PriceSource,
    params: Q2Params,
    *,
    start: date = RECORD_START,
    end: date = PERIOD_END,
    progress: int = 0,
    point_days: set[date] | None = None,
):
    """问 4-2：问 2 的流程 + 波动电价。"""
    res = run_period_q2(
        WARMUP_START,
        end,
        params,
        bundle,
        record_from=start,
        soc_init=SOC_INIT,
        progress=progress,
        point_days=point_days,
        price_source=price_source,
    )
    validate_q2(res, bundle)
    return res


def run_q4_3(
    bundle,
    price_source: PriceSource,
    params: Q3Params,
    *,
    start: date = RECORD_START,
    end: date = PERIOD_END,
    progress: int = 0,
):
    """问 4-3：问 3 的流程 + 波动电价（重优化时同时刷新光伏预报与 $\\lambda$）。"""
    res = run_period_q3(
        WARMUP_START,
        end,
        params,
        bundle,
        record_from=start,
        soc_init=SOC_INIT,
        progress=progress,
        price_source=price_source,
    )
    validate_q3(res, bundle, params.issues)
    return res


def compare_line(label: str, cost: float, base_cost: float, base_label: str) -> str:
    """「label 的费用 X 元，相对 base_label ±D（±p%）」。"""
    diff = cost - base_cost
    pct = 100.0 * (cost / base_cost - 1.0) if base_cost else 0.0
    return f"  {label:<24} {cost:14.2f} 元  相对{base_label} {diff:+12.2f}（{pct:+.3f}%）"


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="问 4：附件 4 波动电价下的电价预测层 + 问 2 / 问 3 重算"
    )
    parser.add_argument("--which", choices=("2", "3", "both"), default="both")
    parser.add_argument(
        "--skip-tuning", action="store_true", help=f"跳过 1 月 K_p 选择，直接用 K_p={K_PRICE}"
    )
    parser.add_argument("--k-price", type=int, default=None)
    parser.add_argument("--k-load", type=int, default=K_LOAD)
    parser.add_argument("--k-pv", type=int, default=K_PV)
    parser.add_argument("--m-scen", type=int, default=M_SCEN)
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--candidate-workers", type=int, default=1,
                        help="每条回测中并行重评候选的进程数；与并行全年任务共享总核数")
    parser.add_argument(
        "--perfect-price", action="store_true", help="完美电价信息变体（费用下界）"
    )
    parser.add_argument(
        "--with-constant",
        action="store_true",
        help="同窗口再跑一遍附件 1 常数电价的问 2 / 问 3，打印全年费用对比",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=RECORD_START)
    parser.add_argument("--end", type=date.fromisoformat, default=PERIOD_END)
    parser.add_argument("--out2", default="results/result4-2.xlsx")
    parser.add_argument("--out3", default="results/result4-3.xlsx")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--progress", type=int, default=30)
    args = parser.parse_args(argv)

    wall = time.perf_counter()
    att4 = load_attachment4()
    bundle2 = load_bundle_q2()
    want = {"2": (True, False), "3": (False, True), "both": (True, True)}[args.which]
    do2, do3 = want

    k_price = K_PRICE if args.k_price is None else args.k_price
    if args.k_price is None and not args.skip_tuning and not args.perfect_price:
        from .tuning import select_kp

        print("在 1 月（1–7 日预热不计分）按问 4-2 管线选择 K_p …", flush=True)
        picked = select_kp(
            bundle2,
            att4,
            base=Q2Params(
                k_load=args.k_load,
                k_pv=args.k_pv,
                m_scen=args.m_scen,
                step_minutes=args.step_minutes,
                candidate_workers=args.candidate_workers,
            ),
        )
        k_price = picked.k_price
        print(f"选定 K_p={k_price}，{picked.window} 总费用 {picked.cost:.2f} 元\n")
    else:
        why = (
            "命令行指定"
            if args.k_price is not None
            else ("完美电价变体不需要基准" if args.perfect_price else "默认（--skip-tuning）")
        )
        print(f"使用参数 K_p={k_price}（{why}）\n")
    print(f"K_L={args.k_load}, K_P={args.k_pv}（问 2 在 1 月选定），M={args.m_scen}")
    if args.perfect_price:
        print("电价来源：完美信息（0:00 已知当天与次日真实电价，无电价残差场景）")
    print()

    out: dict[str, object] = {"k_price": k_price}
    if do2:
        source = make_price_source(att4, bundle2.att1, k_price, args.perfect_price)
        print(
            f"问 4-2 回测 {WARMUP_START.isoformat()} → {args.end.isoformat()}"
            f"（{args.start.isoformat()} 起计入结果）",
            flush=True,
        )
        t0 = time.perf_counter()
        res2 = run_q4_2(
            bundle2,
            source,
            Q2Params(
                k_load=args.k_load,
                k_pv=args.k_pv,
                m_scen=args.m_scen,
                step_minutes=args.step_minutes,
                candidate_workers=args.candidate_workers,
            ),
            start=args.start,
            end=args.end,
            progress=args.progress,
            point_days=set(REPORT_DAYS),
        )
        out["q4_2"] = res2
        print(f"  用时 {time.perf_counter() - t0:.1f}s\n", flush=True)
        if not args.no_write and not args.perfect_price:
            path = write_result2(res2.days, args.out2)
            print(f"已写出 {path}（{len(res2.days)} 行）\n")
        for d in REPORT_DAYS:
            day = res2.by_date(d)
            if day is not None:
                print_day_summary(summarize_day(day, day.price))
        print_period_summary_q2(
            res2, label=f"问 4-2 {args.start.isoformat()}→{args.end.isoformat()} "
        )
        print()

    if do3:
        bundle3 = load_bundle_q3()
        source3 = make_price_source(att4, bundle3.att1, k_price, args.perfect_price)
        print(
            f"问 4-3 回测 {WARMUP_START.isoformat()} → {args.end.isoformat()}"
            f"（{args.start.isoformat()} 起计入结果）",
            flush=True,
        )
        t0 = time.perf_counter()
        res3 = run_q4_3(
            bundle3,
            source3,
            Q3Params(
                k_load=args.k_load,
                k_pv=args.k_pv,
                m_scen=args.m_scen,
                step_minutes=args.step_minutes,
                candidate_workers=args.candidate_workers,
            ),
            start=args.start,
            end=args.end,
            progress=args.progress,
        )
        out["q4_3"] = res3
        print(f"  用时 {time.perf_counter() - t0:.1f}s\n", flush=True)
        if not args.no_write and not args.perfect_price:
            path = write_result3(res3.days, args.out3)
            print(f"已写出 {path}（{len(res3.days)} 行）\n")
        for d in REPORT_DAYS:
            day = res3.by_date(d)
            if day is not None:
                print_day_summary_q3(summarize_day_q3(day, day.price))
        print_period_summary_q3(
            res3, label=f"问 4-3 {args.start.isoformat()}→{args.end.isoformat()} "
        )
        print()

    print("—— 波动电价（问 4）与常数电价（问 2 / 问 3）的全年费用 ——")
    if do2:
        print(f"  问 4-2 总费用 {out['q4_2'].total_cost:14.2f} 元")
    if do3:
        print(f"  问 4-3 总费用 {out['q4_3'].total_cost:14.2f} 元")
    if do2 and do3:
        print(
            compare_line(
                "问 4-3（同窗口）",
                out["q4_3"].total_cost,
                out["q4_2"].total_cost,
                "问 4-2",
            )
        )
    if args.with_constant:
        # 两者的电价基准不同（附件 1 常数电价恰为附件 4 的全年逐段均值），
        # 差额同时包含「电价波动本身」与「电价只能预测」两部分代价。
        if do2:
            base2 = run_period_q2(
                WARMUP_START, args.end, Q2Params(
                    k_load=args.k_load, k_pv=args.k_pv,
                    m_scen=args.m_scen, step_minutes=args.step_minutes,
                    candidate_workers=args.candidate_workers,
                ), bundle2, record_from=args.start, soc_init=SOC_INIT,
                progress=args.progress,
            )
            validate_q2(base2, bundle2)
            out["q2_const"] = base2
            print(f"  问 2（常数电价）总费用 {base2.total_cost:14.2f} 元")
            print(
                compare_line(
                    "问 4-2（波动电价）", out["q4_2"].total_cost, base2.total_cost, "问 2"
                )
            )
        if do3:
            base3 = run_period_q3(
                WARMUP_START, args.end, Q3Params(
                    k_load=args.k_load, k_pv=args.k_pv,
                    m_scen=args.m_scen, step_minutes=args.step_minutes,
                    candidate_workers=args.candidate_workers,
                ), bundle3, record_from=args.start, soc_init=SOC_INIT,
                progress=args.progress,
            )
            validate_q3(base3, bundle3, Q3Params().issues)
            out["q3_const"] = base3
            print(f"  问 3（常数电价）总费用 {base3.total_cost:14.2f} 元")
            print(
                compare_line(
                    "问 4-3（波动电价）", out["q4_3"].total_cost, base3.total_cost, "问 3"
                )
            )
    print(f"\n总运行时间 {time.perf_counter() - wall:.1f}s")
    return out


if __name__ == "__main__":
    main()
