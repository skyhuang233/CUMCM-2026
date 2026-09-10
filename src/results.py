"""结果写出与表 1 / 表 2 / 表 3 摘要。

附件 5 官方模板缺失，按题面重建布局；拿到模板后只改本文件。
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import openpyxl

from .data import INTERVAL_LABELS, INTERVAL_SEGS, SEG_LABELS
from .optimizer import LPResult
from .settlement import cost_q2, emergency_intervals, merge_emergency_intervals  # noqa: F401

# 表 1 指定的 10 分钟段（10:00-10:10 … 20:00-20:10）
TABLE1_SEGS = [60, 72, 84, 96, 108, 120]
TABLE1_LABELS = ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]


def interval_sums(x: np.ndarray, n_per: int = INTERVAL_SEGS) -> np.ndarray:
    """把逐段序列按 4 小时区间求和。"""
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, n_per).sum(axis=1)


def summarize_q1(res: LPResult, price: np.ndarray, soc_init: float) -> dict:
    """表 1（购电）与表 2（充放电与储电量）所需的全部数值。"""
    price = np.asarray(price, dtype=float)
    return {
        "seg_labels": TABLE1_LABELS,
        "seg_purchase": res.G[TABLE1_SEGS],
        "daily_purchase": float(res.G.sum()),
        "purchase_cost": float(price @ res.G),
        "interval_labels": INTERVAL_LABELS,
        "charge": interval_sums(res.C),
        "discharge": interval_sums(res.D),
        "soc_start": float(soc_init),
        "soc_end": float(res.S[-1]),
    }


def write_result1(res: LPResult, soc_init: float, path: str = "results/result1.xlsx", label: str = "附件1") -> str:
    """写出 result1.xlsx：工作表『计划购电量』与『充放电量』。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "计划购电量"
    ws.append(["日期"] + SEG_LABELS)
    ws.append([label] + [round(float(v), 4) for v in res.G])

    ws2 = wb.create_sheet("充放电量")
    ws2.append(["时间段", "充电量", "放电量"])
    charge, discharge = interval_sums(res.C), interval_sums(res.D)
    for name, c, d in zip(INTERVAL_LABELS, charge, discharge):
        ws2.append([name, round(float(c), 4), round(float(d), 4)])
    ws2.append(["0:00 储电量", round(float(soc_init), 4), None])
    ws2.append(["24:00 储电量", round(float(res.S[-1]), 4), None])

    wb.save(path)
    return path


def print_summary(summary: dict) -> None:
    """终端打印表 1 / 表 2。"""
    print("表 1 计划购电量（kWh）")
    for name, val in zip(summary["seg_labels"], summary["seg_purchase"]):
        print(f"  {name:>12}  {val:10.2f}")
    print(f"  {'全天购电量':>12}  {summary['daily_purchase']:10.2f}")
    print(f"  {'全天购电费':>12}  {summary['purchase_cost']:10.2f} 元")
    print()
    print("表 2 充放电量与储电量（kWh）")
    print(f"  {'时间段':>12}  {'充电量':>10}  {'放电量':>10}")
    for name, c, d in zip(summary["interval_labels"], summary["charge"], summary["discharge"]):
        print(f"  {name:>12}  {c:10.2f}  {d:10.2f}")
    print(f"  {'0:00 储电量':>12}  {summary['soc_start']:10.2f}")
    print(f"  {'24:00 储电量':>12}  {summary['soc_end']:10.2f}")


# ---------------------------------------------------------------------------
# 问 2：逐日结果写出与表 1 / 表 2 / 表 3 摘要
# ---------------------------------------------------------------------------

CHARGE_COLUMNS = [f"{name} {kind}" for name in INTERVAL_LABELS for kind in ("充电量", "放电量")]


def summarize_day(day, price: np.ndarray) -> dict:
    """一天的表 1（购电）、表 2（执行充放电与储电量）、表 3（紧急购电区间）数值。

    `day` 需具备 G0, C, D, S, E, soc_start 属性（`run_q2.DayResult` 或等价对象）。
    """
    price = np.asarray(price, dtype=float)
    plan_cost, emergency_cost = cost_q2(price, day.G0, day.E)
    point = getattr(day, "point_solution", None)
    out = {
        "date": day.day,
        "seg_labels": TABLE1_LABELS,
        "seg_purchase": np.asarray(day.G0)[TABLE1_SEGS],
        "daily_purchase": float(np.sum(day.G0)),
        "plan_cost": plan_cost,
        "emergency_cost": emergency_cost,
        "purchase_cost": plan_cost + emergency_cost,
        "emergency_kwh": float(np.sum(day.E)),
        "interval_labels": INTERVAL_LABELS,
        "charge": interval_sums(day.C),
        "discharge": interval_sums(day.D),
        "soc_start": float(day.soc_start),
        "soc_end": float(np.asarray(day.S)[-1]),
        "emergency": emergency_intervals(day.day, day.E),
    }
    if point is not None:
        n = len(day.G0)
        out["point_charge"] = interval_sums(point.C[:n])
        out["point_discharge"] = interval_sums(point.D[:n])
        out["point_soc_end"] = float(point.S[n - 1])
    return out


def print_day_summary(summary: dict) -> None:
    """终端打印某一天的表 1 / 表 2 / 表 3。"""
    d = summary["date"]
    print(f"=== {d.isoformat()} ===")
    print("表 1 计划购电量（kWh）")
    for name, val in zip(summary["seg_labels"], summary["seg_purchase"]):
        print(f"  {name:>12}  {val:10.2f}")
    print(f"  {'全天购电量':>12}  {summary['daily_purchase']:10.2f}")
    print(
        f"  {'全天购电费':>12}  {summary['purchase_cost']:10.2f} 元"
        f"（计划 {summary['plan_cost']:.2f} + 紧急 {summary['emergency_cost']:.2f}）"
    )
    print("表 2 充放电量与储电量（kWh，执行值）")
    print(f"  {'时间段':>12}  {'充电量':>10}  {'放电量':>10}")
    for name, c, dis in zip(summary["interval_labels"], summary["charge"], summary["discharge"]):
        print(f"  {name:>12}  {c:10.2f}  {dis:10.2f}")
    print(f"  {'0:00 储电量':>12}  {summary['soc_start']:10.2f}")
    print(f"  {'24:00 储电量':>12}  {summary['soc_end']:10.2f}")
    print("表 3 紧急购电量（kWh）")
    if not summary["emergency"]:
        print("  （当日无紧急购电）")
    for label, kwh in summary["emergency"]:
        print(f"  {label:>12}  {kwh:10.2f}")
    if "point_charge" in summary:
        print("（对照）点预测解的充放电量与 24:00 储电量")
        for name, c, dis in zip(
            summary["interval_labels"], summary["point_charge"], summary["point_discharge"]
        ):
            print(f"  {name:>12}  {c:10.2f}  {dis:10.2f}")
        print(f"  {'24:00 储电量':>12}  {summary['point_soc_end']:10.2f}")
    print()


def write_result2(days: Sequence, path: str = "results/result2.xlsx") -> str:
    """写出 result2.xlsx：『计划购电量』『充放电量』『紧急购电量』三张工作表。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "计划购电量"
    ws.append(["日期"] + SEG_LABELS)
    for day in days:
        ws.append([day.day.isoformat()] + [round(float(v), 4) for v in day.G0])

    ws2 = wb.create_sheet("充放电量")
    ws2.append(["日期"] + CHARGE_COLUMNS + ["0:00 储电量", "24:00 储电量"])
    for day in days:
        charge, discharge = interval_sums(day.C), interval_sums(day.D)
        row: list = [day.day.isoformat()]
        for c, dis in zip(charge, discharge):
            row += [round(float(c), 4), round(float(dis), 4)]
        row += [round(float(day.soc_start), 4), round(float(np.asarray(day.S)[-1]), 4)]
        ws2.append(row)

    ws3 = wb.create_sheet("紧急购电量")
    ws3.append(["日期", "紧急购电时间段", "紧急购电量"])
    for day in days:
        first = True
        for label, kwh in emergency_intervals(day.day, day.E):
            ws3.append([day.day.isoformat() if first else None, label, round(float(kwh), 4)])
            first = False

    wb.save(path)
    return path
