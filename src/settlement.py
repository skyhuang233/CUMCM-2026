"""结算层：问 2 / 问 3 的费用分解与紧急购电区间合并。

问 2 每段费用 Cost_t = p_t G^0_t + 5 p_t E_t；计划购电按交易时刻电价全额计费，
紧急购电按同一电价的 5 倍计费。问 3 起计划购电量与调整购电量分段计费（见 `cost_q3`）。
紧急购电输出时按题面表 4 合并为连续时间区间。
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .optimizer import ADJUST_DOWN, ADJUST_UP

TOL_KWH = 1e-6  # 判定「该段发生紧急购电」的阈值


def cost_q2(price: np.ndarray, G0: np.ndarray, E: np.ndarray) -> tuple[float, float]:
    """返回 (计划购电费, 紧急购电费)，两者之和为该日总费用。"""
    price = np.asarray(price, dtype=float)
    return float(price @ np.asarray(G0, dtype=float)), float(
        5.0 * price @ np.asarray(E, dtype=float)
    )


def cost_q3(
    price: np.ndarray, G0: np.ndarray, Ga: np.ndarray, E: np.ndarray
) -> dict[str, float]:
    """问 3 每段费用的四项分解，四项之和等于总费用。

    $$\\text{Cost}_t = p_t\\min(G^0_t, G^a_t) + 0.5p_t(G^0_t-G^a_t)^+
      + 1.5p_t(G^a_t-G^0_t)^+ + 5p_tE_t$$

    经济含义：减购时取消的部分承担 50% 违约损失，增购时超出部分按 1.5 倍购买。
    返回键 `plan`（计划费）、`curtail_penalty`（减购违约费）、`extra`（增购费）、
    `emergency`（紧急购电费）、`total`。
    """
    price = np.asarray(price, dtype=float)
    G0 = np.asarray(G0, dtype=float)
    Ga = np.asarray(Ga, dtype=float)
    E = np.asarray(E, dtype=float)
    plan = float(price @ np.minimum(G0, Ga))
    curtail = float(ADJUST_DOWN * price @ np.maximum(G0 - Ga, 0.0))
    extra = float(ADJUST_UP * price @ np.maximum(Ga - G0, 0.0))
    emergency = float(5.0 * price @ E)
    return {
        "plan": plan,
        "curtail_penalty": curtail,
        "extra": extra,
        "emergency": emergency,
        "total": plan + curtail + extra + emergency,
    }


def seg_time_label(minutes: int) -> str:
    """分钟偏移 → 'H:MM'；1440 分钟记为 '24:00'。"""
    return f"{minutes // 60}:{minutes % 60:02d}"


def interval_label(t_start: int, t_end: int) -> str:
    """段 [t_start, t_end]（闭区间）→ 'HH:MM-HH:MM'，起点为首段起始、终点为末段结束。"""
    return f"{seg_time_label(10 * t_start)}-{seg_time_label(10 * (t_end + 1))}"


def merge_emergency_intervals(
    E: np.ndarray, tol: float = TOL_KWH
) -> list[tuple[int, int, float]]:
    """把 E_t > tol 的连续段合并为 (首段, 末段, 区间电量) 列表。"""
    E = np.asarray(E, dtype=float)
    active = E > tol
    out: list[tuple[int, int, float]] = []
    start = None
    for t, flag in enumerate(active):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            out.append((start, t - 1, float(E[start:t].sum())))
            start = None
    if start is not None:
        out.append((start, E.size - 1, float(E[start:].sum())))
    return out


def emergency_intervals(
    d: date, E: np.ndarray, tol: float = TOL_KWH
) -> list[tuple[str, float]]:
    """该日紧急购电区间：[(‘13:00-13:30’, kWh), ...]，格式同题面表 4。"""
    del d  # 日期由调用方填入表格首列，这里只负责区间与电量
    return [
        (interval_label(a, b), kwh) for a, b, kwh in merge_emergency_intervals(E, tol)
    ]


__all__ = [
    "TOL_KWH",
    "cost_q2",
    "cost_q3",
    "emergency_intervals",
    "interval_label",
    "merge_emergency_intervals",
    "seg_time_label",
]
