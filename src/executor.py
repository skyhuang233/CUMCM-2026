"""执行层：以附件 2 真值逐段运行储能滚动控制。

每段开始时用当前储电量、该段真值与剩余时域点预测求解储能滚动 LP，只执行首段的
充放电；紧急购电量与富余电量由该段真值下的平衡式反解，因此恒有 E_t·W_t = 0。
`step_minutes` > 10 时每 k = step_minutes/10 段重解一次，并执行该解的前 k 段。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .data import ETA, SOC_MAX, SOC_MIN, T
from .optimizer import StorageRollingSolver

CLIP_TOL = 1e-9


@dataclass
class DayExecution:
    """一天 144 段的执行值（kWh）。分块执行时由多次 `run_day` 共同填满。"""

    day: date
    C: np.ndarray
    D: np.ndarray
    S: np.ndarray
    E: np.ndarray
    W: np.ndarray
    soc_start: float

    @property
    def soc_end(self) -> float:
        return float(self.S[-1])

    @classmethod
    def empty(cls, day: date, soc_start: float, n_seg: int = T) -> "DayExecution":
        """全零的一天，供分段执行时逐块写入。"""
        return cls(
            day=day,
            C=np.zeros(n_seg),
            D=np.zeros(n_seg),
            S=np.zeros(n_seg),
            E=np.zeros(n_seg),
            W=np.zeros(n_seg),
            soc_start=float(soc_start),
        )


def run_day(
    d: date,
    g_today: np.ndarray,
    soc_init: float,
    load_true: np.ndarray,
    pv_true: np.ndarray,
    load_pred: np.ndarray,
    pv_pred: np.ndarray,
    load_future: np.ndarray,
    pv_future: np.ndarray,
    solver: StorageRollingSolver,
    step_minutes: int = 10,
    n_seg: int = T,
    *,
    t_start: int = 0,
    t_end: int | None = None,
    out: DayExecution | None = None,
) -> DayExecution:
    """执行日期 d 的段区间 `[t_start, t_end)`（默认整天），返回逐段执行值。

    `soc_init` 是区间起点前的储电量 $S_{t_{start}-1}$；`out` 给出时就地写入该区间，
    使问 3 能在 0/36/72/108 处暂停、重优化后带着当前 SOC 续跑同一天。
    """
    g_today = np.asarray(g_today, dtype=float)
    load_true = np.asarray(load_true, dtype=float)
    pv_true = np.asarray(pv_true, dtype=float)
    k = max(1, round(step_minutes / 10))
    t_end = n_seg if t_end is None else int(t_end)

    ex = DayExecution.empty(d, soc_init, n_seg) if out is None else out
    C, D, S, E, W = ex.C, ex.D, ex.S, ex.E, ex.W
    soc = float(soc_init)

    t = int(t_start)
    while t < t_end:
        step = solver.solve(
            t,
            soc,
            load_true[t],
            pv_true[t],
            load_pred,
            pv_pred,
            load_future,
            pv_future,
            g_today,
        )
        for tau in range(t, min(t + k, t_end)):
            c = float(step.C[tau])
            dis = float(step.D[tau])
            c = 0.0 if c < CLIP_TOL else c
            dis = 0.0 if dis < CLIP_TOL else dis
            # 真值平衡：net > 0 缺口 → 紧急购电；net < 0 富余 → 富余电量
            net = load_true[tau] + c - g_today[tau] - pv_true[tau] - dis
            C[tau] = c
            D[tau] = dis
            E[tau] = max(net, 0.0)
            W[tau] = max(-net, 0.0)
            soc = soc + ETA * c - dis / ETA
            assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6, f"{d} 段 {tau} 储电量越界：{soc}"
            soc = min(max(soc, SOC_MIN), SOC_MAX)  # 只压掉 1e-12 级数值噪声
            S[tau] = soc
        t += k

    return ex


__all__ = ["DayExecution", "run_day"]
