"""执行层：以真值逐段运行 DP 价值执行器。

主路径的 ``DPValueExecutor`` 从场景未来费用函数得到缺口段保留 SOC；富余时只使用
当段可用富余电量充电，缺口时只放电并让紧急购电补足余额。旧 ``StorageRollingSolver``
调用签名仍被支持，供论文对照和既有实验复用。``t_start``/``t_end`` 支持问 3 的分块执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np

from .data import ETA, SOC_MAX, SOC_MIN, T, P_MAX_KWH
from .optimizer import StorageRollingSolver
from .value_dp import reserve_level

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
    R: np.ndarray | None = None

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
            R=np.full(n_seg, np.nan),
        )


def run_day_rolling_lp(
    d: date,
    g_today: np.ndarray,
    soc_init: float,
    load_true: np.ndarray,
    pv_true: np.ndarray,
    load_pred: np.ndarray | None = None,
    pv_pred: np.ndarray | object | None = None,
    load_future: np.ndarray | None = None,
    pv_future: np.ndarray | None = None,
    solver=None,
    step_minutes: int = 10,
    n_seg: int = T,
    *,
    t_start: int = 0,
    t_end: int | None = None,
    out: DayExecution | None = None,
    price_fn: Callable[[int], float | np.ndarray] | None = None,
) -> DayExecution:
    """执行日期 d 的段区间 `[t_start, t_end)`（默认整天），返回逐段执行值。

    `soc_init` 是区间起点前的储电量 $S_{t_{start}-1}$；`out` 给出时就地写入该区间，
    使问 3 能在 0/36/72/108 处暂停、重优化后带着当前 SOC 续跑同一天。
    `price_fn(t)` 给出段 t 开始时可得的 48h 电价向量（问 4 的波动电价），缺省沿用
    滚动 LP 构造时的电价。
    """
    if solver is None:
        raise TypeError("run_day_rolling_lp 需要 StorageRollingSolver")
    load_pred = np.asarray(load_true if load_pred is None else load_pred, dtype=float)
    pv_pred = np.asarray(pv_true if pv_pred is None else pv_pred, dtype=float)
    load_future = np.zeros(0) if load_future is None else np.asarray(load_future, dtype=float)
    pv_future = np.zeros(0) if pv_future is None else np.asarray(pv_future, dtype=float)
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
            None if price_fn is None else price_fn(t),
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


def run_day_dp(
    d: date,
    g_today: np.ndarray,
    soc_init: float,
    load_true: np.ndarray,
    pv_true: np.ndarray,
    price_true: np.ndarray | float,
    executor: "DPValueExecutor",
    *,
    t_start: int = 0,
    t_end: int | None = None,
    out: DayExecution | None = None,
    price_fn: Callable[[int], float | np.ndarray] | None = None,
) -> DayExecution:
    """Execute a day with the native-cadence DP value executor."""
    if not isinstance(executor, DPValueExecutor):
        raise TypeError("run_day_dp 需要 DPValueExecutor")
    if executor.price is None and price_true is not None:
        executor.price = np.asarray(price_true, dtype=float)
    load_true = np.asarray(load_true, float)
    pv_true = np.asarray(pv_true, float)
    g_today = np.asarray(g_today, float)
    t_end = T if t_end is None else int(t_end)
    ex = DayExecution.empty(d, soc_init, T) if out is None else out
    soc = float(soc_init)
    for t in range(int(t_start), t_end):
        price = executor.price_at(t, price_fn)
        c0, d0, r0 = executor.step(t, soc, load_true[t], pv_true[t], g_today[t], price)
        c = 0.0 if c0 < CLIP_TOL else float(c0)
        dis = 0.0 if d0 < CLIP_TOL else float(d0)
        net = load_true[t] + c - g_today[t] - pv_true[t] - dis
        ex.C[t], ex.D[t] = c, dis
        ex.E[t], ex.W[t] = max(net, 0.0), max(-net, 0.0)
        soc += ETA * c - dis / ETA
        assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6, f"{d} 段 {t} 储电量越界：{soc}"
        soc = min(max(soc, SOC_MIN), SOC_MAX)
        ex.S[t] = soc
        if ex.R is not None:
            ex.R[t] = float(r0)
    return ex


def run_day(
    d: date,
    g_today: np.ndarray,
    soc_init: float,
    load_true: np.ndarray,
    pv_true: np.ndarray,
    load_pred: np.ndarray | None = None,
    pv_pred: np.ndarray | object | None = None,
    load_future: np.ndarray | None = None,
    pv_future: np.ndarray | None = None,
    solver=None,
    step_minutes: int = 10,
    n_seg: int = T,
    *,
    t_start: int = 0,
    t_end: int | None = None,
    out: DayExecution | None = None,
    price_fn: Callable[[int], float | np.ndarray] | None = None,
) -> DayExecution:
    """Backward-compatible dispatcher for DP and legacy rolling-LP callers."""
    # Compact DP form: run_day(d, g, soc, load, pv, price, executor).
    if solver is None and isinstance(pv_pred, DPValueExecutor):
        executor = pv_pred
        price_true = load_pred if load_pred is not None else executor.price
        if executor.price is None and price_true is not None:
            executor.price = np.asarray(price_true, float)
        return run_day_dp(d, g_today, soc_init, load_true, pv_true, price_true, executor,
                          t_start=t_start, t_end=t_end, out=out, price_fn=price_fn)
    if isinstance(solver, DPValueExecutor):
        return run_day_dp(d, g_today, soc_init, load_true, pv_true,
                          load_pred if load_pred is not None else solver.price,
                          solver, t_start=t_start, t_end=t_end, out=out, price_fn=price_fn)
    return run_day_rolling_lp(d, g_today, soc_init, load_true, pv_true, load_pred,
                              pv_pred, load_future, pv_future, solver, step_minutes,
                              n_seg, t_start=t_start, t_end=t_end, out=out, price_fn=price_fn)


class DPValueExecutor:
    """按未来费用函数执行的因果储能控制器。"""
    def __init__(self, hbar=None, *, price=None):
        self.hbar = hbar or []
        self.price = None if price is None else np.asarray(price, float)
    def prepare(self, t0=0, hbar=None):
        self.t0 = int(t0)
        if hbar is not None: self.hbar = hbar
        return self
    def price_at(self, t, price_fn=None):
        if price_fn is not None:
            a = np.asarray(price_fn(t), float)
            return float(a) if a.ndim == 0 else float(a[t] if a.size > t else a[0])
        j = t - getattr(self, "t0", 0)
        return float(self.price[j]) if self.price is not None and j < self.price.size else 1.0
    def step(self, t, soc_prev, load, pv, g, price):
        r=float(load-pv-g)
        if r <= 0:
            return min(-r, P_MAX_KWH, (SOC_MAX-soc_prev)/ETA), 0.0, np.nan
        reserve=SOC_MIN
        j = t - getattr(self, "t0", 0)
        if self.hbar and j+1 < len(self.hbar): reserve=reserve_level(self.hbar[j+1], price)
        return 0.0, min(r, P_MAX_KWH, ETA*max(soc_prev-reserve,0.0)), reserve


__all__ = ["DayExecution", "run_day", "run_day_dp", "run_day_rolling_lp", "DPValueExecutor"]
