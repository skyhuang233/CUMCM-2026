"""预测/场景层：附件 3 的整点光伏预报 → 10 分钟段电量曲线。

附件 3 的「预报 k 小时」是发布时刻 $h_0$ 之后第 k 个整点的**瞬时功率**（kW），与附件 2
的第 $6k-1$ 段（段末恰为该整点）对齐（全年 MAE 187 kW；当作小时均值对齐则 354 kW）。
把发布时刻 $h_0$ 的已观测真值当作左端点，与 24 个整点预报一起在整点之间线性插值到每个
10 分钟段末，再乘 `DT_H` 得到该段电量（kWh），截断 $\\ge 0$。

信息假设：发布时刻 $h_0$ 只用日期 $\\le d$、时刻 $\\le h_0$ 的真值与该时刻发布的预报。
左端点对 $h_0=0$ 取前一日的 `0:00+1` 段（即 $PV_{d-1,143}$），其余取当天第 $6h_0-1$ 段。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .data import DT_H, T, DailySeries

SEG_PER_HOUR = 6
ISSUE_HOURS = (0, 6, 12, 18)
HORIZON_HOURS = 24  # 附件 3 每次发布覆盖未来 24 个整点


def issue_segment(issue_hour: int) -> int:
    """发布时刻 → 当天段下标 $t_0$（0:00→0, 6:00→36, 12:00→72, 18:00→108）。"""
    if int(issue_hour) != issue_hour or int(issue_hour) not in ISSUE_HOURS:
        raise ValueError(f"issue_hour must be one of {ISSUE_HOURS}")
    return int(issue_hour) * SEG_PER_HOUR


def interpolate_issue(forecast24: np.ndarray, left_anchor_kw: float) -> np.ndarray:
    """整点预报 + 左端点 → 覆盖 $[h_0, h_0+24h)$ 的 144 段电量（kWh）。

    锚点为小时偏移 0（左端点真值）与 1..24（预报值）；第 j 段（j=0..143）的段末在
    小时偏移 $(j+1)/6$ 处，线性插值后截断非负再乘 `DT_H`。
    """
    forecast24 = np.asarray(forecast24, dtype=float).ravel()
    assert forecast24.size == HORIZON_HOURS, f"附件 3 应有 {HORIZON_HOURS} 个整点预报"
    anchors = np.empty(HORIZON_HOURS + 1)
    anchors[0] = float(left_anchor_kw)
    anchors[1:] = forecast24
    hours = np.arange(1, T + 1) / SEG_PER_HOUR
    kw = np.interp(hours, np.arange(HORIZON_HOURS + 1), anchors)
    return np.clip(kw, 0.0, None) * DT_H


class PVIssueForecaster:
    """按发布时刻给出光伏点预测曲线（附件 3 插值 + 已观测真值 + $K_P$ 均值兜底）。"""

    def __init__(self, daily: DailySeries, attachment3: dict[tuple[date, int], np.ndarray]):
        self.dates = list(daily.dates)
        self._pv = np.asarray(daily.pv_kwh, dtype=float)
        self._row = {d: i for i, d in enumerate(self.dates)}
        self.att3 = attachment3

    def anchor_kw(self, d: date, issue_hour: int) -> float:
        """发布时刻的已观测瞬时功率（kW）：$h_0>0$ 取当天第 $6h_0-1$ 段，否则取前一日末段。"""
        if issue_hour:
            return float(self._pv[self._row[d], issue_segment(issue_hour) - 1]) / DT_H
        prev = d - timedelta(days=1)
        if prev not in self._row:
            return 0.0  # 全年首日之前无观测；0:00 光伏本就为零
        return float(self._pv[self._row[prev], T - 1]) / DT_H

    def interpolate(self, d: date, issue_hour: int) -> np.ndarray:
        """日期 d、发布时刻 $h_0$ 的插值曲线（kWh），覆盖绝对段 $[6h_0, 6h_0+144)$。"""
        return interpolate_issue(self.att3[(d, int(issue_hour))], self.anchor_kw(d, issue_hour))

    def residual(self, d: date, issue_hour: int) -> np.ndarray:
        """当天 $[6h_0, 144)$ 段的「真值 − 该时刻插值预报」，长度 $144-6h_0$。"""
        t0 = issue_segment(issue_hour)
        return self._pv[self._row[d], t0:] - self.interpolate(d, issue_hour)[: T - t0]

    def weight_at_zero(self, d: date, historical: np.ndarray) -> float:
        """Causal eq. (36) weight for a January 0:00 issue.

        ``historical`` is the three-day forecast for d.  Only prior completed
        paired forecasts are used; no pair means the specified w=0 fallback.
        """
        if d.month >= 2:
            return 0.388815
        ef, eh = [], []
        for old in self.dates:
            if old >= d or old.month != 1 or old == self.dates[0]:
                continue
            rows = [self._row[x] for x in self.dates if x < old][-3:]
            if not rows or (old, 0) not in self.att3:
                continue
            h = self._pv[rows].mean(0)
            f = self.interpolate(old, 0)
            ef_i = self._pv[self._row[old]] - f
            eh_i = self._pv[self._row[old]] - h
            if np.isfinite(ef_i).all() and np.isfinite(eh_i).all():
                ef.append(ef_i); eh.append(eh_i)
        if not ef: return 0.0
        ef, eh = np.concatenate(ef), np.concatenate(eh)
        den = float(np.sum((ef-eh)**2))
        return 0.0 if den <= 0 else float(np.clip(np.sum(eh*(eh-ef))/den, 0, 1))

    def combined_curves(self, d: date, issue_hour: int, pv_mean_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Formal issue forecast, with the approved 0:00 formal/history mix."""
        today, nxt = self.curves(d, issue_hour, pv_mean_next)
        if int(issue_hour) != 0: return today, nxt
        w = self.weight_at_zero(d, np.asarray(pv_mean_next, float))
        formal = self.interpolate(d, 0)
        # At 0:00 the historical three-day curve is the supplied mean, not
        # ``today`` (which `curves` filled with the formal issue forecast).
        hist = np.asarray(pv_mean_next, float)
        return np.clip(w*formal+(1-w)*hist, 0, None), nxt

    def path(self, d: date, issue_hour: int, pv_mean_next: np.ndarray) -> np.ndarray:
        """The complete 24h path published at an issue (can cross midnight)."""
        today, nxt = self.combined_curves(d, issue_hour, pv_mean_next)
        t0 = issue_segment(issue_hour)
        return np.concatenate([today[t0:], nxt[:t0]])

    def curves(
        self, d: date, issue_hour: int, pv_mean_next: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """发布时刻 $h_0$ 可得的 (当天 144 段, 次日 144 段) 光伏点预测（kWh）。

        当天：$t<6h_0$ 用真值，其余用插值预报。
        次日：$t<6h_0$ 用同一次发布跨日的部分，其余用 $K_P$ 日均值 `pv_mean_next`。
        """
        t0 = issue_segment(issue_hour)
        interp = self.interpolate(d, issue_hour)
        today = np.empty(T)
        today[:t0] = self._pv[self._row[d], :t0]
        today[t0:] = interp[: T - t0]
        nxt = np.asarray(pv_mean_next, dtype=float).copy()
        if t0:
            nxt[:t0] = interp[T - t0 :]
        return today, nxt


__all__ = [
    "HORIZON_HOURS",
    "ISSUE_HOURS",
    "SEG_PER_HOUR",
    "PVIssueForecaster",
    "interpolate_issue",
    "issue_segment",
]
