"""预测/场景层：经验残差场景库。

每天执行结束后把「当日真值 − 当日点预测」的残差轨迹追加进库（O(1) 摊销），
决策日取最近 M 个同类型历史日的残差平移到当日点预测上得到等权场景。
负载与光伏的残差取自同一个历史日，保留两者的同日相关性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .data import day_type

M_SCEN = 12  # 默认场景数


@dataclass
class ResidualEntry:
    """一个历史日的点预测残差轨迹。"""

    day: date
    day_type: int
    r_load: np.ndarray  # (T,)
    r_pv: np.ndarray  # (T,)


@dataclass
class ResidualLibrary:
    """按日类型分组的经验残差库，增量维护。"""

    entries: list[ResidualEntry] = field(default_factory=list)
    _by_type: dict[int, list[ResidualEntry]] = field(default_factory=dict)

    def update(
        self,
        d: date,
        load_true: np.ndarray,
        pv_true: np.ndarray,
        load_pred: np.ndarray,
        pv_pred: np.ndarray,
    ) -> ResidualEntry:
        """追加日期 d 的残差；load_pred/pv_pred 必须是当时（用 < d 数据）作出的点预测。"""
        entry = ResidualEntry(
            day=d,
            day_type=day_type(d),
            r_load=np.asarray(load_true, dtype=float) - np.asarray(load_pred, dtype=float),
            r_pv=np.asarray(pv_true, dtype=float) - np.asarray(pv_pred, dtype=float),
        )
        self.entries.append(entry)
        self._by_type.setdefault(entry.day_type, []).append(entry)
        return entry

    def select(self, d: date, m: int = M_SCEN) -> list[ResidualEntry]:
        """决策日 d 可用的最近 m 个同类型历史日残差（日期严格早于 d）。"""
        pool = [e for e in self._by_type.get(day_type(d), []) if e.day < d]
        return pool[-int(m):] if m > 0 else []

    def scenarios(
        self,
        d: date,
        load_pred: np.ndarray,
        pv_pred: np.ndarray,
        m: int = M_SCEN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """决策日 d 的等权场景曲线 (M',T)；M' = min(m, 可用同类型历史日数)。

        冷启动没有同类型残差时退化为单一的点预测场景。
        """
        load_pred = np.asarray(load_pred, dtype=float)
        pv_pred = np.asarray(pv_pred, dtype=float)
        chosen = self.select(d, m)
        if not chosen:
            return load_pred[None, :].copy(), pv_pred[None, :].copy()
        r_load = np.stack([e.r_load for e in chosen])
        r_pv = np.stack([e.r_pv for e in chosen])
        return (
            np.clip(load_pred[None, :] + r_load, 0.0, None),
            np.clip(pv_pred[None, :] + r_pv, 0.0, None),
        )


@dataclass
class PVResidualLibraryByIssue:
    """问 3：按发布时刻分库的光伏残差，与问 2 的负载残差按同一历史日配对。

    发布时刻 $h_0$ 的库只存当天 $[6h_0, 144)$ 段的「真值 − 该时刻插值预报」，
    因此 0:00 库 144 段、6:00 库 108 段、12:00 库 72 段、18:00 库 36 段。
    场景选择规则与问 2 一致：取最近 $M$ 个同类型历史日；负载残差取自 `load` 库的同一天，
    保留负载与光伏的同日相关性。
    """

    load: ResidualLibrary
    _r: dict[tuple[int, date], np.ndarray] = field(default_factory=dict)

    def update(self, d: date, issue_hour: int, residual_segments: np.ndarray) -> None:
        """追加日期 d、发布时刻 issue_hour 的光伏残差（长度 $144-6h_0$）。"""
        self._r[(int(issue_hour), d)] = np.asarray(residual_segments, dtype=float)

    def scenarios(
        self,
        d: date,
        issue_hour: int,
        load_pred: np.ndarray,
        pv_pred: np.ndarray,
        m: int = M_SCEN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """决策时刻 (d, issue_hour) 的等权场景 (M', n)；load_pred/pv_pred 已切到剩余段。"""
        load_pred = np.asarray(load_pred, dtype=float)
        pv_pred = np.asarray(pv_pred, dtype=float)
        n = pv_pred.size
        chosen = [e for e in self.load.select(d, m) if (int(issue_hour), e.day) in self._r]
        if not chosen:
            return load_pred[None, :].copy(), pv_pred[None, :].copy()
        r_load = np.stack([e.r_load[-n:] for e in chosen])
        r_pv = np.stack([self._r[(int(issue_hour), e.day)] for e in chosen])
        assert r_pv.shape[1] == n, "光伏残差长度与剩余段数不符"
        return (
            np.clip(load_pred[None, :] + r_load, 0.0, None),
            np.clip(pv_pred[None, :] + r_pv, 0.0, None),
        )


__all__ = ["M_SCEN", "ResidualEntry", "ResidualLibrary", "PVResidualLibraryByIssue"]
