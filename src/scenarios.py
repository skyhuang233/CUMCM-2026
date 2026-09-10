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
        pool = [e for e in self._by_type.get(day_type(d), []) if e.day < d]
        chosen = pool[-int(m):] if m > 0 else []
        if not chosen:
            return load_pred[None, :].copy(), pv_pred[None, :].copy()
        r_load = np.stack([e.r_load for e in chosen])
        r_pv = np.stack([e.r_pv for e in chosen])
        return (
            np.clip(load_pred[None, :] + r_load, 0.0, None),
            np.clip(pv_pred[None, :] + r_pv, 0.0, None),
        )


__all__ = ["M_SCEN", "ResidualEntry", "ResidualLibrary"]
