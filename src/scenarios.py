"""预测/场景层：经验残差场景库。

每天执行结束后把「当日真值 − 当日点预测」的残差轨迹追加进库（O(1) 摊销），
决策日取最近 M 个同类型历史日的残差平移到当日点预测上得到等权场景。
负载、光伏与（问 4 的）电价残差取自同一个历史日，保留三者的同日相关性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .data import day_type

M_SCEN = 12  # 默认场景数


def _price_scenarios(
    chosen: list["ResidualEntry"],
    residual_of,
    price_today: np.ndarray,
    price_future: np.ndarray,
) -> np.ndarray | None:
    """把同一批历史日的电价残差平移到点预测上，拼出 (M', n_today + n_future) 场景矩阵。

    任何一个被选中的历史日缺电价残差就返回 None：场景的三个通道必须来自同一天，
    宁可退化为确定电价，也不允许错位配对。
    """
    if not chosen:
        return None
    residuals = [residual_of(e) for e in chosen]
    if any(r is None for r in residuals):
        return None
    price_today = np.asarray(price_today, dtype=float)
    price_future = np.asarray(price_future, dtype=float).ravel()
    r = np.stack([np.asarray(r, dtype=float) for r in residuals])
    assert r.shape[1] == price_today.size, "电价残差长度与点预测曲线不符"
    today = np.clip(price_today[None, :] + r, 0.0, None)
    if price_future.size == 0:
        return today
    return np.concatenate([today, np.tile(price_future, (len(chosen), 1))], axis=1)


@dataclass
class ResidualEntry:
    """一个历史日的点预测残差轨迹。`r_price` 只在问 4 的波动电价下存在。"""

    day: date
    day_type: int
    r_load: np.ndarray  # (T,)
    r_pv: np.ndarray  # (T,)
    r_price: np.ndarray | None = None  # (T,)：真值 − 当日 0:00 的电价点预测


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
        r_price: np.ndarray | None = None,
    ) -> ResidualEntry:
        """追加日期 d 的残差；load_pred/pv_pred 必须是当时（用 < d 数据）作出的点预测。

        `r_price` 是同一天已算好的电价残差（真值 − 当日 0:00 点预测），常数电价下为 None。
        """
        entry = ResidualEntry(
            day=d,
            day_type=day_type(d),
            r_load=np.asarray(load_true, dtype=float) - np.asarray(load_pred, dtype=float),
            r_pv=np.asarray(pv_true, dtype=float) - np.asarray(pv_pred, dtype=float),
            r_price=None if r_price is None else np.asarray(r_price, dtype=float),
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

    def price_scenarios(
        self,
        d: date,
        price_today: np.ndarray,
        price_future: np.ndarray,
        m: int = M_SCEN,
    ) -> np.ndarray | None:
        """与 `scenarios` 同批历史日的电价场景 (M', T + price_future.size)。

        $p_{\\omega,t} = \\mathrm{clip}(\\hat p_t + r^p_{h_\\omega},\\ 0,\\ \\infty)$，
        $h_\\omega$ 与该场景的负载、光伏残差是同一个历史日。次日曲线不加残差
        （与负载 / 光伏的次日点预测曲线一致，都是跨场景共享的确定值）。
        没有可用电价残差时返回 None，调用方退化为确定电价。
        """
        return _price_scenarios(
            self.select(d, m), lambda e: e.r_price, price_today, price_future
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
    _rp: dict[tuple[int, date], np.ndarray] = field(default_factory=dict)

    def update(
        self,
        d: date,
        issue_hour: int,
        residual_segments: np.ndarray,
        price_residual: np.ndarray | None = None,
    ) -> None:
        """追加日期 d、发布时刻 issue_hour 的光伏（与问 4 的电价）残差，长度 $144-6h_0$。"""
        self._r[(int(issue_hour), d)] = np.asarray(residual_segments, dtype=float)
        if price_residual is not None:
            self._rp[(int(issue_hour), d)] = np.asarray(price_residual, dtype=float)

    def _chosen(self, d: date, issue_hour: int, m: int) -> list[ResidualEntry]:
        """该发布时刻可用的历史日（光伏残差已入库），三个通道共用同一批。"""
        return [e for e in self.load.select(d, m) if (int(issue_hour), e.day) in self._r]

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
        chosen = self._chosen(d, issue_hour, m)
        if not chosen:
            return load_pred[None, :].copy(), pv_pred[None, :].copy()
        r_load = np.stack([e.r_load[-n:] for e in chosen])
        r_pv = np.stack([self._r[(int(issue_hour), e.day)] for e in chosen])
        assert r_pv.shape[1] == n, "光伏残差长度与剩余段数不符"
        return (
            np.clip(load_pred[None, :] + r_load, 0.0, None),
            np.clip(pv_pred[None, :] + r_pv, 0.0, None),
        )

    def price_scenarios(
        self,
        d: date,
        issue_hour: int,
        price_today: np.ndarray,
        price_future: np.ndarray,
        m: int = M_SCEN,
    ) -> np.ndarray | None:
        """与 `scenarios` 同批历史日的电价场景；`price_today` 已切到剩余段。"""
        key = int(issue_hour)
        return _price_scenarios(
            self._chosen(d, key, m),
            lambda e: self._rp.get((key, e.day)),
            price_today,
            price_future,
        )


__all__ = ["M_SCEN", "ResidualEntry", "ResidualLibrary", "PVResidualLibraryByIssue"]
