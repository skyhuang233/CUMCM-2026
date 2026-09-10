"""预测/场景层：walk-forward 点预测。

决策日 d 的任何预测只允许使用日期严格早于 d 的附件 2 数据。这条信息假设由
`PointForecaster._rows` 统一把关：所有对历史矩阵的读取都必须先经过它取行号，
函数内部断言取到的行日期 < asof。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .data import Attachment1, DailySeries, day_type


class PointForecaster:
    """负载 / 光伏的逐段点预测器。

    负载：日期 < asof 的最近 `k_load` 个同类型日的逐段均值。
    光伏：日期 < asof 的最近 `k_pv` 天的逐段均值（不分日类型）。
    冷启动：可用历史不足 k 时用全部可用；完全没有可用历史时退回附件 1 曲线。
    """

    def __init__(
        self,
        daily: DailySeries,
        prior: Attachment1,
        k_load: int = 4,
        k_pv: int = 5,
    ):
        self.dates = list(daily.dates)
        self._load = np.asarray(daily.load_kwh, dtype=float)
        self._pv = np.asarray(daily.pv_kwh, dtype=float)
        self._types = np.array([day_type(d) for d in self.dates], dtype=int)
        self._ord = np.array([d.toordinal() for d in self.dates], dtype=int)
        self.prior = prior
        self.k_load = int(k_load)
        self.k_pv = int(k_pv)

    def _rows(self, asof: date, k: int, dtype: int | None = None) -> np.ndarray:
        """最近 k 个可用历史日的行号；只含日期严格早于 asof 的行。"""
        avail = self._ord < asof.toordinal()
        if dtype is not None:
            avail &= self._types == dtype
        rows = np.flatnonzero(avail)[-k:]
        assert rows.size == 0 or int(self._ord[rows].max()) < asof.toordinal(), (
            "walk-forward 违例：取到了不早于决策日的历史行"
        )
        return rows

    def predict(self, d: date, asof: date | None = None) -> tuple[np.ndarray, np.ndarray]:
        """给出 d 的逐段负载 / 光伏点预测（kWh），只用日期 < asof（默认 d）的数据。"""
        cutoff = d if asof is None else asof
        rows_l = self._rows(cutoff, self.k_load, day_type(d))
        rows_p = self._rows(cutoff, self.k_pv)
        load = self._load[rows_l].mean(axis=0) if rows_l.size else self.prior.load_kwh.copy()
        pv = self._pv[rows_p].mean(axis=0) if rows_p.size else self.prior.pv_kwh.copy()
        return np.asarray(load, dtype=float), np.asarray(pv, dtype=float)

    def predict_next(self, d: date) -> tuple[np.ndarray, np.ndarray]:
        """次日曲线：对 d+1 作点预测，但仍只用日期 < d 的数据。"""
        return self.predict(d + timedelta(days=1), asof=d)

    def predict_future(self, d: date, n_days: int) -> tuple[np.ndarray, np.ndarray]:
        """决策日之后 n_days 天的点预测曲线，拼接成 (n_days*T,) 两条曲线。"""
        if n_days <= 0:
            return np.zeros(0), np.zeros(0)
        loads, pvs = [], []
        for k in range(1, n_days + 1):
            load, pv = self.predict(d + timedelta(days=k), asof=d)
            loads.append(load)
            pvs.append(pv)
        return np.concatenate(loads), np.concatenate(pvs)


__all__ = ["PointForecaster"]
