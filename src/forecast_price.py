"""预测/场景层：附件 4 波动电价的点预测与「电价来源」抽象。

信息假设（论文声明）：附件 4 是事后真实电价，只用于结算与「已发生段」的观测。
决策时刻 $(d, t_0)$ 只能看到日期 $<d$ 的整日电价与当天 $\\tau<t_0$ 的段电价。
这条边界由两个入口统一把关：

* `PriceForecaster._rows`：所有对历史电价矩阵的整行读取都先经过它取行号，
  函数内部断言取到的行日期严格早于 asof（与 `forecast.PointForecaster._rows` 同构）；
* `PriceForecaster.observed`：当天唯一的真值入口，只切 `[:t_now]`。

`truth(d)` 返回整日真值，只允许结算与执行层调用。

点预测：$\\hat p_t = \\lambda\\,\\bar p_t$，其中基准 $\\bar p$ 为日期 $<d$ 的最近 $K_p$ 个
同类型日逐段均值（冷启动退回附件 1），日水平因子 $\\lambda$ 由当天已观测段做最小二乘
$\\lambda = \\sum_{\\tau<t_0} p_\\tau\\bar p_\\tau / \\sum_{\\tau<t_0}\\bar p_\\tau^2$ 并裁剪到
$[0.7, 1.3]$；$t_0=0$ 时 $\\lambda=1$（附件 4 的 $\\lambda$ lag-1 相关仅 0.52，昨日因子无用）。
次日曲线 = 次日类型的基准 × 当前 $\\lambda$。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np

from .data import T, Attachment1, day_type

K_PRICE = 4  # 默认 $K_p$：最近同类型日的个数
LAMBDA_LO, LAMBDA_HI = 0.7, 1.3  # 日水平因子的裁剪区间


class PriceSource(Protocol):
    """优化核与结算层看到的电价接口。

    `varies` 为 False 时电价逐日恒定（问 2 / 问 3 的附件 1 常数电价），
    调用方可以据此走「一次装配、不再更新代价向量」的快路径。
    """

    varies: bool

    def truth(self, d: date) -> np.ndarray:
        """日期 d 的 144 段结算电价（事后真值）。"""

    def predict(
        self, d: date, t_now: int = 0, observed: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """决策时刻 $(d, t_{now})$ 可得的 (当天 144 段, 次日 144 段) 电价曲线。

        当天前 $t_{now}$ 段填已观测真值，其余为点预测。
        """

    def residual(self, d: date, t_now: int = 0) -> np.ndarray | None:
        """当天 $[t_{now}, 144)$ 段的「真值 − 该时刻点预测」，常数电价返回 None。"""


@dataclass
class ConstantPriceSource:
    """附件 1 的常数电价：问 2 / 问 3 的默认电价来源，逐日相同、无残差场景。"""

    price: np.ndarray
    varies: bool = False

    def truth(self, d: date) -> np.ndarray:
        del d
        return self.price

    def observed(self, d: date, t_now: int) -> np.ndarray:
        del d
        return self.price[: int(t_now)]

    def predict(
        self, d: date, t_now: int = 0, observed: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del d, t_now, observed
        return self.price, self.price

    def residual(self, d: date, t_now: int = 0) -> np.ndarray | None:
        del d, t_now
        return None


class PriceForecaster:
    """附件 4 电价的 walk-forward 点预测器（决策侧唯一的电价入口）。"""

    varies = True

    def __init__(
        self,
        dates: list[date],
        prices: np.ndarray,
        prior: Attachment1,
        k: int = K_PRICE,
        *,
        lambda_lo: float = LAMBDA_LO,
        lambda_hi: float = LAMBDA_HI,
    ):
        self.dates = list(dates)
        self._price = np.asarray(prices, dtype=float)
        assert self._price.shape[0] == len(self.dates)
        self._types = np.array([day_type(d) for d in self.dates], dtype=int)
        self._ord = np.array([d.toordinal() for d in self.dates], dtype=int)
        self._row = {d: i for i, d in enumerate(self.dates)}
        self.prior = np.asarray(prior.price, dtype=float)
        self.k = int(k)
        self.lambda_lo = float(lambda_lo)
        self.lambda_hi = float(lambda_hi)
        self._baseline_cache: dict[tuple[int, int], np.ndarray] = {}

    # -- 信息边界 -------------------------------------------------------------

    def _rows(self, asof: date, k: int, dtype: int) -> np.ndarray:
        """最近 k 个同类型历史日的行号；只含日期严格早于 asof 的行。"""
        avail = (self._ord < asof.toordinal()) & (self._types == dtype)
        rows = np.flatnonzero(avail)[-k:]
        assert rows.size == 0 or int(self._ord[rows].max()) < asof.toordinal(), (
            "walk-forward 违例：取到了不早于决策日的历史电价行"
        )
        return rows

    def observed(self, d: date, t_now: int) -> np.ndarray:
        """当天已发生的段电价 $p_{d,\\tau},\\ \\tau<t_{now}$（决策侧唯一的当天真值入口）。"""
        t_now = int(t_now)
        assert 0 <= t_now <= T, f"段下标越界：{t_now}"
        if t_now == 0:
            return np.zeros(0)
        return self._price[self._row[d], :t_now]

    def truth(self, d: date) -> np.ndarray:
        """整日结算电价；只允许结算与执行层调用，不得进入预测路径。"""
        return self._price[self._row[d]]

    # -- 点预测 ---------------------------------------------------------------

    def baseline(self, d: date, asof: date | None = None) -> np.ndarray:
        """日期 $<$ asof（默认 d）的最近 $K_p$ 个 d 类型日逐段均值；无历史时用附件 1。"""
        cutoff = d if asof is None else asof
        key = (cutoff.toordinal(), day_type(d))
        hit = self._baseline_cache.get(key)
        if hit is not None:
            return hit
        rows = self._rows(cutoff, self.k, day_type(d))
        base = self._price[rows].mean(axis=0) if rows.size else self.prior.copy()
        base = np.asarray(base, dtype=float)
        self._baseline_cache[key] = base
        return base

    def level_factor(self, base: np.ndarray, observed: np.ndarray) -> float:
        """已观测段对基准的最小二乘水平因子 $\\lambda$，裁剪到 $[0.7, 1.3]$。"""
        observed = np.asarray(observed, dtype=float).ravel()
        n = observed.size
        if n == 0:
            return 1.0
        b = np.asarray(base, dtype=float)[:n]
        denom = float(b @ b)
        if denom <= 0.0:
            return 1.0
        return float(np.clip(float(observed @ b) / denom, self.lambda_lo, self.lambda_hi))

    def predict(
        self, d: date, t_now: int = 0, observed: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """决策时刻 $(d, t_{now})$ 的 (当天 144 段, 次日 144 段) 电价曲线。"""
        t_now = int(t_now)
        obs = self.observed(d, t_now) if observed is None else np.asarray(observed, float)
        assert obs.size == t_now, "已观测段数与决策时刻不符"
        base = self.baseline(d, asof=d)
        lam = self.level_factor(base, obs)
        today = np.empty(T)
        today[:t_now] = obs
        today[t_now:] = np.clip(lam * base[t_now:], 0.0, None)
        base_next = self.baseline(d + timedelta(days=1), asof=d)
        return today, np.clip(lam * base_next, 0.0, None)

    def residual(self, d: date, t_now: int = 0) -> np.ndarray:
        """当天 $[t_{now}, 144)$ 段的「真值 − 该时刻点预测」，用于电价残差库。"""
        t_now = int(t_now)
        today, _ = self.predict(d, t_now)
        return self.truth(d)[t_now:] - today[t_now:]


class PerfectPriceSource:
    """完美电价信息变体：决策时直接看到当天与次日的真实电价（费用下界对比用）。"""

    varies = True

    def __init__(self, dates: list[date], prices: np.ndarray):
        self.dates = list(dates)
        self._price = np.asarray(prices, dtype=float)
        self._row = {d: i for i, d in enumerate(self.dates)}
        self._last = self.dates[-1]

    def truth(self, d: date) -> np.ndarray:
        return self._price[self._row[d]]

    def observed(self, d: date, t_now: int) -> np.ndarray:
        return self._price[self._row[d], : int(t_now)]

    def predict(
        self, d: date, t_now: int = 0, observed: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del t_now, observed
        nxt = d + timedelta(days=1)
        next_price = self._price[self._row[nxt]] if nxt in self._row else self.truth(d)
        return self.truth(d), next_price

    def residual(self, d: date, t_now: int = 0) -> np.ndarray | None:
        """完美信息下点预测无误差，故不生成电价残差场景。"""
        del d, t_now
        return None


def horizon_price(
    src: PriceSource, d: date, t_now: int = 0, n_days: int = 2
) -> np.ndarray:
    """决策时刻 $(d, t_{now})$ 的 $24n$ 小时电价向量：当天 144 段 ‖ 次日曲线重复 $n-1$ 次。"""
    today, nxt = src.predict(d, t_now)
    today = np.asarray(today, dtype=float)
    if n_days <= 1:
        return today
    return np.concatenate([today] + [np.asarray(nxt, dtype=float)] * (n_days - 1))


__all__ = [
    "K_PRICE",
    "LAMBDA_HI",
    "LAMBDA_LO",
    "ConstantPriceSource",
    "PerfectPriceSource",
    "PriceForecaster",
    "PriceSource",
    "horizon_price",
]
