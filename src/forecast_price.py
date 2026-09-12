"""Causal Q4 day-level × intraday-shape price forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

import numpy as np

from .data import Attachment1, T, day_type

K_PRICE = 35
PRICE_FLOOR = 1e-4
LAMBDA_LO, LAMBDA_HI = 0.7, 1.3


class PriceSource(Protocol):
    varies: bool
    def truth(self, d: date) -> np.ndarray: ...
    def predict(self, d: date, t_now: int = 0, observed=None, **kw): ...
    def residual(self, d: date, t_now: int = 0): ...


@dataclass
class ConstantPriceSource:
    price: np.ndarray
    varies: bool = False
    def truth(self, d): return self.price
    def observed(self, d, t_now): return self.price[:int(t_now)]
    def predict(self, d, t_now=0, observed=None, **kw): return self.price, self.price
    def residual(self, d, t_now=0): return None


class PriceForecaster:
    """Baseline Q4 forecaster; ``mode='legacy_level'`` enables old increment."""
    varies = True

    def __init__(self, dates, prices, prior: Attachment1, k=K_PRICE, *,
                 branch="q4_2", net_load=None, mode="baseline"):
        if branch not in {"q4_2", "q4_3"}:
            raise ValueError("branch must be 'q4_2' or 'q4_3'")
        if mode not in {"baseline", "legacy_level"}:
            raise ValueError("unknown price forecast mode")
        self.dates = list(dates)
        self._price = np.asarray(prices, float)
        self._row = {d: i for i, d in enumerate(self.dates)}
        self._ord = np.array([d.toordinal() for d in self.dates])
        self.prior = np.asarray(prior.price, float)
        self.k, self.branch, self.net_load, self.mode = int(k), branch, net_load, mode

    def _validate_t(self, t_now):
        t = int(t_now)
        if t != t_now or not 0 <= t <= T:
            raise ValueError("t_now must be an integer segment in [0, T]")
        return t

    def _rows(self, asof, k=None):
        rows = np.flatnonzero(self._ord < asof.toordinal())[-(self.k if k is None else int(k)):]
        assert not rows.size or self._ord[rows].max() < asof.toordinal()
        return rows

    def observed(self, d, t_now):
        return self._price[self._row[d], :self._validate_t(t_now)].copy()

    def truth(self, d):
        return self._price[self._row[d]]

    def _shape(self, asof):
        rows = self._rows(asof)
        if not rows.size:
            return self.prior / max(float(self.prior.mean()), PRICE_FLOOR)
        levels = self._price[rows].mean(axis=1)
        return np.mean(self._price[rows] / np.maximum(levels[:, None], PRICE_FLOOR), axis=0)

    def _net(self, d, t_now):
        if self.net_load is None:
            return None
        try:
            return float(self.net_load(d, t_now))
        except TypeError:
            return float(self.net_load(d))

    def _fit(self, asof, branch, t_now):
        rows = self._rows(asof)
        if rows.size < 3:
            return None
        y = self._price[rows].mean(axis=1)
        days = [self.dates[i] for i in rows]
        if branch == "q4_3":
            x = np.array([self._price[self._row[d - timedelta(days=1)]].mean()
                          if d - timedelta(days=1) in self._row else np.nan for d in days])
        else:
            x = np.array([self._net(d, t_now) for d in days]) if self.net_load else np.full(len(days), np.nan)
            x = x / 1e5
        valid = np.isfinite(x)
        if valid.sum() < 2:
            return None
        design = np.column_stack([np.ones(valid.sum()), x[valid]])
        if np.linalg.matrix_rank(design) < 2:
            return None
        intercept, slope = np.linalg.lstsq(design, y[valid], rcond=None)[0]
        return float(intercept), float(np.clip(slope, 0, .99) if branch == "q4_3" else slope)

    def _rho_p(self, asof):
        rows = self._rows(asof)
        if rows.size < 3:
            return 0.0
        shape = self._shape(asof)
        residual = (self._price[rows] / np.maximum(self._price[rows].mean(1)[:, None], PRICE_FLOOR) - shape).ravel()
        denominator = float(residual[:-1] @ residual[:-1])
        return 0.0 if denominator == 0 else float(np.clip((residual[:-1] @ residual[1:]) / denominator, 0, .99))

    def baseline(self, d, asof=None):
        cutoff = d if asof is None else asof
        rows = self._rows(cutoff)
        level = self.prior.mean() if not rows.size else self._price[rows].mean()
        return self._shape(cutoff) * max(float(level), .01)

    def level_factor(self, base, observed):
        base, observed = np.asarray(base, float)[:len(observed)], np.asarray(observed, float)
        den = float(base @ base)
        if not len(observed) or den <= 0:
            return 1.0
        return float(np.clip((observed @ base) / den, LAMBDA_LO, LAMBDA_HI))

    def predict(self, d, t_now=0, observed=None, *, net_load_pred=None,
                net_load_next=None, branch=None):
        t = self._validate_t(t_now)
        active_branch = self.branch if branch is None else branch
        if active_branch not in {"q4_2", "q4_3"}:
            raise ValueError("invalid branch")
        obs = self.observed(d, t) if observed is None else np.asarray(observed, float)
        if obs.size != t:
            raise ValueError("observed must have exactly t_now values")
        if self.mode == "legacy_level":
            rows = np.flatnonzero(
                (self._ord < d.toordinal())
                & np.array([day_type(x) == day_type(d) for x in self.dates], dtype=bool)
            )[-self.k:]
            base = self._price[rows].mean(axis=0) if rows.size else self.prior.copy()
            lam = self.level_factor(base, obs)
            nxt_d = d + timedelta(days=1)
            rows = np.flatnonzero(
                (self._ord < d.toordinal())
                & np.array([day_type(x) == day_type(nxt_d) for x in self.dates], dtype=bool)
            )[-self.k:]
            nxt_base = self._price[rows].mean(axis=0) if rows.size else self.prior.copy()
            today = np.maximum(lam * base, PRICE_FLOOR)
            today[:t] = obs
            return today, np.maximum(lam * nxt_base, PRICE_FLOOR)
        fit = self._fit(d, active_branch, t)
        yesterday = self._price[self._row[d-timedelta(days=1)]].mean() if d-timedelta(days=1) in self._row else self.prior.mean()
        if fit is None:
            level = yesterday
        elif active_branch == "q4_3":
            level = fit[0] + fit[1] * yesterday
        else:
            feature = self._net(d, t) if net_load_pred is None else net_load_pred
            level = fit[0] + fit[1] * (0.0 if feature is None else float(feature)) / 1e5
        level = max(float(level), .01)
        today = np.maximum(level * self._shape(d), PRICE_FLOOR)
        rho, error = self._rho_p(d), 0.0
        if t:
            error = float(obs[-1] - today[t-1])
            today[t:] = np.maximum(today[t:] + rho**np.arange(1, T-t+1)*error, PRICE_FLOOR)
        today[:t] = obs
        if fit is not None and active_branch == "q4_3":
            next_level = fit[0] + fit[1] * level
        elif fit is not None and net_load_next is not None:
            next_level = fit[0] + fit[1] * float(net_load_next) / 1e5
        else:
            next_level = level
        nxt = np.maximum(max(float(next_level), .01) * self._shape(d), PRICE_FLOOR)
        if t:
            nxt = np.maximum(nxt + rho**np.arange(T-t+1, 2*T-t+1)*error, PRICE_FLOOR)
        return today, nxt

    def residual(self, d, t_now=0):
        t = self._validate_t(t_now)
        predicted, _ = self.predict(d, t)
        return self.truth(d)[t:] - predicted[t:]

    def terminal_value(self, price_path, t_now=0):
        t = self._validate_t(t_now)
        path = np.asarray(price_path, float)
        if path.ndim == 2:
            path = path.mean(axis=0)
        if path.ndim != 1:
            raise ValueError("price_path must have shape (n,) or (M,n)")
        if t == 0:
            return float(path[:30].mean() / .9)
        q = T - t
        if path.size < q + 30:
            raise ValueError("path does not cover next-day 0:00--5:00")
        return float(path[q:q+30].mean() / .9)


class PerfectPriceSource:
    varies = True
    def __init__(self, dates, prices): self.dates=list(dates); self._price=np.asarray(prices,float); self._row={d:i for i,d in enumerate(self.dates)}
    def truth(self,d): return self._price[self._row[d]]
    def observed(self,d,t_now): return self.truth(d)[:int(t_now)]
    def predict(self,d,t_now=0,observed=None,**kw):
        nxt=d+timedelta(days=1); return self.truth(d), self._price[self._row[nxt]] if nxt in self._row else self.truth(d)
    def residual(self,d,t_now=0): return None

def horizon_price(src,d,t_now=0,n_days=2):
    today,nxt=src.predict(d,t_now); return today if n_days<=1 else np.concatenate([today]+[nxt]*(n_days-1))

__all__=["K_PRICE","LAMBDA_LO","LAMBDA_HI","PRICE_FLOOR","ConstantPriceSource","PerfectPriceSource","PriceForecaster","PriceSource","horizon_price"]
