"""Causal load-level/shape and three-day PV forecasts.

``reference_baseline=True`` is the approved paper baseline.  The former
day-type mean survives only as the explicit ``False`` incremental setting.
"""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
from .data import Attachment1, DailySeries, day_type

LOW_START = date(2025, 1, 15)

def load_kind(d: date) -> int:
    """Paper k: Fri/Sat are low (0), all other days high (1)."""
    return 0 if d.weekday() in (4, 5) else 1

class PointForecaster:
    def __init__(self, daily: DailySeries, prior: Attachment1, k_load: int = 3,
                 k_pv: int = 3, reference_baseline: bool = True):
        self.dates = list(daily.dates); self._load = np.asarray(daily.load_kwh, float)
        self._pv = np.asarray(daily.pv_kwh, float); self.prior = prior
        self._ord = np.array([x.toordinal() for x in self.dates]); self._row = {x:i for i,x in enumerate(self.dates)}
        self._types = np.array([day_type(x) for x in self.dates]); self.k_load, self.k_pv = int(k_load), int(k_pv)
        self.reference_baseline = bool(reference_baseline)

    def _rows(self, asof: date, k: int, dtype: int | None = None) -> np.ndarray:
        ok = self._ord < asof.toordinal()
        if dtype is not None: ok &= self._types == dtype
        rows = np.flatnonzero(ok)[-int(k):]
        assert not rows.size or self._ord[rows].max() < asof.toordinal()
        return rows

    def _warm_load(self, d: date, cutoff: date) -> np.ndarray:
        hist = np.flatnonzero(self._ord < cutoff.toordinal())
        if not hist.size:
            return self.prior.load_kwh.copy()
        rows = hist[np.array([self.dates[i].weekday() == d.weekday() for i in hist])][-3:]
        if rows.size < 2: rows = hist[-7:]
        return self._load[rows].mean(0) if rows.size else self.prior.load_kwh.copy()

    def _paper_load(self, d: date, cutoff: date) -> np.ndarray:
        if d < LOW_START: return np.asarray(self._warm_load(d, cutoff), float)
        hist = np.flatnonzero(self._ord < cutoff.toordinal())
        rows = hist[np.array([load_kind(self.dates[i]) == load_kind(d) for i in hist])][-3:]
        if not rows.size: return np.asarray(self._warm_load(d, cutoff), float)
        totals = self._load[rows].sum(1); shape = self._load[rows].sum(0) / max(float(totals.sum()), 1e-12)
        beta = []
        for i in hist:
            if self._ord[i] < cutoff.toordinal() - 35: continue
            prev = self._row.get(self.dates[i] - timedelta(days=1)); den = load_kind(self.dates[i]) - load_kind(self.dates[i] - timedelta(days=1))
            if prev is not None and den and self._load[i].sum() > 0 and self._load[prev].sum() > 0:
                beta.append(np.log(self._load[i].sum()/self._load[prev].sum()) / den)
        previous_day = d - timedelta(days=1)
        # For future requests d's level is itself a causal prediction.
        if previous_day in self._row and previous_day < cutoff:
            previous = float(self._load[self._row[previous_day]].sum())
        else:
            previous = float(self._paper_load(previous_day, cutoff).sum()) if previous_day >= LOW_START else float(self._warm_load(previous_day, cutoff).sum())
        b = float(np.median(beta)) if beta else 0.0
        return np.clip(previous*np.exp(b*(load_kind(d)-load_kind(previous_day)))*shape, 0, None)

    def predict(self, d: date, asof: date | None = None) -> tuple[np.ndarray, np.ndarray]:
        cutoff = d if asof is None else asof
        if not self.reference_baseline:
            rl, rp = self._rows(cutoff,self.k_load,day_type(d)), self._rows(cutoff,self.k_pv)
            return (self._load[rl].mean(0) if rl.size else self.prior.load_kwh.copy(), self._pv[rp].mean(0) if rp.size else self.prior.pv_kwh.copy())
        rows = self._rows(cutoff, 3)
        return self._paper_load(d, cutoff), (self._pv[rows].mean(0) if rows.size else self.prior.pv_kwh.copy())

    def predict_next(self, d: date) -> tuple[np.ndarray, np.ndarray]: return self.predict(d + timedelta(days=1), asof=d)
    def predict_future(self, d: date, n_days: int) -> tuple[np.ndarray, np.ndarray]:
        if n_days <= 0: return np.zeros(0), np.zeros(0)
        x = [self.predict(d+timedelta(days=k), asof=d) for k in range(1,n_days+1)]
        return np.concatenate([z[0] for z in x]), np.concatenate([z[1] for z in x])

__all__ = ["LOW_START", "PointForecaster", "load_kind"]
