import numpy as np
import pytest
from datetime import date, timedelta

from src.data import DailySeries, T, day_type, load_attachment1, load_attachment2
from src.forecast import PointForecaster


@pytest.fixture(scope="module")
def bundle():
    return load_attachment1(), load_attachment2()


def test_cold_start_uses_attachment1_prior(bundle):
    att1, daily = bundle
    fc = PointForecaster(daily, att1)
    load, pv = fc.predict(date(2025, 1, 1))
    assert np.allclose(load, att1.load_kwh)
    assert np.allclose(pv, att1.pv_kwh)


def test_predict_uses_only_strictly_past_rows(bundle):
    """把日期 ≥ 决策日的行毒化成 NaN：预测仍然有限，即证明没有读到未来行。"""
    att1, daily = bundle
    d = date(2025, 2, 1)
    poisoned_load = daily.load_kwh.copy()
    poisoned_pv = daily.pv_kwh.copy()
    future = np.array([x >= d for x in daily.dates])
    poisoned_load[future] = np.nan
    poisoned_pv[future] = np.nan
    poisoned = DailySeries(dates=daily.dates, load_kwh=poisoned_load, pv_kwh=poisoned_pv)

    clean = PointForecaster(daily, att1)
    guarded = PointForecaster(poisoned, att1)
    for pred in (guarded.predict(d), guarded.predict_next(d)):
        assert np.isfinite(pred[0]).all() and np.isfinite(pred[1]).all()
    assert np.allclose(guarded.predict(d)[0], clean.predict(d)[0])
    assert np.allclose(guarded.predict(d)[1], clean.predict(d)[1])
    assert np.allclose(guarded.predict_next(d)[0], clean.predict_next(d)[0])


def test_window_rows_are_recorded_and_all_in_the_past(bundle, monkeypatch):
    att1, daily = bundle
    fc = PointForecaster(daily, att1)
    d = date(2025, 6, 21)
    seen: list[tuple[date, ...]] = []
    original = PointForecaster._rows

    def spy(self, asof, k, dtype=None):
        rows = original(self, asof, k, dtype)
        seen.append(tuple(self.dates[i] for i in rows))
        return rows

    monkeypatch.setattr(PointForecaster, "_rows", spy)
    fc.predict(d)
    fc.predict_next(d)
    assert seen, "预测过程没有经过 _rows，无法证明 walk-forward"
    for group in seen:
        assert all(h < d for h in group)


def test_load_window_is_same_day_type_and_length_k(bundle):
    att1, daily = bundle
    fc = PointForecaster(daily, att1, k_load=4, k_pv=5)
    d = date(2025, 6, 21)  # 周六 → 日类型 1
    rows_l = fc._rows(d, fc.k_load, day_type(d))
    rows_p = fc._rows(d, fc.k_pv)
    assert len(rows_l) == 4
    assert all(day_type(fc.dates[i]) == day_type(d) for i in rows_l)
    assert len(rows_p) == 5
    load, _ = fc.predict(d)
    assert np.allclose(load, daily.load_kwh[rows_l].mean(axis=0))
    assert load.shape == (T,)


def test_predict_next_uses_next_day_type_but_today_cutoff(bundle):
    att1, daily = bundle
    fc = PointForecaster(daily, att1)
    d = date(2025, 6, 19)  # 周四；次日 6-20 为周五 → 日类型 1
    load_next, _ = fc.predict_next(d)
    rows = fc._rows(d, fc.k_load, day_type(d + timedelta(days=1)))
    assert np.allclose(load_next, daily.load_kwh[rows].mean(axis=0))
    assert all(fc.dates[i] < d for i in rows)


def test_predict_future_concatenates_days(bundle):
    att1, daily = bundle
    fc = PointForecaster(daily, att1)
    d = date(2025, 5, 5)
    load, pv = fc.predict_future(d, 2)
    assert load.shape == (2 * T,) and pv.shape == (2 * T,)
    assert np.allclose(load[:T], fc.predict(d + timedelta(days=1), asof=d)[0])
    assert fc.predict_future(d, 0)[0].size == 0
