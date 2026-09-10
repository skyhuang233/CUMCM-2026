from datetime import date

import numpy as np

from src.data import T, day_type, load_attachment1, load_attachment2, load_attachment3, load_attachment4


def test_attachment1_shape_and_daily_totals():
    att = load_attachment1()
    assert att.price.shape == (T,)
    assert att.load_kwh.shape == (T,)
    assert att.pv_kwh.shape == (T,)
    assert abs(att.load_kwh.sum() - 111024.8) < 1.0
    assert abs(att.pv_kwh.sum() - 55482.8) < 1.0
    assert att.price.min() > 0


def test_attachment2_full_year():
    series = load_attachment2()
    assert series.load_kwh.shape == (365, T)
    assert series.pv_kwh.shape == (365, T)
    assert not np.isnan(series.load_kwh).any()
    assert not np.isnan(series.pv_kwh).any()
    assert series.dates[0] == date(2025, 1, 1)
    assert series.dates[-1] == date(2025, 12, 31)


def test_attachment4_prices():
    dates, price = load_attachment4()
    assert price.shape == (365, T)
    assert len(dates) == 365
    assert price.min() > 0


def test_attachment3_forecasts():
    fc = load_attachment3()
    assert len(fc) == 365 * 4
    assert sorted({h for _, h in fc}) == [0, 6, 12, 18]
    assert fc[(date(2025, 1, 1), 0)].shape == (24,)
    assert all(v.shape == (24,) for v in fc.values())


def test_day_type():
    assert day_type(date(2025, 3, 21)) == 1  # 周五
    assert day_type(date(2025, 3, 22)) == 1  # 周六
    assert day_type(date(2025, 3, 20)) == 0  # 周四
