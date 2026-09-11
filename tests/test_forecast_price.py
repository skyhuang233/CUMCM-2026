import numpy as np
import pytest
from datetime import date, timedelta

from src.data import T, day_type, load_attachment1, load_attachment4
from src.forecast_price import (
    LAMBDA_HI,
    LAMBDA_LO,
    ConstantPriceSource,
    PerfectPriceSource,
    PriceForecaster,
    horizon_price,
)


@pytest.fixture(scope="module")
def att():
    return load_attachment1()


@pytest.fixture(scope="module")
def att4():
    return load_attachment4()


@pytest.fixture(scope="module")
def fc(att, att4):
    dates, prices = att4
    return PriceForecaster(dates, prices, att, k=4)


def test_baseline_is_the_mean_of_the_last_k_same_type_past_days(fc, att4):
    dates, prices = att4
    d = date(2025, 6, 21)  # 周六 → 日类型 1
    rows = [
        i for i, h in enumerate(dates) if h < d and day_type(h) == day_type(d)
    ][-4:]
    assert len(rows) == 4
    assert all(dates[i] < d for i in rows)
    assert np.allclose(fc.baseline(d), prices[rows].mean(axis=0))


def test_cold_start_baseline_falls_back_to_attachment1(fc, att):
    """全年首日之前没有同类型历史，基准退回附件 1 的常数电价曲线。"""
    assert np.allclose(fc.baseline(date(2025, 1, 1)), att.price)


def test_baseline_never_reads_rows_at_or_after_the_decision_day(att, att4):
    """把日期 ≥ 决策日的电价行毒化成 NaN：基准与预测仍有限且与干净数据一致。"""
    dates, prices = att4
    d = date(2025, 5, 10)
    poisoned = prices.copy()
    poisoned[[i for i, h in enumerate(dates) if h >= d]] = np.nan
    clean = PriceForecaster(dates, prices, att, k=4)
    guarded = PriceForecaster(dates, poisoned, att, k=4)
    for t_now in (0, 36, 72):
        obs = clean.observed(d, t_now)  # 当天真值由调用方给出（执行层已发生的段）
        a = clean.predict(d, t_now, obs)
        b = guarded.predict(d, t_now, obs)
        assert np.isfinite(b[0]).all() and np.isfinite(b[1]).all()
        assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_predict_never_reads_segments_at_or_after_the_decision_segment(att, att4):
    """把当天 $\\tau\\ge t_0$ 的电价毒化成 NaN：$t_0$ 时刻的预测必须不受影响。"""
    dates, prices = att4
    d, t0 = date(2025, 5, 10), 72
    i = dates.index(d)
    poisoned = prices.copy()
    poisoned[i, t0:] = np.nan
    poisoned[i + 1 :] = np.nan
    clean = PriceForecaster(dates, prices, att, k=4)
    guarded = PriceForecaster(dates, poisoned, att, k=4)
    a_today, a_next = clean.predict(d, t0)
    b_today, b_next = guarded.predict(d, t0)
    assert np.isfinite(b_today[t0:]).all() and np.isfinite(b_next).all()
    assert np.allclose(a_today[t0:], b_today[t0:])
    assert np.allclose(a_next, b_next)
    assert guarded.observed(d, t0).size == t0
    assert np.isfinite(guarded.observed(d, t0)).all()


def test_lambda_is_the_least_squares_ratio(fc):
    base = np.linspace(0.4, 1.2, 24)
    observed = 1.17 * base + np.array([0.01, -0.02] * 12)
    expected = float(observed @ base / (base @ base))
    assert abs(fc.level_factor(base, observed) - expected) < 1e-12
    assert LAMBDA_LO < expected < LAMBDA_HI


def test_lambda_is_one_without_observations(fc):
    base = np.full(T, 0.8)
    assert fc.level_factor(base, np.zeros(0)) == 1.0
    assert fc.level_factor(np.zeros(T), np.full(10, 0.5)) == 1.0  # 基准全零时退回 1


def test_lambda_is_clipped(fc):
    base = np.full(36, 0.8)
    assert fc.level_factor(base, 5.0 * base) == LAMBDA_HI
    assert fc.level_factor(base, 0.1 * base) == LAMBDA_LO


def test_predict_pastes_observed_truth_and_scales_the_rest(fc, att4):
    dates, prices = att4
    d, t0 = date(2025, 7, 15), 36
    i = dates.index(d)
    today, nxt = fc.predict(d, t0)
    base = fc.baseline(d)
    lam = fc.level_factor(base, prices[i, :t0])
    assert np.allclose(today[:t0], prices[i, :t0])  # 已发生段用真值
    assert np.allclose(today[t0:], lam * base[t0:])
    assert np.allclose(nxt, lam * fc.baseline(d + timedelta(days=1), asof=d))
    assert today.shape == (T,) and nxt.shape == (T,)
    assert (today >= 0).all() and (nxt >= 0).all()


def test_next_day_uses_the_next_day_type_baseline(fc):
    d = date(2025, 6, 19)  # 周四（类型 0）；次日周五（类型 1）
    _, nxt = fc.predict(d, 0)
    assert np.allclose(nxt, fc.baseline(d + timedelta(days=1), asof=d))
    assert not np.allclose(nxt, fc.baseline(d))  # 两个日类型的基准确实不同


def test_residual_is_truth_minus_the_point_forecast_of_that_moment(fc, att4):
    dates, prices = att4
    d, t0 = date(2025, 9, 23), 108
    i = dates.index(d)
    r = fc.residual(d, t0)
    today, _ = fc.predict(d, t0)
    assert r.shape == (T - t0,)
    assert np.allclose(r, prices[i, t0:] - today[t0:])
    assert np.allclose(fc.residual(d, 0), prices[i] - fc.predict(d, 0)[0])


def test_forecast_beats_the_all_history_same_type_mean(att, att4):
    """$K_p=4$ 的近期同类型均值应显著优于全历史同类型均值（题面已核实的事实）。"""
    dates, prices = att4
    fc4 = PriceForecaster(dates, prices, att, k=4)
    fc_all = PriceForecaster(dates, prices, att, k=400)
    days = [d for d in dates if d >= date(2025, 3, 1)]
    err4 = np.mean([np.abs(fc4.residual(d, 0)).mean() for d in days])
    err_all = np.mean([np.abs(fc_all.residual(d, 0)).mean() for d in days])
    assert err4 < err_all
    assert err4 < 0.05  # 元/kWh


def test_constant_price_source_is_flat_and_residual_free(att):
    src = ConstantPriceSource(att.price)
    assert src.varies is False
    today, nxt = src.predict(date(2025, 3, 20), 0)
    assert np.allclose(today, att.price) and np.allclose(nxt, att.price)
    assert src.residual(date(2025, 3, 20), 0) is None
    assert np.allclose(src.truth(date(2025, 8, 1)), att.price)


def test_perfect_price_source_sees_today_and_tomorrow(att4):
    dates, prices = att4
    src = PerfectPriceSource(dates, prices)
    d = date(2025, 4, 4)
    today, nxt = src.predict(d, 0)
    assert np.allclose(today, prices[dates.index(d)])
    assert np.allclose(nxt, prices[dates.index(d) + 1])
    assert src.residual(d, 0) is None
    # 全年末日没有次日，退回当天曲线而不是越界
    last_today, last_next = src.predict(dates[-1], 0)
    assert np.allclose(last_today, last_next)


def test_horizon_price_concatenates_today_and_repeated_next(fc, att):
    d = date(2025, 3, 20)
    today, nxt = fc.predict(d, 0)
    assert np.allclose(horizon_price(fc, d, 0, 1), today)
    p48 = horizon_price(fc, d, 0, 2)
    assert p48.shape == (2 * T,)
    assert np.allclose(p48[:T], today) and np.allclose(p48[T:], nxt)
    assert horizon_price(fc, d, 0, 3).shape == (3 * T,)
    # 常数电价来源在 48h 时域上就是附件 1 的两次平铺
    assert np.array_equal(
        horizon_price(ConstantPriceSource(att.price), d, 0, 2), np.tile(att.price, 2)
    )
