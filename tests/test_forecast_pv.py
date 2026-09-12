import numpy as np
import pytest
from datetime import date, timedelta

from src.data import DT_H, T, load_attachment2, load_attachment3
from src.forecast_pv import (
    HORIZON_HOURS,
    ISSUE_HOURS,
    SEG_PER_HOUR,
    PVIssueForecaster,
    interpolate_issue,
    issue_segment,
)


@pytest.fixture(scope="module")
def data():
    return load_attachment2(), load_attachment3()


@pytest.fixture(scope="module")
def pv_fc(data):
    daily, att3 = data
    return PVIssueForecaster(daily, att3)


def test_hourly_values_land_on_the_hour_and_midpoints_are_linear():
    """第 k 小时的预报值必须出现在段末恰为该整点的第 6k−1 段上，段间线性。"""
    forecast = np.arange(1.0, HORIZON_HOURS + 1) * 100.0
    kwh = interpolate_issue(forecast, left_anchor_kw=0.0)
    assert kwh.shape == (T,)
    kw = kwh / DT_H
    for k in range(1, HORIZON_HOURS + 1):
        assert abs(kw[SEG_PER_HOUR * k - 1] - forecast[k - 1]) < 1e-9
    # 左端点 0 与第 1 小时 100 之间：第 j 段末在 (j+1)/6 小时处
    for j in range(SEG_PER_HOUR):
        assert abs(kw[j] - 100.0 * (j + 1) / SEG_PER_HOUR) < 1e-9


def test_interpolation_is_clipped_at_zero():
    forecast = np.linspace(-500.0, 500.0, HORIZON_HOURS)
    kwh = interpolate_issue(forecast, left_anchor_kw=-1000.0)
    assert (kwh >= 0.0).all()
    assert kwh.max() > 0.0


def test_issue_segment_maps_hours_to_segments():
    assert [issue_segment(h) for h in ISSUE_HOURS] == [0, 36, 72, 108]


def test_zero_issue_left_anchor_is_previous_day_last_segment(data, pv_fc):
    daily, att3 = data
    d = date(2025, 6, 21)
    prev = daily.pv_kwh[daily.dates.index(d - timedelta(days=1))]
    assert abs(pv_fc.anchor_kw(d, 0) - prev[T - 1] / DT_H) < 1e-9
    # 其余发布时刻取当天第 6h0−1 段
    today = daily.pv_kwh[daily.dates.index(d)]
    for h in (6, 12, 18):
        assert abs(pv_fc.anchor_kw(d, h) - today[issue_segment(h) - 1] / DT_H) < 1e-9


def test_curves_use_truth_for_past_forecast_for_rest_and_cross_midnight(data, pv_fc):
    daily, att3 = data
    d = date(2025, 6, 21)
    truth = daily.pv_kwh[daily.dates.index(d)]
    kp_mean = np.full(T, 777.0)
    for h in ISSUE_HOURS:
        t0 = issue_segment(h)
        today, nxt = pv_fc.curves(d, h, kp_mean)
        assert today.shape == (T,) and nxt.shape == (T,)
        assert np.allclose(today[:t0], truth[:t0])  # 已过去的段用真值
        interp = pv_fc.interpolate(d, h)
        assert np.allclose(today[t0:], interp[: T - t0])
        assert np.allclose(nxt[:t0], interp[T - t0 :])  # 跨日部分落到次日数组
        assert np.allclose(nxt[t0:], kp_mean[t0:])  # 预报未覆盖的段用 K_P 均值
        assert (today >= 0).all() and (nxt >= 0).all()


def test_zero_issue_next_day_is_all_kp_mean(pv_fc):
    kp_mean = np.arange(T, dtype=float)
    _, nxt = pv_fc.curves(date(2025, 3, 20), 0, kp_mean)
    assert np.allclose(nxt, kp_mean)  # 0:00 预报不覆盖次日


def test_residual_is_truth_minus_forecast_over_remaining_segments(data, pv_fc):
    daily, _ = data
    d = date(2025, 9, 23)
    truth = daily.pv_kwh[daily.dates.index(d)]
    for h in ISSUE_HOURS:
        t0 = issue_segment(h)
        r = pv_fc.residual(d, h)
        assert r.shape == (T - t0,)
        assert np.allclose(r, truth[t0:] - pv_fc.interpolate(d, h)[: T - t0])


def test_decision_at_h0_uses_only_data_available_at_h0(data):
    """把 $t\\ge 6h_0$ 的真值与该时刻之后发布的预报毒化成 NaN，曲线必须不变。"""
    daily, att3 = data
    d = date(2025, 6, 21)
    clean = PVIssueForecaster(daily, att3)
    kp_mean = np.full(T, 500.0)
    for h in (6, 12, 18):
        t0 = issue_segment(h)
        poisoned_pv = daily.pv_kwh.copy()
        i = daily.dates.index(d)
        poisoned_pv[i, t0:] = np.nan
        poisoned_pv[i + 1 :, :] = np.nan  # 次日及以后的真值一律不可见
        poisoned_att3 = dict(att3)
        for later in ISSUE_HOURS:
            if later > h:
                poisoned_att3[(d, later)] = np.full(HORIZON_HOURS, np.nan)
        for future_day in daily.dates[i + 1 :]:
            for hh in ISSUE_HOURS:
                poisoned_att3[(future_day, hh)] = np.full(HORIZON_HOURS, np.nan)
        guarded = PVIssueForecaster(
            type(daily)(dates=daily.dates, load_kwh=daily.load_kwh, pv_kwh=poisoned_pv),
            poisoned_att3,
        )
        today_g, next_g = guarded.curves(d, h, kp_mean)
        today_c, next_c = clean.curves(d, h, kp_mean)
        assert np.isfinite(today_g).all() and np.isfinite(next_g).all()
        assert np.allclose(today_g, today_c) and np.allclose(next_g, next_c)


def test_zero_issue_combines_formal_and_three_day_history(pv_fc):
    d = date(2025, 2, 1); hist = np.full(T, 17.)
    today, _ = pv_fc.combined_curves(d, 0, hist)
    assert np.allclose(today, .388815 * pv_fc.interpolate(d, 0) + .611185 * hist)


def test_january_weight_has_no_sample_and_degenerate_fallback(pv_fc):
    assert pv_fc.weight_at_zero(date(2025,1,2), np.zeros(T)) == 0.
