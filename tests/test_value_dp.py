import numpy as np

from src.data import SOC_INIT, SOC_MAX, SOC_MIN
from src.value_dp import (
    ConvexPiecewiseLinear,
    evaluate_plan,
    future_cost,
    next_day_value,
    reserve_level,
)


def test_next_day_value_accepts_an_empty_future_horizon():
    value = next_day_value(np.zeros(0), np.zeros(0), np.zeros(0))
    assert value(SOC_INIT) == 0.0


def test_future_cost_and_reserve_are_bounded_on_a_one_step_deficit():
    terminal = ConvexPiecewiseLinear.constant(0.0)
    hbar = future_cost(np.array([1.0]), [[100.0]], [[0.0]], [0.0], terminal)
    assert len(hbar) == 2
    reserve = reserve_level(hbar[1], 1.0)
    assert SOC_MIN <= reserve <= SOC_MAX
    # With zero terminal value, holding energy has no future benefit.
    assert reserve == SOC_MIN


def test_evaluate_plan_returns_reusable_future_cost_functions():
    score, hbar = evaluate_plan(
        np.array([0.0, 0.0]), [[10000.0, 10000.0]], [[0.0, 0.0]],
        np.array([1.0, 1.0]), SOC_INIT, return_hbar=True,
    )
    assert score > 0.0
    assert len(hbar) == 3
    assert all(np.all(np.isfinite(f.values)) for f in hbar)
