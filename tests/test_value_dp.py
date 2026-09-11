import numpy as np

from src.data import SOC_INIT, SOC_MAX, SOC_MIN
from src.value_dp import (
    ConvexPiecewiseLinear,
    average_functions,
    evaluate_plan,
    future_cost,
    next_day_value,
    reserve_level,
)


def _reference_next_day_value(price, load, pv, *, terminal=None, grid):
    """独立的逐状态/逐动作参考实现，用于检验向量化递推。"""
    from src.data import ETA, P_MAX_KWH, SOC_MAX, SOC_MIN

    price = np.asarray(price, float).ravel()
    load = np.asarray(load, float).ravel()
    pv = np.asarray(pv, float).ravel()
    g = np.asarray(grid, float)
    v = np.zeros(g.size) if terminal is None else np.asarray(terminal(g), float)
    for t in range(price.size - 1, -1, -1):
        new = np.empty_like(v)
        net = load[t] - pv[t]
        for i, e in enumerate(g):
            lo_x = max(-P_MAX_KWH / ETA, SOC_MIN - e)
            hi_x = min(ETA * P_MAX_KWH, SOC_MAX - e)
            best = np.inf
            for x in np.linspace(lo_x, hi_x, 33):
                ee = e + x
                q = x / ETA if x >= 0 else ETA * x
                best = min(best, price[t] * max(net + q, 0.0) + np.interp(ee, g, v))
            new[i] = best
        v = new
    return v


def _reference_future_cost(price, load_scen, pv_scen, g_plan, terminal_value, *, grid):
    """独立的逐场景/逐状态参考实现，用于检验向量化递推。"""
    from src.data import ETA, P_MAX_KWH, SOC_MAX, SOC_MIN

    p = np.asarray(price, float)
    load = np.atleast_2d(np.asarray(load_scen, float))
    pv = np.atleast_2d(np.asarray(pv_scen, float))
    g_plan = np.asarray(g_plan, float).ravel()
    g = np.asarray(grid, float)
    result = []
    for w in range(load.shape[0]):
        v = np.asarray(terminal_value(g), float)
        path = [None] * (load.shape[1] + 1)
        path[-1] = ConvexPiecewiseLinear(g, v)
        for t in range(load.shape[1] - 1, -1, -1):
            r = load[w, t] - pv[w, t] - g_plan[t]
            vals = []
            for e in g:
                if r <= 0:
                    c = min(-r, P_MAX_KWH, (SOC_MAX - e) / ETA)
                    vals.append(np.interp(e + ETA * c, g, v))
                    continue
                ylo = max(SOC_MIN, e - P_MAX_KWH / ETA)
                ys = g[(g >= ylo - 1e-9) & (g <= e + 1e-9)]
                if ys.size == 0:
                    ys = np.array([ylo])
                ys = np.unique(np.concatenate([ys, [np.clip(e - r / ETA, ylo, e)]]))
                d = ETA * (e - ys)
                vals.append(np.min(5 * p[w, t] * np.maximum(r - d, 0.0) + np.interp(ys, g, v)))
            v = np.asarray(vals)
            path[t] = ConvexPiecewiseLinear(g, v)
        result.append(path)
    return [
        average_functions([path[t] for path in result])
        for t in range(load.shape[1] + 1)
    ]


def test_next_day_value_accepts_an_empty_future_horizon():
    value = next_day_value(np.zeros(0), np.zeros(0), np.zeros(0))
    assert value(SOC_INIT) == 0.0


def test_next_day_value_matches_independent_reference_on_small_grid():
    grid = np.array([SOC_MIN, 2300.0, 4100.0, 6700.0, SOC_MAX])
    price = np.array([0.4, 1.1, 0.7])
    load = np.array([900.0, 120.0, 700.0])
    pv = np.array([100.0, 500.0, 0.0])
    terminal = lambda z: 0.03 * (np.asarray(z) - 5000.0) ** 2
    got = next_day_value(price, load, pv, terminal=terminal, grid=grid)
    want = _reference_next_day_value(price, load, pv, terminal=terminal, grid=grid)
    assert np.allclose(got.values, want, rtol=0.0, atol=1e-12)


def test_future_cost_matches_independent_reference_for_surplus_and_deficit():
    grid = np.array([SOC_MIN, 2300.0, 4100.0, 6700.0, SOC_MAX])
    price = np.array([[0.4, 1.1, 0.7], [0.8, 0.6, 1.3]])
    load = np.array([[500.0, 1800.0, 900.0], [1600.0, 300.0, 2200.0]])
    pv = np.array([[700.0, 100.0, 0.0], [0.0, 800.0, 100.0]])
    g_plan = np.array([100.0, 300.0, 500.0])
    terminal = lambda z: 0.02 * (np.asarray(z) - 6000.0) ** 2
    got = future_cost(price, load, pv, g_plan, terminal, grid=grid)
    want = _reference_future_cost(price, load, pv, g_plan, terminal, grid=grid)
    assert len(got) == len(want) == 4
    for got_fn, want_fn in zip(got, want):
        assert np.allclose(got_fn.breakpoints, want_fn.breakpoints, rtol=0.0, atol=0.0)
        assert np.allclose(got_fn.values, want_fn.values, rtol=0.0, atol=1e-12)


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
