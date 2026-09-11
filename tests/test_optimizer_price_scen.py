import numpy as np
import pytest

from src.data import EPS, SOC_INIT, T, load_attachment1
from src.optimizer import (
    ADJUST_DOWN,
    ADJUST_UP,
    StorageRollingSolver,
    _saa_structure,
    first_stage_price,
    saa_cost_vector,
    solve_readjust,
    solve_saa,
    solve_saa_point,
)


@pytest.fixture(scope="module")
def att():
    return load_attachment1()


@pytest.fixture(scope="module")
def curves(att):
    """M=3 的小算例：负载场景分散、光伏相同、次日曲线 = 当天点预测。"""
    L, PV = att.load_kwh, att.pv_kwh
    price48 = np.tile(att.price, 2)
    L_scen = np.stack([L * 0.9, L, L * 1.1])
    PV_scen = np.stack([PV, PV, PV])
    return price48, L, PV, L_scen, PV_scen


def test_cost_vector_without_price_scen_is_byte_identical(curves):
    """`price_scen=None` 时的代价向量与「单一电价」旧式装配逐位相同。"""
    price48, *_ = curves
    m, n_today, n_future = 12, T, T
    struct = _saa_structure(m, n_today, n_future)

    old = np.zeros(struct.n_var)
    old[struct.g0_off : struct.g0_off + n_today] = price48[:n_today]
    for block in struct.blocks:
        old[block["E"] : block["E"] + struct.n_h] = 5.0 * price48 / m
        old[block["C"] : block["C"] + struct.n_h] = EPS
        old[block["D"] : block["D"] + struct.n_h] = EPS
        old[block["W"] : block["W"] + struct.n_h] = EPS
        old[block["G"] : block["G"] + n_future] = price48[n_today:] / m

    new = saa_cost_vector(struct, price48, None, EPS)
    assert np.array_equal(new, old)


def test_solution_is_unchanged_when_price_scen_is_omitted(curves):
    """同一组曲线下，显式传 None 与不传参数给出逐位相同的 G0。"""
    price48, L, PV, L_scen, PV_scen = curves
    a = solve_saa(price48, L_scen, PV_scen, L, PV, SOC_INIT)
    b = solve_saa(price48, L_scen, PV_scen, L, PV, SOC_INIT, price_scen=None)
    assert np.array_equal(a.G0, b.G0)
    assert a.objective == b.objective


def test_identical_price_scenarios_match_the_deterministic_price(curves):
    """所有场景电价相同时，结果与确定电价路径一致（数值上只差舍入）。"""
    price48, L, PV, L_scen, PV_scen = curves
    ps = np.tile(price48, (L_scen.shape[0], 1))
    det = solve_saa(price48, L_scen, PV_scen, L, PV, SOC_INIT)
    scen = solve_saa(price48, L_scen, PV_scen, L, PV, SOC_INIT, price_scen=ps)
    assert np.abs(det.G0 - scen.G0).max() < 1e-6
    assert abs(det.objective - scen.objective) < 1e-6


def test_first_stage_is_priced_at_the_scenario_mean(curves):
    """两个场景电价不同时，$G^0$ 的目标系数 = 场景均价；E 与次日 G 按各自场景计价。"""
    price48, *_ = curves
    m = 2
    struct = _saa_structure(m, T, T)
    ps = np.stack([price48 * 0.5, price48 * 1.5])
    cost = saa_cost_vector(struct, price48, ps, EPS)

    mean_price = ps[:, :T].mean(axis=0)
    assert np.allclose(cost[struct.g0_off : struct.g0_off + T], mean_price)
    assert np.allclose(mean_price, price48[:T])  # 0.5 与 1.5 的均值就是原电价
    for w, block in enumerate(struct.blocks):
        assert np.allclose(cost[block["E"] : block["E"] + struct.n_h], 5.0 * ps[w] / m)
        assert np.allclose(cost[block["G"] : block["G"] + T], ps[w, T:] / m)
    assert np.allclose(first_stage_price(price48, ps, T), mean_price)
    assert np.array_equal(first_stage_price(price48, None, T), price48[:T])


def test_expensive_scenarios_shift_the_plan_to_cheap_segments(curves):
    """把某一段的场景电价整体抬高，第一阶段就把购电挪走（均价确实进了目标）。"""
    price48, L, PV, L_scen, PV_scen = curves
    m = L_scen.shape[0]
    base = np.tile(price48, (m, 1))
    spike = base.copy()
    spike[:, 30] *= 6.0
    flat = solve_saa(price48, L_scen, PV_scen, L, PV, SOC_INIT, price_scen=base)
    moved = solve_saa(price48, L_scen, PV_scen, L, PV, SOC_INIT, price_scen=spike)
    assert moved.G0[30] < flat.G0[30] - 1e-6


def test_price_scen_shape_is_checked(curves):
    price48, L, PV, L_scen, PV_scen = curves
    with pytest.raises(AssertionError):
        solve_saa(
            price48,
            L_scen,
            PV_scen,
            L,
            PV,
            SOC_INIT,
            price_scen=np.tile(price48, (L_scen.shape[0] + 1, 1)),
        )


def test_readjust_matches_the_deterministic_path_with_identical_scenarios(att):
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    point = solve_saa_point(price48, L, PV, L, PV, SOC_INIT)
    G0, t0 = point.G[:T], 36
    scen_L = np.stack([L[t0:] * 1.2, L[t0:] * 0.9])
    scen_PV = np.stack([PV[t0:] * 0.5, PV[t0:] * 1.3])
    args = (t0, price48, G0, scen_L, scen_PV, L, PV, float(point.S[t0 - 1]))
    det = solve_readjust(*args)
    scen = solve_readjust(*args, price_scen=np.tile(price48, (2, 1)))
    assert np.abs(det.Ga - scen.Ga).max() < 1e-6
    assert abs(det.adjust_cost - scen.adjust_cost) < 1e-6


def test_readjust_prices_the_adjustment_at_the_scenario_mean(att):
    """两场景电价不同时，$\\Delta^\\pm$ 按均价、紧急购电按各场景电价。"""
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    point = solve_saa_point(price48, L, PV, L, PV, SOC_INIT)
    G0, t0 = point.G[:T], 36
    n_rem = T - t0
    scen_L = np.stack([L[t0:], L[t0:]])
    scen_PV = np.stack([PV[t0:] * 0.5, PV[t0:] * 0.5])
    ps = np.stack([price48 * 0.5, price48 * 1.5])
    mean_day = ps[:, :T].mean(axis=0)

    cheap = solve_readjust(
        t0, price48, G0, scen_L, scen_PV, L, PV, float(point.S[t0 - 1]),
        price_scen=np.tile(price48 * 0.5, (2, 1)),
    )
    mixed = solve_readjust(
        t0, price48, G0, scen_L, scen_PV, L, PV, float(point.S[t0 - 1]), price_scen=ps
    )
    # 均价 = 原电价，故 mixed 的调整费系数与确定电价一致、是 cheap 的 2 倍口径
    assert mixed.dplus.sum() > 0.0
    expected = float(
        ADJUST_UP * mean_day @ mixed.dplus - ADJUST_DOWN * mean_day @ mixed.dminus
    )
    assert abs(mixed.adjust_cost - expected) < 1e-6
    assert cheap.adjust_cost < mixed.adjust_cost
    assert mixed.dplus[:t0].max() == 0.0 and mixed.dplus.size == T
    assert n_rem == mixed.dplus[t0:].size


def test_rolling_solver_price_argument_only_swaps_the_cost_vector(att):
    """逐段传入的电价与「用该电价构造求解器」等价；不传时沿用构造电价。"""
    price48 = np.tile(att.price, 2)
    other = price48 * np.linspace(0.6, 1.4, price48.size)
    L, PV = att.load_kwh, att.pv_kwh
    g = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT).G0
    t, soc = 48, 5200.0

    base = StorageRollingSolver(price48)
    dedicated = StorageRollingSolver(other)
    a = base.solve(t, soc, L[t] * 1.3, PV[t] * 0.4, L, PV, L, PV, g, other)
    b = dedicated.solve(t, soc, L[t] * 1.3, PV[t] * 0.4, L, PV, L, PV, g)
    for name in ("C", "D", "W", "E", "S", "G"):
        assert np.allclose(getattr(a, name), getattr(b, name), atol=1e-6)
    assert abs(a.objective - b.objective) < 1e-6

    kept = base.solve(t, soc, L[t] * 1.3, PV[t] * 0.4, L, PV, L, PV, g)
    same = base.solve(t, soc, L[t] * 1.3, PV[t] * 0.4, L, PV, L, PV, g, None)
    assert np.array_equal(kept.E, same.E)
    assert np.array_equal(base.cost, StorageRollingSolver(price48).cost)  # 未被污染


def test_rolling_solver_cost_for_matches_the_construction_time_layout(att):
    price48 = np.tile(att.price, 2)
    solver = StorageRollingSolver(price48)
    off = solver.off
    cost = solver.cost_for(price48)
    assert np.array_equal(cost, solver.cost)
    other = price48 * 2.0
    cost = solver.cost_for(other)
    assert np.array_equal(cost[off["G"] : off["G"] + T], np.zeros(T))  # 当天购电是常数
    assert np.array_equal(cost[off["G"] + T : off["G"] + 2 * T], other[T:])
    assert np.array_equal(cost[off["E"] : off["E"] + 2 * T], 5.0 * other)
