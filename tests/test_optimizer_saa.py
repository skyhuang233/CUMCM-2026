import numpy as np
import pytest

from src.data import EPS, ETA, P_MAX_KWH, SOC_INIT, SOC_MAX, SOC_MIN, T, load_attachment1
from src.optimizer import (
    StorageRollingSolver,
    solve_deterministic,
    solve_saa,
    solve_saa_point,
)


@pytest.fixture(scope="module")
def att():
    return load_attachment1()


def test_saa_with_zero_residual_matches_deterministic_48h(att):
    """M=1、残差为零、次日曲线与当天相同时，SAA 退化为 48h 确定性 LP。"""
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    saa = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT)
    point = solve_saa_point(price48, L, PV, L, PV, SOC_INIT)
    det = solve_deterministic(
        price48, np.tile(L, 2), np.tile(PV, 2), SOC_INIT, periodic=False
    )
    assert np.abs(saa.G0 - det.G[:T]).max() < 1e-4
    assert point.E.max() < 1e-6  # 紧急购电永不划算
    assert np.minimum(point.C, point.D).max() < 1e-6


def test_explicit_point_solution_satisfies_balance_and_soc(att):
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    saa = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT)
    pt = solve_saa_point(price48, L, PV, L, PV, SOC_INIT)
    load48, pv48 = np.tile(L, 2), np.tile(PV, 2)
    balance = pt.G + pt.E + pv48 + pt.D - load48 - pt.C - pt.W
    assert np.abs(balance).max() < 1e-6
    prev = np.concatenate([[SOC_INIT], pt.S[:-1]])
    assert np.abs(pt.S - prev - ETA * pt.C + pt.D / ETA).max() < 1e-6
    assert pt.S.min() >= SOC_MIN - 1e-6 and pt.S.max() <= SOC_MAX + 1e-6
    assert np.allclose(pt.G[:T], saa.G0, atol=1e-6)


def test_saa_first_stage_is_shared_across_scenarios(att):
    """场景越分散，第一阶段购电量越保守（对 5 倍紧急电价的对冲）。"""
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    tight = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT)
    spread = solve_saa(
        price48,
        np.stack([L * 0.9, L, L * 1.1]),
        np.stack([PV, PV, PV]),
        L,
        PV,
        SOC_INIT,
    )
    assert spread.G0.shape == (T,)
    assert spread.G0.sum() > tight.G0.sum()
    assert spread.G0.min() >= -1e-9


def test_saa_is_invariant_when_identical_scenarios_are_repeated(att):
    """重复相同场景不应改变 sample-average 目标中的正则权重。"""
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    one = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT)
    repeated = solve_saa(
        price48,
        np.repeat(L[None, :], 3, axis=0),
        np.repeat(PV[None, :], 3, axis=0),
        L,
        PV,
        SOC_INIT,
    )
    assert np.allclose(repeated.G0, one.G0, rtol=0.0, atol=1e-6)
    assert repeated.objective == pytest.approx(one.objective, abs=1e-6)


def test_rolling_lp_buys_emergency_in_cheap_segment_and_saves_storage():
    """小 T 例子：两个缺口段电价 0.4 与 1.0，储能只够补一段。

    跨期最优 = 在 1.0 段放电、在 0.4 段紧急购电；贪心（先到先放）会在 0.4 段放电、
    在 1.0 段紧急购电，费用高 2.5 倍。
    """
    n = 12
    gap = 500.0
    price = np.full(n, 5.0)
    price[4] = 0.4
    price[9] = 1.0
    solver = StorageRollingSolver(price, n_today=n)

    load = np.zeros(n)
    load[4] = gap
    load[9] = gap
    pv = np.zeros(n)
    g_today = np.zeros(n)
    soc_init = SOC_MIN + gap / ETA  # 恰好只够补一个缺口

    step = solver.solve(
        0, soc_init, load[0], pv[0], load, pv, np.zeros(0), np.zeros(0), g_today
    )
    assert step.D[9] > gap - 1e-6  # 高价段靠放电
    assert step.E[9] < 1e-6
    assert step.E[4] > gap - 1e-6  # 低价段紧急购电
    assert step.D[4] < 1e-6
    assert max(step.C.max(), step.D.max()) <= P_MAX_KWH + 1e-6

    smart = 5.0 * float(price @ step.E)
    greedy = 5.0 * price[9] * gap  # 先在 0.4 段放电、1.0 段被迫紧急购电
    assert smart < greedy - 1e-6
    assert abs(smart - 5.0 * price[4] * gap) < 1e-3


def test_rolling_solver_freezes_past_segments_and_starts_from_current_soc(att):
    """段 t 之前的变量必须全为零，SOC 链从传入的当前储电量起算。"""
    price48 = np.tile(att.price, 2)
    solver = StorageRollingSolver(price48)
    L, PV = att.load_kwh, att.pv_kwh
    g = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT).G0
    t, soc_prev = 60, 4321.0
    step = solver.solve(t, soc_prev, L[t] * 1.2, PV[t] * 0.8, L, PV, L, PV, g)
    for arr in (step.C, step.D, step.W, step.E, step.G):
        assert np.abs(arr[:t]).max() < 1e-9
    assert np.abs(step.S[:t] - soc_prev).max() < 1e-6
    assert abs(step.S[t] - (soc_prev + ETA * step.C[t] - step.D[t] / ETA)) < 1e-6
    # 当天剩余段的购电量被钉死为 G0，次日购电量自由
    assert np.abs(step.G[t:T] - g[t:]).max() < 1e-6
    assert step.S.min() >= SOC_MIN - 1e-6 and step.S.max() <= SOC_MAX + 1e-6


def test_rolling_first_segment_uses_truth_not_forecast(att):
    """首段用真值：真值缺口大于预测时，首段的紧急购电/放电必须据真值给出。"""
    price48 = np.tile(att.price, 2)
    solver = StorageRollingSolver(price48)
    L, PV = att.load_kwh, att.pv_kwh
    g = np.zeros(T)
    t = 100
    step = solver.solve(t, SOC_MIN, L[t] + 400.0, 0.0, L, PV, L, PV, g)
    net = L[t] + 400.0 + step.C[t] - step.D[t]
    assert abs(step.E[t] - step.W[t] - net) < 1e-6
    assert step.E[t] > 0 and step.W[t] < 1e-6


def test_saa_horizon_can_be_24h(att):
    """时域参数化：n_future=0 时退化为 24h 单日 LP。"""
    L, PV = att.load_kwh, att.pv_kwh
    saa = solve_saa(att.price, L[None, :], PV[None, :], np.zeros(0), np.zeros(0), SOC_INIT)
    assert saa.G0.shape == (T,)
    assert saa.plan_cost > 0
    assert abs(saa.objective - saa.plan_cost) < 100 * EPS * L.sum()
