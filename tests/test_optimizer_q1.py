import numpy as np
import pytest

from src.data import ETA, P_MAX_KWH, SOC_INIT, SOC_MAX, SOC_MIN, T, load_attachment1
from src.optimizer import solve_deterministic
from src.results import interval_sums, summarize_q1


@pytest.fixture(scope="module")
def solved():
    att = load_attachment1()
    res = solve_deterministic(att.price, att.load_kwh, att.pv_kwh, SOC_INIT, periodic=True)
    return att, res


def test_purchase_cost_matches_reference(solved):
    _, res = solved
    assert abs(res.purchase_cost - 35126.9) < 1.0


def test_no_simultaneous_charge_discharge(solved):
    _, res = solved
    assert np.minimum(res.C, res.D).max() < 1e-6


def test_balance_and_soc_dynamics_residuals(solved):
    att, res = solved
    balance = res.G + att.pv_kwh + res.D - att.load_kwh - res.C - res.W
    assert np.abs(balance).max() < 1e-6
    prev = np.concatenate([[SOC_INIT], res.S[:-1]])
    soc = res.S - prev - ETA * res.C + res.D / ETA
    assert np.abs(soc).max() < 1e-6


def test_bounds_and_periodicity(solved):
    _, res = solved
    assert res.S.min() >= SOC_MIN - 1e-6
    assert res.S.max() <= SOC_MAX + 1e-6
    assert max(res.C.max(), res.D.max()) <= P_MAX_KWH + 1e-6
    assert res.G.min() >= -1e-9
    assert abs(res.S[-1] - SOC_INIT) < 1e-6


def test_periodic_dual_price(solved):
    _, res = solved
    assert abs(abs(res.duals["periodic"]) - 0.471) < 0.01


def test_surplus_only_when_pv_exceeds_load(solved):
    att, res = solved
    assert res.W[att.pv_kwh <= att.load_kwh].max() < 1e-6


def test_summary_shapes(solved):
    att, res = solved
    s = summarize_q1(res, att.price, SOC_INIT)
    assert len(s["seg_purchase"]) == 6
    assert len(s["charge"]) == 6 and len(s["discharge"]) == 6
    assert abs(s["daily_purchase"] - res.G.sum()) < 1e-9
    assert abs(interval_sums(res.C).sum() - res.C.sum()) < 1e-9


def test_assembler_is_parameterized_in_T():
    """同一装配路径应能处理任意段数（后续单元的 48h 时域）。"""
    n = 12
    price = np.linspace(0.4, 1.0, n)
    load = np.full(n, 3000.0)
    pv = np.zeros(n)
    res = solve_deterministic(price, load, pv, SOC_INIT, periodic=True)
    assert res.G.shape == (n,)
    assert abs(res.S[-1] - SOC_INIT) < 1e-6
    assert n != T
