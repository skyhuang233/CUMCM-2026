import numpy as np
import pytest
from datetime import date

from src.data import ETA, SOC_INIT, SOC_MAX, SOC_MIN, T, load_attachment1
from src.executor import run_day
from src.optimizer import StorageRollingSolver, solve_saa, solve_saa_point


@pytest.fixture(scope="module")
def point_case():
    """真值 = 点预测的理想日：SAA 冻结 G0，执行层用同一条曲线跑完一天。"""
    att = load_attachment1()
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    saa = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT)
    point = solve_saa_point(price48, L, PV, L, PV, SOC_INIT)
    solver = StorageRollingSolver(price48)
    execution = run_day(
        date(2025, 3, 20), saa.G0, SOC_INIT, L, PV, L, PV, L, PV, solver
    )
    return att, saa, point, execution


def test_no_emergency_when_truth_equals_point_forecast(point_case):
    _, _, point, ex = point_case
    assert point.E.max() < 1e-6
    assert ex.E.max() < 1e-6


def test_execution_invariants(point_case):
    _, _, _, ex = point_case
    assert np.minimum(ex.E, ex.W).max() < 1e-9  # E_t · W_t = 0
    assert np.minimum(ex.C, ex.D).max() < 1e-6  # 不同时充放电
    assert ex.S.min() >= SOC_MIN - 1e-6 and ex.S.max() <= SOC_MAX + 1e-6
    prev = np.concatenate([[ex.soc_start], ex.S[:-1]])
    assert np.abs(ex.S - prev - ETA * ex.C + ex.D / ETA).max() < 1e-6


def test_execution_balance_uses_truth(point_case):
    att, saa, _, ex = point_case
    residual = saa.G0 + ex.E + att.pv_kwh + ex.D - att.load_kwh - ex.C - ex.W
    assert np.abs(residual).max() < 1e-6


def test_emergency_appears_when_truth_exceeds_plan():
    """真值负载高于预测、光伏低于预测时必须出现紧急购电，且仍满足全部不变量。"""
    att = load_attachment1()
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    saa = solve_saa(price48, L[None, :], PV[None, :], L, PV, SOC_INIT)
    solver = StorageRollingSolver(price48)
    L_true, PV_true = L * 1.3, PV * 0.5
    ex = run_day(date(2025, 3, 20), saa.G0, SOC_INIT, L_true, PV_true, L, PV, L, PV, solver)
    assert ex.E.sum() > 0
    assert np.minimum(ex.E, ex.W).max() < 1e-9
    assert ex.S.min() >= SOC_MIN - 1e-6 and ex.S.max() <= SOC_MAX + 1e-6
    residual = saa.G0 + ex.E + PV_true + ex.D - L_true - ex.C - ex.W
    assert np.abs(residual).max() < 1e-6


def test_step_minutes_reduces_number_of_solves(monkeypatch):
    att = load_attachment1()
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    solver = StorageRollingSolver(price48)
    calls: list[int] = []
    original = StorageRollingSolver.solve

    def spy(self, t, *args, **kwargs):
        calls.append(t)
        return original(self, t, *args, **kwargs)

    monkeypatch.setattr(StorageRollingSolver, "solve", spy)
    ex = run_day(
        date(2025, 3, 20), np.zeros(T), SOC_INIT, L, PV, L, PV, L, PV, solver,
        step_minutes=60,
    )
    assert len(calls) == T // 6
    assert calls[:3] == [0, 6, 12]
    assert ex.S.shape == (T,)
    assert ex.S.min() >= SOC_MIN - 1e-6 and ex.S.max() <= SOC_MAX + 1e-6
