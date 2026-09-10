import numpy as np
from datetime import date

from src.data import T
from src.results import merge_emergency_intervals
from src.settlement import cost_q2, emergency_intervals, interval_label


def test_emergency_intervals_merge_consecutive_segments():
    E = np.zeros(T)
    E[78] = 100.0
    E[79] = 200.0
    E[80] = 300.0
    E[88] = 50.0
    out = emergency_intervals(date(2025, 3, 20), E)
    assert out == [("13:00-13:30", 600.0), ("14:40-14:50", 50.0)]


def test_interval_label_endpoints():
    assert interval_label(0, 0) == "0:00-0:10"
    assert interval_label(78, 80) == "13:00-13:30"
    assert interval_label(143, 143) == "23:50-24:00"


def test_merge_returns_segment_indices():
    E = np.zeros(T)
    E[10:13] = 1.0
    assert merge_emergency_intervals(E) == [(10, 12, 3.0)]
    assert merge_emergency_intervals(np.zeros(T)) == []


def test_cost_decomposition_sums_to_total():
    rng = np.random.default_rng(3)
    price = rng.uniform(0.3, 1.2, T)
    G0 = rng.uniform(0, 900, T)
    E = np.zeros(T)
    E[20:24] = 30.0
    plan, emergency = cost_q2(price, G0, E)
    assert abs(plan - float(price @ G0)) < 1e-9
    assert abs(emergency - 5.0 * float(price @ E)) < 1e-9
    assert abs((plan + emergency) - float(price @ (G0 + 5.0 * E))) < 1e-9


def test_zero_emergency_costs_nothing():
    price = np.full(T, 0.5)
    plan, emergency = cost_q2(price, np.ones(T), np.zeros(T))
    assert emergency == 0.0
    assert abs(plan - 0.5 * T) < 1e-9
    assert emergency_intervals(date(2025, 1, 1), np.zeros(T)) == []
