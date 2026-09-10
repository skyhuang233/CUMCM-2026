import numpy as np

from src.data import T
from src.settlement import cost_q3


def test_curtailment_pays_plan_on_the_kept_part_plus_half_on_the_cancelled_part():
    """$G^0=100, G^a=80, p=1$ → 计划费 80 + 减购违约 10。"""
    price = np.ones(1)
    c = cost_q3(price, np.array([100.0]), np.array([80.0]), np.zeros(1))
    assert c["plan"] == 80.0
    assert c["curtail_penalty"] == 10.0
    assert c["extra"] == 0.0
    assert c["emergency"] == 0.0
    assert c["total"] == 90.0


def test_extra_purchase_pays_plan_plus_one_and_a_half_on_the_excess():
    """$G^0=100, G^a=120, p=1$ → 计划费 100 + 增购 30。"""
    price = np.ones(1)
    c = cost_q3(price, np.array([100.0]), np.array([120.0]), np.zeros(1))
    assert c["plan"] == 100.0
    assert c["extra"] == 30.0
    assert c["curtail_penalty"] == 0.0
    assert c["total"] == 130.0


def test_no_adjustment_reduces_to_q2_pricing():
    from src.settlement import cost_q2

    rng = np.random.default_rng(11)
    price = rng.uniform(0.3, 1.2, T)
    G0 = rng.uniform(0.0, 900.0, T)
    E = np.zeros(T)
    E[30:34] = 25.0
    c = cost_q3(price, G0, G0, E)
    plan_q2, emergency_q2 = cost_q2(price, G0, E)
    assert abs(c["plan"] - plan_q2) < 1e-9
    assert abs(c["emergency"] - emergency_q2) < 1e-9
    assert c["curtail_penalty"] == 0.0 and c["extra"] == 0.0
    assert abs(c["total"] - (plan_q2 + emergency_q2)) < 1e-9


def test_decomposition_sums_to_total_on_mixed_adjustments():
    rng = np.random.default_rng(7)
    price = rng.uniform(0.3, 1.2, T)
    G0 = rng.uniform(0.0, 900.0, T)
    Ga = np.clip(G0 + rng.normal(0.0, 200.0, T), 0.0, None)
    E = np.clip(rng.normal(0.0, 30.0, T), 0.0, None)
    c = cost_q3(price, G0, Ga, E)
    parts = c["plan"] + c["curtail_penalty"] + c["extra"] + c["emergency"]
    assert abs(parts - c["total"]) < 1e-6
    # 逐段闭式与向量化结果一致
    per_seg = (
        price * np.minimum(G0, Ga)
        + 0.5 * price * np.maximum(G0 - Ga, 0.0)
        + 1.5 * price * np.maximum(Ga - G0, 0.0)
        + 5.0 * price * E
    )
    assert abs(float(per_seg.sum()) - c["total"]) < 1e-6


def test_emergency_is_five_times_price():
    price = np.full(T, 0.5)
    c = cost_q3(price, np.zeros(T), np.zeros(T), np.ones(T))
    assert abs(c["emergency"] - 5.0 * 0.5 * T) < 1e-9
    assert c["total"] == c["emergency"]
