import numpy as np
import pytest

from src.data import SOC_INIT, T, load_attachment1
from src.optimizer import solve_readjust, solve_saa, solve_saa_point


@pytest.fixture(scope="module")
def att():
    return load_attachment1()


@pytest.fixture(scope="module")
def case(att):
    """点预测 = 单一场景的理想日：48h 确定性解给出 G^0 与逐段 SOC。"""
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    point = solve_saa_point(price48, L, PV, L, PV, SOC_INIT)
    return price48, L, PV, point


def test_no_adjustment_when_nothing_changed_at_midnight(att):
    """场景与点预测都没变、$G^0$ 取自同一数据的 SAA 解时，$t_0=0$ 的重优化不调整。"""
    price48 = np.tile(att.price, 2)
    L, PV = att.load_kwh, att.pv_kwh
    scen_L = np.stack([L * 0.95, L, L * 1.05])
    scen_PV = np.stack([PV, PV * 0.9, PV * 1.1])
    saa = solve_saa(price48, scen_L, scen_PV, L, PV, SOC_INIT)
    rj = solve_readjust(0, price48, saa.G0, scen_L, scen_PV, L, PV, SOC_INIT)
    assert np.abs(rj.dplus).max() < 1e-6
    assert np.abs(rj.dminus).max() < 1e-6
    assert np.abs(rj.Ga - saa.G0).max() < 1e-6
    assert abs(rj.adjust_cost) < 1e-6


def test_no_adjustment_at_0600_when_nothing_changed(case):
    """带着最优轨迹的 $S_{35}$ 在 6:00 重优化，同样不该调整（LP 的后缀最优性）。"""
    price48, L, PV, point = case
    t0 = 36
    rj = solve_readjust(
        t0, price48, point.G[:T], L[None, t0:], PV[None, t0:], L, PV, float(point.S[t0 - 1])
    )
    assert np.abs(rj.dplus).max() < 1e-6
    assert np.abs(rj.dminus).max() < 1e-6
    assert np.abs(rj.Ga - point.G[:T]).max() < 1e-6


def test_past_segments_are_untouched(case):
    price48, L, PV, point = case
    t0 = 72
    G0 = point.G[:T]
    rj = solve_readjust(
        t0, price48, G0, L[None, t0:] * 1.2, PV[None, t0:] * 0.4, L, PV, 6000.0
    )
    assert np.abs(rj.Ga[:t0] - G0[:t0]).max() < 1e-9
    assert np.abs(rj.dplus[:t0]).max() < 1e-12
    assert np.abs(rj.dminus[:t0]).max() < 1e-12


def test_delta_pair_is_complementary_and_bounded_by_plan(case):
    """$\\Delta^+_t\\Delta^-_t = 0$、$\\Delta^-_t\\le G^0_t$、$G^a\\ge0$。"""
    price48, L, PV, point = case
    t0 = 36
    G0 = point.G[:T]
    for scale_l, scale_pv in ((1.25, 0.4), (0.7, 1.6)):
        rj = solve_readjust(
            t0,
            price48,
            G0,
            L[None, t0:] * scale_l,
            PV[None, t0:] * scale_pv,
            L,
            PV,
            float(point.S[t0 - 1]),
        )
        assert np.minimum(rj.dplus, rj.dminus).max() < 1e-6
        assert (rj.dminus <= G0 + 1e-9).all()
        assert rj.Ga.min() >= -1e-9


def test_halving_pv_forces_extra_purchase_not_curtailment(case):
    """把 $t\\ge t_0$ 的光伏预测下调 50% 时，缺口只能靠增购补：$\\sum\\Delta^+>0$，$\\Delta^-=0$。"""
    price48, L, PV, point = case
    t0 = 36
    G0 = point.G[:T]
    rj = solve_readjust(
        t0, price48, G0, L[None, t0:], PV[None, t0:] * 0.5, L, PV, float(point.S[t0 - 1])
    )
    assert rj.dplus.sum() > 0.0
    assert rj.dminus.max() < 1e-6
    assert rj.adjust_cost > 0.0


def test_surplus_pv_triggers_curtailment_of_the_plan(case):
    """光伏远超计划时会减购：取消部分只赔 50%，比买下来白白富余便宜。"""
    price48, L, PV, point = case
    t0 = 36
    G0 = point.G[:T]
    rj = solve_readjust(
        t0, price48, G0, L[None, t0:] * 0.5, PV[None, t0:] * 2.0, L, PV, float(point.S[t0 - 1])
    )
    assert rj.dminus.sum() > 0.0
    assert rj.adjust_cost < 0.0  # 减购省下的计划费大于违约金


def test_readjust_objective_includes_the_frozen_plan_cost(case):
    """目标里含常数 $\\sum_{t\\ge t_0} p_tG^0_t$；不调整时它就是全部第一阶段费用。"""
    price48, L, PV, point = case
    t0 = 36
    G0 = point.G[:T]
    rj = solve_readjust(
        t0, price48, G0, L[None, t0:], PV[None, t0:], L, PV, float(point.S[t0 - 1])
    )
    plan_rem = float(price48[t0:T] @ G0[t0:])
    assert rj.objective > plan_rem  # 还含次日购电与吞吐惩罚
    assert abs(rj.adjust_cost) < 1e-6
