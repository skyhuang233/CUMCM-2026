import pytest
from datetime import date

from src.run_q2 import Params, load_bundle, run_period
from src.tuning import JAN_SCORE_FROM, select_k

# 小网格 + 60 分钟步长，让测试在秒级完成；结构与全网格完全相同。
END = date(2025, 1, 5)
PARAMS = Params(k_load=4, k_pv=5, step_minutes=60)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def january(bundle):
    return run_period(date(2025, 1, 1), END, PARAMS, bundle)


def test_score_from_excludes_cold_start_days(bundle, january):
    """得分只累计 score_from 及之后的日子，预热日照跑但不计分。"""
    score_from = date(2025, 1, 3)
    res = select_k(
        bundle, (4,), (5,), base=PARAMS, end=END, score_from=score_from, verbose=False
    )
    scored = [d for d in january.days if d.day >= score_from]
    expected = sum(d.plan_cost + d.emergency_cost for d in scored)
    assert len(scored) < len(january.days)  # 确实排除了冷启动日
    assert abs(res.cost - expected) < 1e-6
    assert abs(res.candidates[0].cost_full - january.total_cost) < 1e-6
    assert res.cost < res.candidates[0].cost_full


def test_score_from_defaults_to_january_8():
    assert JAN_SCORE_FROM == date(2025, 1, 8)


def test_warmup_is_identical_regardless_of_score_from(bundle, january):
    """改变计分窗口不改变回测本身：整月费用与 1 月末 SOC 必须一致。"""
    a = select_k(
        bundle, (4,), (5,), base=PARAMS, end=END,
        score_from=date(2025, 1, 1), verbose=False,
    )
    b = select_k(
        bundle, (4,), (5,), base=PARAMS, end=END,
        score_from=date(2025, 1, 4), verbose=False,
    )
    assert abs(a.candidates[0].cost_full - b.candidates[0].cost_full) < 1e-6
    assert abs(a.candidates[0].soc_end - b.candidates[0].soc_end) < 1e-9
    assert abs(a.cost - january.total_cost) < 1e-6  # 窗口 = 全区间时退化为整月总费用


def test_result_table_and_lookup(bundle):
    res = select_k(
        bundle, (3, 4), (5,), base=PARAMS, end=END,
        score_from=date(2025, 1, 3), verbose=False,
    )
    assert len(res.candidates) == 2
    assert res.get(4, 5) is not None and res.get(9, 9) is None
    assert res.cost == min(c.cost for c in res.candidates)
    table = res.table()
    assert "2025-01-03→2025-01-05" in table
    assert table.count("\n") == 1 + 2  # 表头两行 + 2 个候选
