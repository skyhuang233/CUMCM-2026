import numpy as np
import openpyxl
import pytest
from datetime import date

from src.data import SOC_INIT, SOC_MAX, SOC_MIN, T
from src.results import write_result3
from src.run_q3 import Params, block_bounds_of, load_bundle, run_period, validate


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def short_run(bundle):
    res = run_period(
        date(2025, 1, 1),
        date(2025, 1, 4),
        Params(step_minutes=60),
        bundle,
        record_from=date(2025, 1, 2),
        soc_init=SOC_INIT,
    )
    return bundle, res


def test_block_bounds_follow_the_submit_rule():
    assert block_bounds_of((0, 6, 12, 18)) == [(0, 0, 36), (6, 36, 72), (12, 72, 108), (18, 108, 144)]
    assert block_bounds_of((0,)) == [(0, 0, 144)]
    assert block_bounds_of((0, 12)) == [(0, 0, 72), (12, 72, 144)]


def test_each_segment_has_exactly_one_submitted_value(short_run):
    """0:00–6:00 段 $G^a=G^0$；全部段 $G^a\\ge0$；形状 144。"""
    _, res = short_run
    for day in res.days:
        assert day.Ga.shape == (T,)
        assert np.allclose(day.Ga[:36], day.G0[:36])
        assert day.Ga.min() >= -1e-9


def test_readjustment_only_moves_the_current_block(bundle):
    """6:00 重优化后只有 36–71 段可能变，72–143 段直到 12:00 之前仍等于 $G^0$。"""
    only_six = run_period(
        date(2025, 1, 1),
        date(2025, 1, 2),
        Params(step_minutes=60, issues=(0, 6)),
        bundle,
        record_from=date(2025, 1, 2),
        soc_init=SOC_INIT,
    )
    six_and_twelve = run_period(
        date(2025, 1, 1),
        date(2025, 1, 2),
        Params(step_minutes=60, issues=(0, 6, 12)),
        bundle,
        record_from=date(2025, 1, 2),
        soc_init=SOC_INIT,
    )
    a, b = only_six.days[0], six_and_twelve.days[0]
    # 两个变体在 12:00 之前完全一致：同一个 0:00 计划、同一次 6:00 重优化
    assert np.allclose(a.G0, b.G0)
    assert np.allclose(a.Ga[:72], b.Ga[:72])
    # 只有 0,6 时 6:00 的提交一直延伸到当天末；加入 12:00 后 72 段起被覆盖
    assert not np.allclose(a.Ga[72:], b.Ga[72:])


def test_soc_is_continuous_across_blocks_and_days(short_run):
    _, res = short_run
    for prev, cur in zip(res.days, res.days[1:]):
        assert abs(cur.soc_start - prev.soc_end) < 1e-9
    for day in res.days:
        assert day.S.min() >= SOC_MIN - 1e-6 and day.S.max() <= SOC_MAX + 1e-6
        assert np.minimum(day.C, day.D).max() < 1e-6
        assert np.minimum(day.E, day.W).max() < 1e-9
        assert (day.S > 0).all()  # 四个区间都被执行填满，没有留空段


def test_cost_decomposition_sums_to_total(short_run):
    _, res = short_run
    for day in res.days:
        parts = day.plan_cost + day.curtail_cost + day.extra_cost + day.emergency_cost
        assert abs(day.total_cost - parts) < 1e-6
    assert abs(res.total_cost - sum(d.total_cost for d in res.days)) < 1e-6


def test_execution_balances_against_the_submitted_purchase(short_run):
    bundle, res = short_run
    for day in res.days:
        load_true, pv_true = bundle.truth(day.day)
        residual = day.Ga + day.E + pv_true + day.D - load_true - day.C - day.W
        assert np.abs(residual).max() < 1e-6


def test_validate_passes_and_catches_a_broken_submit_rule(short_run):
    import copy

    bundle, res = short_run
    validate(res, bundle, (0, 6, 12, 18))
    broken = copy.deepcopy(res)
    broken.days[0].Ga[10] += 50.0  # 0:00–6:00 段不允许偏离 G^0
    with pytest.raises(AssertionError):
        validate(broken, bundle, (0, 6, 12, 18))


def test_write_result3_layout(short_run, tmp_path):
    _, res = short_run
    path = write_result3(res.days, str(tmp_path / "result3.xlsx"))
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["计划购电量", "调整购电量", "充放电量", "紧急购电量"]
    for name in ("计划购电量", "调整购电量"):
        ws = wb[name]
        assert ws.max_row == len(res.days) + 1
        assert ws.max_column == T + 1
        assert ws.cell(1, 2).value == "0:10" and ws.cell(1, T + 1).value == "0:00+1"
        assert ws.cell(2, 1).value == res.days[0].day.isoformat()
    charge = wb["充放电量"]
    assert charge.max_column == 2 * 6 + 2 + 1
    assert charge.cell(1, charge.max_column).value == "24:00 储电量"
    assert [c.value for c in wb["紧急购电量"][1]] == ["日期", "紧急购电时间段", "紧急购电量"]


def test_issue_set_must_contain_midnight(bundle):
    with pytest.raises(AssertionError):
        run_period(
            date(2025, 1, 1),
            date(2025, 1, 1),
            Params(step_minutes=60, issues=(6, 12)),
            bundle,
        )


def test_no_future_data_reaches_any_decision(bundle):
    """把末日之后的真值与预报全部毒化成 NaN：$G^0$、$G^a$ 与费用必须逐位不变。"""
    from src.data import DailySeries

    d0, d1 = date(2025, 1, 1), date(2025, 1, 3)
    params = Params(step_minutes=60)
    clean = run_period(d0, d1, params, bundle, soc_init=SOC_INIT)

    i = bundle.daily.dates.index(d1)
    load = bundle.daily.load_kwh.copy()
    pv = bundle.daily.pv_kwh.copy()
    load[i + 1 :] = np.nan
    pv[i + 1 :] = np.nan
    att3 = {
        k: (v if k[0] <= d1 else np.full(24, np.nan)) for k, v in bundle.att3.items()
    }
    poisoned = type(bundle)(
        att1=bundle.att1,
        daily=DailySeries(dates=bundle.daily.dates, load_kwh=load, pv_kwh=pv),
        att3=att3,
    )
    guarded = run_period(d0, d1, params, poisoned, soc_init=SOC_INIT)

    for a, b in zip(clean.days, guarded.days):
        assert np.array_equal(a.G0, b.G0)
        assert np.array_equal(a.Ga, b.Ga)
    assert clean.total_cost == guarded.total_cost
