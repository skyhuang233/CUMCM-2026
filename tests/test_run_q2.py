import numpy as np
import openpyxl
import pytest
from datetime import date

from src.data import SOC_INIT, SOC_MAX, SOC_MIN, T
from src.results import write_result2
from src.run_q2 import Params, load_bundle, run_period, validate


@pytest.fixture(scope="module")
def short_run():
    bundle = load_bundle()
    res = run_period(
        date(2025, 1, 1),
        date(2025, 1, 4),
        Params(step_minutes=60),
        bundle,
        record_from=date(2025, 1, 2),
        soc_init=SOC_INIT,
    )
    return bundle, res


def test_records_only_from_record_from(short_run):
    _, res = short_run
    assert [d.day for d in res.days] == [
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 4),
    ]


def test_soc_is_continuous_across_days_and_in_bounds(short_run):
    _, res = short_run
    for prev, cur in zip(res.days, res.days[1:]):
        assert abs(cur.soc_start - prev.soc_end) < 1e-9
    for day in res.days:
        assert day.S.min() >= SOC_MIN - 1e-6 and day.S.max() <= SOC_MAX + 1e-6
        assert np.minimum(day.C, day.D).max() < 1e-6
        assert np.minimum(day.E, day.W).max() < 1e-9


def test_cost_decomposition_matches_total(short_run):
    bundle, res = short_run
    price = bundle.att1.price
    for day in res.days:
        assert abs(day.plan_cost - float(price @ day.G0)) < 1e-6
        assert abs(day.emergency_cost - 5.0 * float(price @ day.E)) < 1e-6
        assert abs(day.total_cost - (day.plan_cost + day.emergency_cost)) < 1e-9
    assert abs(res.total_cost - sum(d.total_cost for d in res.days)) < 1e-6


def test_validate_passes_on_a_real_run(short_run):
    bundle, res = short_run
    validate(res, bundle)  # 平衡、E·W=0、SOC 界与跨日连续、费用分解


def test_validate_rejects_a_broken_soc_chain(short_run):
    import copy

    bundle, res = short_run
    broken = copy.deepcopy(res)
    broken.days[1].soc_start += 100.0
    with pytest.raises(AssertionError):
        validate(broken, bundle)


def test_write_result2_layout(short_run, tmp_path):
    _, res = short_run
    path = write_result2(res.days, str(tmp_path / "result2.xlsx"))
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["计划购电量", "充放电量", "紧急购电量"]
    plan = wb["计划购电量"]
    assert plan.max_row == len(res.days) + 1
    assert plan.max_column == T + 1
    assert plan.cell(1, 2).value == "0:10" and plan.cell(1, T + 1).value == "0:00+1"
    assert plan.cell(2, 1).value == res.days[0].day.isoformat()

    charge = wb["充放电量"]
    assert charge.max_column == 2 * 6 + 2 + 1
    assert charge.cell(1, 2).value == "0:00-4:00 充电量"
    assert charge.cell(1, charge.max_column).value == "24:00 储电量"

    emergency = wb["紧急购电量"]
    assert [c.value for c in emergency[1]] == ["日期", "紧急购电时间段", "紧急购电量"]
    # 同一日期多区间时日期只在首行填写
    seen_dates = [emergency.cell(r, 1).value for r in range(2, emergency.max_row + 1)]
    assert all(v is None or isinstance(v, str) for v in seen_dates)
