from datetime import date

import pytest

from src.data import SOC_INIT
from src.run_q2 import Params, load_bundle, run_period
from src.tuning import HORIZON_GRID, _january, select_k, select_kp

END = date(2025, 1, 5)
PARAMS = Params(k_load=4, k_pv=5, step_minutes=60)

@pytest.fixture(scope="module")
def bundle(): return load_bundle()

def test_normal_january_is_standby_but_calibration_executes(bundle):
    normal = run_period(date(2025,1,1), END, PARAMS, bundle)
    calibration = _january(bundle, PARAMS, END)
    assert normal.days == [] and normal.soc_end == SOC_INIT
    assert calibration.days and calibration.total_cost > 0 and calibration.soc_end != SOC_INIT

def test_score_window_excludes_preceding_calibration_days(bundle):
    res = select_k(bundle,(4,),(5,),base=PARAMS,end=END,score_from=date(2025,1,3),verbose=False)
    run = _january(bundle, PARAMS, END)
    expected=sum(d.plan_cost+d.emergency_cost for d in run.days if d.day >= date(2025,1,3))
    assert res.cost == pytest.approx(expected)
    assert res.cost < res.candidates[0].cost_full

def test_calibration_isolated_from_formal_february_state(bundle):
    calibration = _january(bundle, PARAMS, date(2025,1,31))
    formal = run_period(date(2025,1,1), date(2025,2,1),
                        Params(k_load=4,k_pv=5,m_scen=2,step_minutes=60), bundle)
    assert calibration.soc_end != SOC_INIT
    assert formal.days[0].soc_start == SOC_INIT

def test_tuning_is_explicit_increment_and_supported_horizons_only(bundle):
    result = select_k(bundle,(3,4),(5,),base=PARAMS,end=END,score_from=date(2025,1,3),verbose=False)
    assert len(result.candidates)==2 and result.cost==min(c.cost for c in result.candidates)
    assert HORIZON_GRID == (1,2)

def test_price_tuning_keeps_base_forecast_identity(monkeypatch, bundle):
    from types import SimpleNamespace
    from src.data import load_attachment4
    import src.tuning as tuning
    seen = []
    def fake_run(start, end, params, bundle, **kwargs):
        seen.append((params, kwargs['price_source']))
        day = SimpleNamespace(day=date(2025,1,8), plan_cost=2., emergency_cost=3., E=__import__('numpy').ones(1))
        return SimpleNamespace(days=[day], soc_end=SOC_INIT, total_cost=5.)
    monkeypatch.setattr(tuning, 'run_period', fake_run)
    base = Params(reference_baseline=True, horizon_days=2, scenario_same_type=True)
    select_kp(bundle, load_attachment4(), k_grid=(2,), base=base, verbose=False)
    params, source = seen[0]
    assert params.calibration and params.reference_baseline is True
    assert params.horizon_days == 2 and params.scenario_same_type is True
    assert source.mode == 'legacy_level'
