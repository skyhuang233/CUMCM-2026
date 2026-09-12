import numpy as np
from datetime import date, datetime, timedelta

from src.data import T, day_type
from src.scenarios import IssuedPathLibrary, ResidualLibrary


def _fill(lib: ResidualLibrary, days: list[date], tag: float) -> None:
    """给每个历史日灌入可辨识的残差：负载 +tag_i，光伏 −tag_i。"""
    for i, d in enumerate(days):
        marker = tag + i
        lib.update(
            d,
            np.full(T, 100.0 + marker),
            np.full(T, 1000.0 - marker),
            np.full(T, 100.0),
            np.full(T, 1000.0),
        )


def test_scenario_count_capped_by_m_and_same_day_type():
    lib = ResidualLibrary()
    days = [date(2025, 1, 1) + timedelta(days=k) for k in range(60)]
    _fill(lib, days, 1.0)
    d = date(2025, 3, 20)
    L, PV = lib.scenarios(d, np.full(T, 100.0), np.full(T, 1000.0), m=12)
    assert L.shape == (12, T) and PV.shape == (12, T)
    assert (L >= 0).all() and (PV >= 0).all()
    # 每条场景的负载与光伏残差必须来自同一历史日：r_L = +marker、r_PV = −marker
    r_load = L[:, 0] - 100.0
    r_pv = PV[:, 0] - 1000.0
    assert np.allclose(r_load, -r_pv)


def test_scenarios_exclude_the_decision_day_and_other_day_type():
    lib = ResidualLibrary()
    d = date(2025, 3, 20)  # 周四 → 日类型 0
    same_type = [h for h in (d - timedelta(days=k) for k in range(1, 20)) if day_type(h) == day_type(d)]
    _fill(lib, sorted(same_type), 1.0)
    # 决策日自身与异类型日也入库，但都不应被选中
    lib.update(d, np.full(T, 999.0), np.full(T, 0.0), np.full(T, 0.0), np.full(T, 0.0))
    other = date(2025, 3, 21)  # 周五 → 日类型 1
    lib.update(other, np.full(T, 888.0), np.full(T, 0.0), np.full(T, 0.0), np.full(T, 0.0))

    L, _ = lib.scenarios(d, np.full(T, 100.0), np.full(T, 1000.0), m=12)
    assert L.shape[0] == min(12, len(same_type))
    assert L.max() < 900.0


def test_cold_start_falls_back_to_point_forecast():
    lib = ResidualLibrary()
    d = date(2025, 1, 1)
    load_pred, pv_pred = np.full(T, 120.0), np.full(T, 60.0)
    L, PV = lib.scenarios(d, load_pred, pv_pred, m=12)
    assert L.shape == (1, T) and PV.shape == (1, T)
    assert np.allclose(L[0], load_pred) and np.allclose(PV[0], pv_pred)


def test_scenarios_are_clipped_at_zero():
    lib = ResidualLibrary()
    d = date(2025, 3, 20)
    h = d - timedelta(days=7)
    lib.update(h, np.zeros(T), np.zeros(T), np.full(T, 500.0), np.full(T, 500.0))
    L, PV = lib.scenarios(d, np.full(T, 100.0), np.full(T, 100.0), m=12)
    assert (L == 0).all() and (PV == 0).all()


def test_library_update_is_incremental():
    lib = ResidualLibrary()
    days = [date(2025, 1, 1) + timedelta(days=k) for k in range(10)]
    _fill(lib, days, 1.0)
    assert len(lib.entries) == 10
    assert sum(len(v) for v in lib._by_type.values()) == 10


def test_issued_paths_are_complete_paired_and_only_available_after_completion():
    lib = IssuedPathLibrary(); d = date(2025, 1, 2)
    for marker in (1., 2.):
        lib.update(d - timedelta(days=int(marker)), 6, np.full(T, marker),
                   np.full(T, -marker), np.full(T, 10*marker))
    L,V,P = lib.scenarios(datetime(2025,1,2,6), 6, np.zeros(T), np.full(T, 5.), np.ones(T), m=1)
    assert L.shape == V.shape == P.shape == (1,T)
    assert np.all(L[0] == 1.) and np.all(V[0] == 4.) and np.all(P[0] == 11.)
    L,_,_ = lib.scenarios(d-timedelta(days=1),6,np.ones(T),np.ones(T),m=30)
    assert L.shape == (1,T)  # newest issuance has not completed yet
    assert lib.select(d,6,m=0) == []
    import pytest
    with pytest.raises(ValueError):
        lib.update(d - timedelta(days=1), 6, np.zeros(T), np.zeros(T))


def test_issued_path_pool_can_explicitly_filter_day_type():
    asof = datetime(2025, 1, 20, 5)  # Monday, before Sunday's path completes
    full, typed = IssuedPathLibrary(), IssuedPathLibrary(same_type=True)
    for lib in (full, typed):
        lib.update(date(2025,1,17), 6, np.full(T, 1.), np.zeros(T))  # Friday type 1
        lib.update(date(2025,1,18), 6, np.full(T, 2.), np.zeros(T))  # Saturday type 1
        lib.update(date(2025,1,19), 6, np.full(T, 3.), np.zeros(T))  # Sunday type 0; not complete at 6
    # Full pool chooses the latest completed Saturday; typed pool has no
    # completed type-0 entry, so both expose their different selection rules.
    assert full.select(asof, 6, 1)[0].r_load[0] == 2.
    assert typed.select(asof, 6, 1) == []
