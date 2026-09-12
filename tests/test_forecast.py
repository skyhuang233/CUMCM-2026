from datetime import date
import numpy as np
from src.data import DailySeries, load_attachment1, load_attachment2
from src.forecast import LOW_START, PointForecaster, load_kind

def test_low_type_is_opposite_legacy_day_type():
    assert load_kind(date(2025,1,17)) == 0 and load_kind(date(2025,1,16)) == 1

def test_cold_start_and_january_warmup_are_causal():
    a,d = load_attachment1(),load_attachment2(); f=PointForecaster(d,a)
    l,p=f.predict(date(2025,1,1)); assert np.allclose(l,a.load_kwh) and np.allclose(p,a.pv_kwh)
    l,_=f.predict(date(2025,1,10)); hist=[i for i,x in enumerate(d.dates) if x < date(2025,1,10)]
    assert np.allclose(l,d.load_kwh[hist[-7:]].mean(0))

def test_shape_is_sum_of_curves_over_sum_of_totals():
    a,d=load_attachment1(),load_attachment2(); f=PointForecaster(d,a); day=date(2025,6,21)
    rows=[i for i,x in enumerate(d.dates) if x<day and load_kind(x)==load_kind(day)][-3:]
    l,_=f.predict(day); shape=d.load_kwh[rows].sum(0)/d.load_kwh[rows].sum()
    assert np.allclose(l/l.sum(),shape)

def test_future_poisoning_cannot_change_prediction():
    a,d=load_attachment1(),load_attachment2(); day=date(2025,5,2); i=d.dates.index(day)
    bad=DailySeries(d.dates,d.load_kwh.copy(),d.pv_kwh.copy()); bad.load_kwh[i:]=np.nan;bad.pv_kwh[i:]=np.nan
    assert np.allclose(PointForecaster(d,a).predict(day)[0],PointForecaster(bad,a).predict(day)[0])

def test_old_mean_is_explicit_increment():
    a,d=load_attachment1(),load_attachment2(); f=PointForecaster(d,a,reference_baseline=False); day=date(2025,6,21)
    rows=f._rows(day, f.k_load, 1); assert np.allclose(f.predict(day)[0],d.load_kwh[rows].mean(0))
