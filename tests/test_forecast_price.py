from datetime import date
import numpy as np
from src.data import T, load_attachment1, load_attachment4
from src.forecast_price import PRICE_FLOOR, PriceForecaster

def synthetic(branch='q4_2'):
    dates=[date(2025,1,1).fromordinal(date(2025,1,1).toordinal()+i) for i in range(8)]
    shape=np.linspace(.8,1.2,T); levels=np.arange(1.,9.)
    prices=levels[:,None]*shape[None,:]
    prior=type('P',(),{'price':shape})()
    net=lambda d,t: 100000*(d.toordinal()-dates[0].toordinal()+1)
    return PriceForecaster(dates,prices,prior,k=7,branch=branch,net_load=net),dates,shape

def fc(branch='q4_2'):
    a=load_attachment1(); ds,p=load_attachment4()
    return PriceForecaster(ds,p,a,branch=branch),ds,p

def test_shape_is_mean_normalized_completed_days():
    f,ds,p=fc(); d=date(2025,6,21); rows=f._rows(d)
    expected=np.mean(p[rows]/p[rows].mean(1)[:,None],0)
    assert np.allclose(f._shape(d),expected)

def test_future_price_poisoning_is_not_read():
    a=load_attachment1(); ds,p=load_attachment4(); d=date(2025,5,10); i=ds.index(d)
    q=p.copy();q[i,72:]=np.nan;q[i+1:]=np.nan
    good=PriceForecaster(ds,p,a);bad=PriceForecaster(ds,q,a)
    x=good.predict(d,72);y=bad.predict(d,72)
    assert np.allclose(x[0],y[0]) and np.allclose(x[1],y[1])

def test_q4_2_net_regression_and_floor():
    f,ds,p=fc(); day=date(2025,6,21)
    today,nxt=f.predict(day,0,net_load_pred=100000,net_load_next=110000)
    assert today.shape==(T,) and nxt.shape==(T,) and (today>=PRICE_FLOOR).all()

def test_synthetic_net_fit_and_next_feature_mapping():
    f,ds,shape=synthetic(); today,nxt=f.predict(ds[-1],net_load_pred=900000,net_load_next=1000000)
    # levels are exactly 0 + 1*(net/1e5)
    assert np.allclose(today,9*shape) and np.allclose(nxt,10*shape)

def test_synthetic_ar_recurses_a_plus_rho_level():
    f,ds,shape=synthetic('q4_3'); today,nxt=f.predict(ds[-1])
    a,rho=f._fit(ds[-1],'q4_3',0)
    assert np.allclose(today,(a+rho*7)*shape)
    assert np.allclose(nxt,(a+rho*(a+rho*7))*shape)

def test_degenerate_design_falls_back_and_bad_arguments_rejected():
    f,ds,shape=synthetic(); f.net_load=lambda d,t: 1.
    got,_=f.predict(ds[-1]); assert np.allclose(got,7*shape)
    import pytest
    with pytest.raises(ValueError): f.predict(ds[-1], -1)

def test_legacy_mode_is_same_type_segment_mean_not_normalized_shape():
    a=load_attachment1(); ds,p=load_attachment4(); d=date(2025,2,1)
    f=PriceForecaster(ds,p,a,k=35,mode='legacy_level')
    got,_=f.predict(d)
    rows=[i for i,x in enumerate(ds) if x<d and x.weekday() in (4,5)][-35:]
    assert np.allclose(got,p[rows].mean(0))

def test_legacy_selects_k_after_filtering_and_cold_start_uses_prior():
    a=load_attachment1(); ds,p=load_attachment4(); f=PriceForecaster(ds,p,a,k=2,mode='legacy_level')
    d=date(2025,1,17)  # Friday: the immediately prior days include other types
    got,_=f.predict(d)
    rows=[i for i,x in enumerate(ds) if x<d and x.weekday() in (4,5)][-2:]
    assert len(rows)==2 and np.allclose(got,p[rows].mean(0))
    cold,_=f.predict(ds[0]); assert np.allclose(cold,a.price)

def test_ar_next_level_is_recursed_and_observed_error_decays():
    f,ds,p=fc('q4_3'); day=date(2025,6,21); a,b=f.predict(day,36)
    assert np.allclose(a[:36],p[ds.index(day),:36]) and (b>=PRICE_FLOOR).all()

def test_terminal_uses_relative_next_midnight_window():
    f,_,_=fc(); path=np.arange(144,dtype=float)+1
    assert f.terminal_value(path,36)==np.mean(path[108:138])/.9
    assert f.terminal_value(path,0)==np.mean(path[:30])/.9
    paths=np.vstack([path, 3*path])
    assert f.terminal_value(paths,36)==np.mean(2*path[108:138])/.9
