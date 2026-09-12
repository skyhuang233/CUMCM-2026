"""Independent LP checks for the continuous (not gridded) value DP."""
import numpy as np
from scipy.optimize import linprog

from src.data import ETA, P_MAX_KWH, SOC_INIT, SOC_MAX, SOC_MIN
from src.value_dp import ConvexPiecewiseLinear, evaluate_plan, future_cost, next_day_value, periodic_value, recover_periodic_actions, reserve_level


def _lp(price, load, pv, soc, *, g=None, terminal=None):
    """Small independent physical LP with a literal fixed terminal when given."""
    n=len(price); fixed=g is not None; g=np.zeros(n) if g is None else np.asarray(g,float)
    # free G, C, D, W, S; fixed-plan case adds E and fixes G via bounds
    cols=6 if fixed else 5; G,C,D,W,S=(0,1,2,3,4); E=5
    c=np.zeros(cols*n); c[G*n:(G+1)*n]=price
    if fixed:c[E*n:(E+1)*n]=5*np.asarray(price)
    A=[]; b=[]
    for t in range(n):
        row=np.zeros(cols*n); row[G*n+t]=1; row[D*n+t]=1; row[C*n+t]=-1; row[W*n+t]=-1
        if fixed:row[E*n+t]=1
        A.append(row); b.append(load[t]-pv[t])
        row=np.zeros(cols*n); row[S*n+t]=1; row[C*n+t]=-ETA; row[D*n+t]=1/ETA
        if t:row[S*n+t-1]=-1; rhs=0
        else:rhs=soc
        A.append(row); b.append(rhs)
    if terminal is not None:
        row=np.zeros(cols*n); row[S*n+n-1]=1; A.append(row); b.append(terminal)
    bounds=[]
    for block in range(cols):
        for t in range(n):
            if block==G: bounds.append((g[t],g[t]) if fixed else (0,None))
            elif block in (C,D): bounds.append((0,P_MAX_KWH))
            elif block==S: bounds.append((SOC_MIN,SOC_MAX))
            else: bounds.append((0,None))
    out=linprog(c,A_eq=np.array(A),b_eq=np.array(b),bounds=bounds,method="highs")
    assert out.success
    return out.fun


def test_q1_exact_dp_matches_independent_original_objective_lp():
    price=np.array([.4,1.1,.7,.2]); load=np.array([900.,120.,700.,400.]); pv=np.array([100.,500.,0.,0.])
    got=periodic_value(price,load,pv,SOC_INIT)(SOC_INIT)
    want=_lp(price,load,pv,SOC_INIT,terminal=SOC_INIT)
    assert abs(got-want)<1e-7


def test_q1_forward_recovery_returns_feasible_periodic_actions():
    p=np.array([.4,1.1,.7,.2]); l=np.array([900.,120.,700.,400.]); v=np.array([100.,500.,0.,0.])
    out=recover_periodic_actions(p,l,v,SOC_INIT)
    assert abs(out.objective-periodic_value(p,l,v,SOC_INIT)(SOC_INIT))<1e-7
    assert abs(out.S[-1]-SOC_INIT)<1e-7
    assert np.max(np.abs(out.G+v+out.D-l-out.C-out.W))<1e-7
    assert np.minimum(out.C,out.D).max()<1e-9


def test_q1_forward_recovery_accounts_for_surplus_as_waste():
    out=recover_periodic_actions(np.array([1.,1.]),np.array([0.,0.]),np.array([3000.,0.]),SOC_INIT)
    assert abs(out.S[-1]-SOC_INIT)<1e-7
    assert out.W[0]>0
    assert np.max(np.abs(out.G+np.array([3000.,0.])+out.D-out.C-out.W))<1e-7


def test_free_purchase_dp_matches_lp_at_non_grid_inventory():
    price=np.array([.4,1.1,.7]); load=np.array([900.,120.,700.]); pv=np.array([100.,500.,0.]); stock=4321.234
    assert abs(next_day_value(price,load,pv)(stock)-_lp(price,load,pv,stock))<1e-7


def test_fixed_purchase_future_cost_matches_independent_lp_with_terminal():
    price=np.array([.4,1.1,.7]); load=np.array([[900.,120.,700.]]); pv=np.array([[100.,500.,0.]])
    g=np.array([300.,300.,300.]); terminal=ConvexPiecewiseLinear.point(6000.)
    got=future_cost(price,load,pv,g,terminal)[0](6000.)
    # H excludes the already-committed ordinary purchase bill.
    assert abs(got-(_lp(price,load[0],pv[0],6000.,g=g,terminal=6000.)-price@g))<1e-7


def test_single_point_terminal_is_not_a_large_penalty_or_grid():
    f=ConvexPiecewiseLinear.point(6000.)
    assert np.isinf(f(5999.999)) and f(6000.)==0
    with np.testing.assert_raises(ValueError): next_day_value([1],[1],[0],grid=np.linspace(SOC_MIN,SOC_MAX,3))


def test_true_nonconvex_polyline_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "non-convex"):
        ConvexPiecewiseLinear(np.array([0., 1., 2.]), np.array([0., 2., 1.]))


def test_internal_constructor_retains_polyline_invariant_checks():
    with np.testing.assert_raises_regex(ValueError, "invalid breakpoints"):
        ConvexPiecewiseLinear._internal([1.0, 0.0], [0.0, 1.0])
    with np.testing.assert_raises_regex(ValueError, "non-convex"):
        ConvexPiecewiseLinear._internal([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])


def test_historical_soc_min_near_duplicate_is_stably_merged():
    # Q2 2025-02-01 previously produced this clipped zero-width prefix;
    # keeping it made a false 3.6667 -> 3.4107 slope inversion.
    f = ConvexPiecewiseLinear(
        np.array([1200.0, 1200.0 + 1e-10, 1200.86605904, 1224.95030531]),
        np.array([6172.20083346, 6172.20083346, 6175.15474000, 6257.30001980]),
    )
    assert f.breakpoints.size == 3
    assert np.all(np.diff(f.slopes) >= -1e-7)


def test_near_duplicate_merge_keeps_right_finite_domain_boundary():
    f = ConvexPiecewiseLinear(
        np.array([1200.0, 10799.9999999999, 10800.0]), np.array([0.0, 0.0, 0.0])
    )
    assert f.hi == SOC_MAX
    assert f(SOC_MAX) == 0.0


def test_real_narrow_affine_piece_is_not_merged_as_roundoff():
    f = ConvexPiecewiseLinear(
        np.array([1200.0, 1200.000001, 1201.0]), np.array([0.0, 0.000001, 2.0])
    )
    assert f.breakpoints.size == 3


def test_unsorted_breakpoints_are_rejected_before_numerical_merging():
    with np.testing.assert_raises_regex(ValueError, "invalid breakpoints"):
        ConvexPiecewiseLinear(np.array([1.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]))


def test_reserve_uses_right_end_of_flat_minimum():
    f=ConvexPiecewiseLinear(np.array([SOC_MIN,5000.,6000.,SOC_MAX]),np.array([1.,0.,0.,2.]))
    assert reserve_level(f,0.)==6000.


def test_nearby_affine_breakpoints_do_not_amplify_ordinate_roundoff():
    import pytest

    x = np.array([4878.97538874, 4878.97538941, 4897.46699308])
    y = 12000.0 - .899833 * (x - x[0])
    for construct in (ConvexPiecewiseLinear, ConvexPiecewiseLinear._internal):
        f = construct(x, y)
        np.testing.assert_array_equal(f.x, x)
        np.testing.assert_array_equal(f.y, y)
        # A material concave peak at the same narrow interval still fails.
        nonconvex = y.copy()
        nonconvex[1] += 1e-6
        with pytest.raises(ValueError, match="non-convex"):
            construct(x, nonconvex)


def test_reserve_matches_affine_shift_reference_on_random_clipped_domains():
    rng = np.random.default_rng(20260912)
    for _ in range(100):
        x = np.r_[SOC_MIN, np.sort(rng.uniform(SOC_MIN, SOC_MAX, 7)), SOC_MAX]
        slopes = np.sort(rng.uniform(-2.0, 2.0, x.size - 1))
        y = np.empty_like(x)
        y[0] = rng.uniform(-100.0, 100.0)
        y[1:] = y[0] + np.cumsum(slopes * np.diff(x))
        f = ConvexPiecewiseLinear(x, y)
        lo, hi = np.sort(rng.uniform(SOC_MIN, SOC_MAX, 2))
        price = rng.uniform(0.0, 3.0)
        expected = f.add_linear(5.0 * price * ETA).max_argmin(lo, hi)
        assert reserve_level(f, price, lo=lo, hi=hi) == expected


def test_candidate_score_uses_path_specific_continuation():
    # Two correlated paths share the current-action H-bar, but their midnight
    # stocks must enter their *own* free-purchase K continuation.  This is a
    # regression counterexample: replacing K_w by average K gives 7487.5.
    p=np.array([[2.,.1],[2.,5.]])
    load=np.array([[900.,3000.],[0.,3000.]])
    pv=np.array([[0.,0.],[900.,0.]])
    terminal=ConvexPiecewiseLinear.constant(0.)
    score=evaluate_plan([0.],load,pv,p.mean(axis=0),2000.,terminal,price_scen=p)
    assert abs(score-6466.666666666667)<2e-6
    assert abs(score-7487.5)>100.
