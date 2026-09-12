"""Exact continuous convex piecewise-linear storage dynamic programmes."""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np

from .data import ETA, P_MAX_KWH, SOC_MAX, SOC_MIN

_TOL = 1e-9
_ULP_MULTIPLIER = 1024.0


def _merge_near_duplicate_breakpoints(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove zero-width numerical segments created by repeated clipping.

    Convex recurrences can independently create ``SOC_MIN`` and a value one
    ulp above it.  Keeping that almost-zero segment amplifies roundoff into a
    spurious slope inversion.  The left endpoint is retained, so domains and
    all material affine pieces are unchanged.
    """
    if x.size < 2:
        return x, y
    spacing = np.spacing(np.abs(x))
    gaps = np.diff(x)
    adjacent_tol = _ULP_MULTIPLIER * np.maximum(spacing[:-1], spacing[1:])
    # Most recurrence outputs have no near-coincident breakpoints. Avoid a
    # Python loop in that common case; the fallback preserves the exact
    # previous-kept-point semantics when a merge is actually needed.
    if np.all(gaps > adjacent_tol):
        return x, y
    keep = [0]
    for i in range(1, x.size):
        tolerance = _ULP_MULTIPLIER * max(spacing[keep[-1]], spacing[i])
        if x[i] - x[keep[-1]] > tolerance:
            keep.append(i)
        elif i == x.size - 1:
            # Preserve the right finite-domain boundary.  It is semantically
            # observable even when its final affine segment is near-zero.
            keep[-1] = i
    return x[keep], y[keep]


def _validated_polyline_arrays(x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce, normalize, and verify a convex polyline representation.

    The DP-only constructor uses this too: it skips dataclass initialization,
    not the checks that expose a malformed numerical recurrence.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    # Action costs and their reflections are frequently one/two-point
    # curves. Scalar checks retain the full contract without allocating
    # several temporary NumPy arrays for these tiny inputs.
    if x.size <= 2:
        if x.size != y.size or not x.size:
            raise ValueError("invalid breakpoints")
        x0, y0 = float(x[0]), float(y[0])
        if not math.isfinite(x0) or not math.isfinite(y0):
            raise ValueError("invalid breakpoints")
        if x.size == 1:
            return x, y, np.empty(0)
        x1, y1 = float(x[1]), float(y[1])
        if not math.isfinite(x1) or not math.isfinite(y1) or x1 <= x0:
            raise ValueError("invalid breakpoints")
        tolerance = _ULP_MULTIPLIER * max(math.ulp(x0), math.ulp(x1))
        if x1 - x0 <= tolerance:
            return x[1:], y[1:], np.empty(0)
        return x, y, np.array([(y1-y0)/(x1-x0)])
    if (
        x.size != y.size
        or not x.size
        or np.any(~np.isfinite(x))
        or np.any(~np.isfinite(y))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("invalid breakpoints")
    x, y = _merge_near_duplicate_breakpoints(x, y)
    widths = np.diff(x)
    slopes = np.diff(y) / widths
    if slopes.size > 1 and np.any(np.diff(slopes) < -1e-7):
        # Subtracting close ordinates and dividing by a very narrow interval
        # amplifies a few ulps into a spurious slope inversion. Verify the
        # equivalent chord inequality in cost units before rejecting it.
        # Preserve every breakpoint/value; this is a validation tolerance,
        # not a hull projection or a relaxation of the optimization model.
        span = widths[:-1] + widths[1:]
        chord = y[:-2] + (widths[:-1] / span) * (y[2:] - y[:-2])
        scale = np.maximum.reduce([np.abs(y[:-2]), np.abs(y[1:-1]), np.abs(y[2:])])
        allowance = 8.0 * np.spacing(scale) + 1e-7 * widths[:-1] * widths[1:] / span
        failures = np.flatnonzero(y[1:-1] - chord > allowance)
        if failures.size:
            bad = int(failures[0])
            raise ValueError(
                "non-convex polyline "
                f"at x={x[bad:bad + 3]!r}, y={y[bad:bad + 3]!r}, "
                f"slopes={slopes[bad:bad + 2]!r}"
            )
    return x, y, slopes


@dataclass(frozen=True)
class ConvexPiecewiseLinear:
    """Proper convex polyline; its value is +inf outside the finite domain.

    A one-point domain is supported deliberately: Q1's terminal constraint is
    an indicator at 6000 kWh, not a large numerical penalty.
    """
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def _internal(cls, x, y):
        """Build a polyline from an operation that already preserves convexity.

        It avoids dataclass initialization overhead while retaining the same
        finite, monotonic, and convexity checks as public construction.
        """
        obj = object.__new__(cls)
        x, y, slopes = _validated_polyline_arrays(x, y)
        object.__setattr__(obj, "x", x)
        object.__setattr__(obj, "y", y)
        object.__setattr__(obj, "_slopes", slopes)
        return obj

    def __post_init__(self):
        x, y, slopes = _validated_polyline_arrays(self.x, self.y)

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "_slopes", slopes)

    @classmethod
    def constant(cls, value, lo=SOC_MIN, hi=SOC_MAX):
        if abs(hi - lo) <= _TOL:
            return cls(np.array([lo]), np.array([value]))
        return cls(np.array([lo, hi]), np.array([value, value]))

    @classmethod
    def point(cls, x, value=0.0):
        return cls(np.array([float(x)]), np.array([float(value)]))

    @property
    def lo(self):
        return float(self.x[0])

    @property
    def hi(self):
        return float(self.x[-1])

    @property
    def slopes(self):
        return self._slopes

    @property
    def breakpoints(self):
        return self.x

    @property
    def values(self):
        return self.y

    def __call__(self, z):
        a = np.asarray(z, float)
        out = np.full(a.shape, np.inf)
        inside = (a >= self.lo - _TOL) & (a <= self.hi + _TOL)
        if self.x.size == 1:
            out[np.abs(a - self.lo) <= _TOL] = self.y[0]
        else:
            out[inside] = np.interp(
                np.clip(a[inside], self.lo, self.hi), self.x, self.y
            )
        return float(out) if a.ndim == 0 else out

    def add_linear(self, slope, intercept=0.0):
        return type(self)._internal(
            self.x, self.y + float(slope) * self.x + float(intercept)
        )

    with_linear = add_linear

    def max_argmin(self, lo=SOC_MIN, hi=SOC_MAX):
        lo, hi = max(self.lo, lo), min(self.hi, hi)
        if lo > hi + _TOL:
            raise ValueError("empty minimisation domain")
        x = np.unique(np.r_[lo, self.x[(self.x > lo) & (self.x < hi)], hi])
        y = self(x)
        return float(x[np.flatnonzero(y<=y.min()+1e-8)[-1]])

    min_arg = max_argmin
    argmin = max_argmin

    def reflected(self):
        return type(self)._internal(-self.x[::-1], self.y[::-1])

@dataclass(frozen=True)
class DeterministicDPDispatch:
    """Forward-recovered Q1 actions from the independent continuous DP."""
    G: np.ndarray
    C: np.ndarray
    D: np.ndarray
    W: np.ndarray
    S: np.ndarray
    objective: float

def _segments(lo, value, slopes, lengths):
    # Recurrences call this with many short slope arrays.  Preallocate once
    # while retaining the original append/merge arithmetic exactly.
    slopes = np.asarray(slopes, float)
    lengths = np.asarray(lengths, float)
    x = np.empty(slopes.size + 1, float)
    y = np.empty(slopes.size + 1, float)
    x[0], y[0] = float(lo), float(value)
    count = 1
    for slope, length in zip(slopes, lengths):
        if length <= _TOL:
            continue
        previous_slope = ((y[count - 1] - y[count - 2]) /
                          (x[count - 1] - x[count - 2])) if count > 1 else None
        if previous_slope is not None and abs(slope - previous_slope) <= _TOL:
            x[count - 1] += length
            y[count - 1] += slope * length
        else:
            x[count] = x[count - 1] + length
            y[count] = y[count - 1] + slope * length
            count += 1
    return ConvexPiecewiseLinear._internal(x[:count], y[:count])

def inf_convolution(left, right):
    """Exact min-plus convolution: merge the two sorted slope multisets."""
    slopes = np.r_[left.slopes, right.slopes]
    lengths = np.r_[np.diff(left.x), np.diff(right.x)]
    order = np.argsort(slopes, kind="stable")
    return _segments(
        left.lo + right.lo, left.y[0] + right.y[0], slopes[order], lengths[order]
    )

def _restrict(fn, lo=SOC_MIN, hi=SOC_MAX):
    """Intersect a polyline's finite domain with physical SOC bounds."""
    lo, hi = max(float(lo), fn.lo), min(float(hi), fn.hi)
    if lo > hi + _TOL:
        raise ValueError("empty physical SOC domain")
    # Most DP steps are already inside physical bounds.  The polyline is
    # immutable and validated at construction, so avoid rebuilding it when
    # clipping leaves both endpoints unchanged.
    if abs(lo - fn.lo) <= _TOL and abs(hi - fn.hi) <= _TOL:
        return fn
    x = np.unique(np.r_[lo, fn.x[(fn.x > lo) & (fn.x < hi)], hi])
    return ConvexPiecewiseLinear._internal(x, fn(x))

def average_functions(fs):
    fs = list(fs)
    if not fs:
        return ConvexPiecewiseLinear.constant(0.0)
    lo, hi = max(f.lo for f in fs), min(f.hi for f in fs)
    if lo > hi + _TOL:
        raise ValueError("no common finite domain")
    first = fs[0]
    if all(np.array_equal(f.x, first.x) for f in fs[1:]):
        return ConvexPiecewiseLinear._internal(
            first.x, sum(f.y for f in fs) / len(fs)
        )
    x = np.unique(
        np.concatenate(([lo, hi], *[f.x[(f.x > lo) & (f.x < hi)] for f in fs]))
    )
    return ConvexPiecewiseLinear._internal(x, sum(f(x) for f in fs) / len(fs))

def _action_cost(price, net, lo, hi, multiplier=1.):
    points = [lo, hi]
    if lo < _TOL and hi > _TOL:
        points.append(0.0)
    kink = -net / ETA if net >= 0 else -ETA * net
    if lo + _TOL < kink < hi - _TOL:
        points.append(kink)
    x = np.array(sorted(set(points)))
    q = np.where(x >= 0, x / ETA, ETA * x)
    return ConvexPiecewiseLinear._internal(
        x, multiplier * float(price) * np.maximum(net + q, 0.0)
    )


def _reflected_action_arrays(price, net, lo, hi, multiplier=1.):
    """Return a checked action curve in the orientation used by convolution.

    A backward step used to materialise both ``action`` and
    ``action.reflected()`` as CPWL instances.  The reflection is an exact
    invariant-preserving rearrangement of an already checked polyline, so its
    object allocation and a second identical validation pass are unnecessary.
    Keep the action validation itself: in particular, a non-finite or
    non-convex action (for example from an invalid price) must fail before a
    slope merge could conceal it.
    """
    points = [lo, hi]
    if lo < _TOL and hi > _TOL:
        points.append(0.0)
    kink = -net / ETA if net >= 0 else -ETA * net
    if lo + _TOL < kink < hi - _TOL:
        points.append(kink)
    x = np.array(sorted(set(points)))
    q = np.where(x >= 0, x / ETA, ETA * x)
    x, y, _ = _validated_polyline_arrays(
        x, multiplier * float(price) * np.maximum(net + q, 0.0)
    )
    # Reflection preserves finite ordinates, strict breakpoint order, and
    # convexity.  Compute slopes in reflected coordinates rather than simply
    # reversing them so the arithmetic exactly matches CPWL.reflected().
    x, y = -x[::-1], y[::-1]
    return x, y, np.diff(y) / np.diff(x)


def _inf_convolution_verified_right(left, right_x, right_y, right_slopes):
    """Convolve ``left`` with validated right-hand CPWL arrays.

    This is intentionally private: callers may only supply arrays produced
    by ``_reflected_action_arrays``.  The resulting recurrence is still built
    through ``_segments`` and therefore receives the ordinary full CPWL
    validation before it can be clipped or exposed.
    """
    slopes = np.r_[left.slopes, right_slopes]
    lengths = np.r_[np.diff(left.x), np.diff(right_x)]
    order = np.argsort(slopes, kind="stable")
    return _segments(
        left.lo + float(right_x[0]), left.y[0] + right_y[0],
        slopes[order], lengths[order]
    )


def _free_step(nxt, price, net):
    x, y, slopes = _reflected_action_arrays(
        price, net, -P_MAX_KWH / ETA, ETA * P_MAX_KWH
    )
    return _restrict(_inf_convolution_verified_right(nxt, x, y, slopes))

def _surplus_step(nxt, surplus):
    # A surplus segment admits charge in [0, available].  Its zero-cost
    # infimal convolution is exact and convex; because future cost is
    # non-increasing, its argmin is the executor's "charge as much as can be
    # used" action, without baking a non-convex forced-action composition
    # into the backwards recurrence.
    inc = ETA * min(float(surplus), P_MAX_KWH)
    action_x = np.array([0.0, inc]) if inc > _TOL else np.array([0.0])
    action_y = np.zeros(action_x.size)
    action_x, action_y, _ = _validated_polyline_arrays(action_x, action_y)
    x, y = -action_x[::-1], action_y[::-1]
    return _restrict(_inf_convolution_verified_right(nxt, x, y, np.diff(y) / np.diff(x)))

def _fixed_step(nxt, price, residual):
    if residual <= 0:
        return _surplus_step(nxt, -residual)
    x, y, slopes = _reflected_action_arrays(
        5.0 * price, residual, -P_MAX_KWH / ETA, 0.0
    )
    return _restrict(_inf_convolution_verified_right(nxt, x, y, slopes))

def _terminal(terminal):
    if terminal is None:
        return ConvexPiecewiseLinear.constant(0.0)
    if isinstance(terminal, ConvexPiecewiseLinear):
        return terminal
    raise TypeError("terminal must be ConvexPiecewiseLinear; sampled callable terminals are not continuous DP")

def next_day_value(price,load,pv,*,terminal=None,grid=None):
    if grid is not None:
        raise ValueError("SOC grids are not part of continuous DP")
    price, load, pv = (np.asarray(a, float).ravel() for a in (price, load, pv))
    if not (price.size == load.size == pv.size):
        raise ValueError("curve lengths differ")
    value = _terminal(terminal)
    for t in range(price.size - 1, -1, -1):
        value = _free_step(value, price[t], load[t] - pv[t])
    return value

def periodic_value(price,load,pv,soc_init=6000.,*,grid=None):
    if grid is not None:raise ValueError("periodic_value no longer accepts a grid")
    return next_day_value(price,load,pv,terminal=ConvexPiecewiseLinear.point(soc_init))

def recover_periodic_actions(price, load, pv, soc_init=6000.) -> DeterministicDPDispatch:
    """Independently recover Q1's optimal continuous-DP action trajectory."""
    p,l,v=(np.asarray(a,float).ravel() for a in (price,load,pv))
    if not(p.size==l.size==v.size): raise ValueError("curve lengths differ")
    values=[None]*(p.size+1); values[-1]=ConvexPiecewiseLinear.point(soc_init)
    for t in range(p.size-1,-1,-1): values[t]=_free_step(values[t+1],p[t],l[t]-v[t])
    g=np.zeros(p.size); c=np.zeros(p.size); d=np.zeros(p.size); w=np.zeros(p.size); s=np.zeros(p.size); stock=float(soc_init)
    for t in range(p.size):
        net=l[t]-v[t]; lo=max(SOC_MIN,stock-P_MAX_KWH/ETA,values[t+1].lo); hi=min(SOC_MAX,stock+ETA*P_MAX_KWH,values[t+1].hi)
        kink=stock+(-net/ETA if net>=0 else -ETA*net)
        # Objective kinks are continuation breakpoints, the zero-purchase
        # kink, and delta=0 where psi changes efficiency slope.
        candidates=np.unique(np.r_[lo,hi,values[t+1].x[(values[t+1].x>lo)&(values[t+1].x<hi)], np.clip(kink,lo,hi), np.clip(stock,lo,hi)])
        x=candidates-stock; q=np.where(x>=0,x/ETA,ETA*x)
        cost=p[t]*np.maximum(net+q,0)+values[t+1](candidates)
        nxt=float(candidates[np.flatnonzero(cost<=cost.min()+1e-8)[-1]])
        delta=nxt-stock
        if delta>=0: c[t]=delta/ETA
        else: d[t]=-ETA*delta
        g[t]=max(net+(delta/ETA if delta>=0 else ETA*delta),0.)
        w[t]=max(g[t]+v[t]+d[t]-l[t]-c[t],0.)
        stock=nxt; s[t]=stock
    return DeterministicDPDispatch(g,c,d,w,s,float(p@g))

def future_cost(price,load_scen,pv_scen,g_plan,terminal_value=None,*,n_fixed=None,grid=None,return_paths=False):
    """Average pathwise future costs; remainder after ``n_fixed`` is free purchase."""
    if grid is not None:raise ValueError("SOC grids are not part of continuous DP")
    L,PV=np.atleast_2d(np.asarray(load_scen,float)),np.atleast_2d(np.asarray(pv_scen,float))
    if L.shape!=PV.shape:raise ValueError("load/PV scenario shape mismatch")
    p=np.asarray(price,float); p=np.tile(p[None,:],(L.shape[0],1)) if p.ndim==1 else p
    if p.shape!=L.shape:raise ValueError("price/scenario shape mismatch")
    G=np.asarray(g_plan,float).ravel(); fixed=G.size if n_fixed is None else int(n_fixed)
    if fixed<0 or fixed>L.shape[1] or G.size<fixed:raise ValueError("invalid fixed horizon")
    paths=[]
    for w in range(L.shape[0]):
        seq=[None]*(L.shape[1]+1); v=_terminal(terminal_value); seq[-1]=v
        for t in range(L.shape[1]-1,-1,-1):
            v=_fixed_step(v,p[w,t],L[w,t]-PV[w,t]-G[t]) if t<fixed else _free_step(v,p[w,t],L[w,t]-PV[w,t])
            seq[t]=v
        paths.append(seq)
    average=[average_functions([path[t] for path in paths]) for t in range(L.shape[1]+1)]
    return (average,paths) if return_paths else average

def reserve_level(h_next, price, *, lo=SOC_MIN, hi=SOC_MAX):
    """Return the rightmost reserve minimizer on the clipped finite domain."""
    lo, hi = max(h_next.lo, lo), min(h_next.hi, hi)
    if lo > hi + _TOL:
        raise ValueError("empty minimisation domain")
    x = np.unique(np.r_[lo, h_next.x[(h_next.x > lo) & (h_next.x < hi)], hi])
    values = h_next(x) + 5.0 * float(price) * ETA * x
    return float(x[np.flatnonzero(values <= values.min() + 1e-8)[-1]])

def evaluate_plan(g_plan,load_scen,pv_scen,price,soc_init,terminal_value=None,*,price_scen=None,base_plan=None,return_hbar=False,n_fixed=None):
    G=np.asarray(g_plan,float).ravel(); L=np.atleast_2d(np.asarray(load_scen,float)); PV=np.atleast_2d(np.asarray(pv_scen,float))
    ps=np.asarray(price if price_scen is None else price_scen,float); ps=np.tile(ps[None,:],(L.shape[0],1)) if ps.ndim==1 else ps
    hbar,paths=future_cost(ps,L,PV,G,terminal_value,n_fixed=n_fixed,return_paths=True); fixed=G.size if n_fixed is None else int(n_fixed); base=None if base_plan is None else np.asarray(base_plan,float); common=ps[:,:fixed].mean(axis=0); scores=[]
    for w in range(L.shape[0]):
        s=float(soc_init); emergency=0.
        for t in range(fixed):
            r=L[w,t]-PV[w,t]-G[t]
            if r<=0:s+=ETA*min(-r,P_MAX_KWH,(SOC_MAX-s)/ETA)
            else:
                d=min(r,P_MAX_KWH,ETA*max(s-reserve_level(hbar[t+1],ps[w,t]),0)); s-=d/ETA; emergency+=5.*ps[w,t]*(r-d)
        ordinary=common@G[:fixed] if base is None else common@np.minimum(base[:fixed],G[:fixed])+.5*common@np.maximum(base[:fixed]-G[:fixed],0)+1.5*common@np.maximum(G[:fixed]-base[:fixed],0)
        # The common H-bar determines causal actions, but each realised path
        # is scored by its own cross-midnight continuation K, preserving the
        # approved load/PV/price pairing rather than scoring with average H.
        scores.append(float(ordinary+emergency+paths[w][fixed](s)))
    score=float(np.mean(scores)); return (score,hbar) if return_hbar else score

__all__=["ConvexPiecewiseLinear","DeterministicDPDispatch","average_functions","inf_convolution","next_day_value","periodic_value","recover_periodic_actions","future_cost","reserve_level","evaluate_plan"]
