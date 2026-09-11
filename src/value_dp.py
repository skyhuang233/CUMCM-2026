"""储能的凸价值函数与轻量级 DP 执行支持。

实现采用精确的分段线性接口；递推在 SOC 断点上做单调插值，因而不会把
SOC 或充放电量离散成整数。该模块刻意保持纯 numpy，便于候选计划重评复用。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import ETA, P_MAX_KWH, SOC_MAX, SOC_MIN


@dataclass
class ConvexPiecewiseLinear:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, float).ravel(); self.y = np.asarray(self.y, float).ravel()
        if self.x.size != self.y.size or self.x.size < 2: raise ValueError("invalid breakpoints")
        if np.any(np.diff(self.x) <= 0): raise ValueError("x must increase")

    @classmethod
    def constant(cls, value: float, lo: float = SOC_MIN, hi: float = SOC_MAX):
        return cls(np.array([lo, hi]), np.array([value, value], float))

    def __call__(self, z):
        a = np.asarray(z, float)
        out = np.interp(a, self.x, self.y, left=np.inf, right=np.inf)
        return float(out) if a.ndim == 0 else out

    def value(self, z): return self(z)

    @property
    def slopes(self): return np.diff(self.y) / np.diff(self.x)

    @property
    def breakpoints(self): return self.x

    @property
    def values(self): return self.y

    def add_linear(self, slope: float, intercept: float = 0.0):
        return type(self)(self.x.copy(), self.y + slope*self.x + intercept)

    def with_linear(self, slope: float, intercept: float = 0.0):
        return self.add_linear(slope, intercept)

    def average(self, other): return average_functions([self, other])

    def min_arg(self, lo=SOC_MIN, hi=SOC_MAX):
        mask = (self.x >= lo-1e-9) & (self.x <= hi+1e-9)
        if not np.any(mask): return float(lo)
        vals = self.y[mask]; xs = self.x[mask]; m = np.nanmin(vals)
        return float(xs[np.flatnonzero(vals <= m+1e-8)[-1]])

    def max_argmin(self, lo=SOC_MIN, hi=SOC_MAX): return self.min_arg(lo, hi)

    def argmin(self, lo=SOC_MIN, hi=SOC_MAX): return self.min_arg(lo, hi)

    def inf_convolution(self, other, *, lo=SOC_MIN, hi=SOC_MAX):
        """在共同 SOC 定义域上计算下确界卷积的折线近似。

        两个输入均为凸折线时，最优点只需检查断点及端点；这里保留所有
        输入断点组成的网格并在线性插值上求最小值，接口上等价于精确的
        一维凸折线卷积，且适合递推中的小规模调用。
        """
        xs = np.unique(np.concatenate([self.x, other.x]))
        xs = xs[(xs >= lo - 1e-9) & (xs <= hi + 1e-9)]
        vals = []
        for z in xs:
            y = np.unique(np.concatenate([
                self.x[(self.x >= lo) & (self.x <= z)],
                z - other.x[(other.x >= lo) & (other.x <= z)],
            ]))
            y = y[(y >= lo - 1e-9) & (y <= hi + 1e-9)]
            vals.append(np.min(np.asarray(self(y)) + np.asarray(other(z - y))))
        return type(self)(xs, np.asarray(vals))

    def convolve(self, other, **kwargs):
        return self.inf_convolution(other, **kwargs)


def average_functions(fs):
    fs = list(fs)
    if not fs: return ConvexPiecewiseLinear.constant(0.0)
    xs = np.unique(np.concatenate([f.x for f in fs])); ys = np.zeros_like(xs)
    for f in fs: ys += np.asarray(f(xs), float)
    return ConvexPiecewiseLinear(xs, ys / len(fs))


def lower_envelope(fs):
    """多条折线逐点取最小值（用于调试/候选比较）。"""
    fs = list(fs)
    if not fs: return ConvexPiecewiseLinear.constant(0.0)
    xs = np.unique(np.concatenate([f.x for f in fs]))
    ys = np.min(np.vstack([np.asarray(f(xs), float) for f in fs]), axis=0)
    return ConvexPiecewiseLinear(xs, ys)


def _grid(n=193): return np.linspace(SOC_MIN, SOC_MAX, n)


def next_day_value(price, load, pv, *, terminal=None, grid=None):
    """储能次日价值函数 V(e)，终端价值默认为 0。"""
    price, load, pv = map(lambda a: np.asarray(a,float).ravel(), (price,load,pv))
    g = _grid() if grid is None else np.asarray(grid,float)
    v = np.zeros(g.size) if terminal is None else np.asarray(terminal(g),float)
    for t in range(len(price)-1, -1, -1):
        new = np.empty_like(v)
        n = load[t]-pv[t]
        for i,e in enumerate(g):
            best = np.inf
            lo_x = max(-P_MAX_KWH / ETA, SOC_MIN - e)
            hi_x = min(ETA * P_MAX_KWH, SOC_MAX - e)
            for x in np.linspace(lo_x, hi_x, 33):
                ee=e+x
                q=x/ETA if x>=0 else ETA*x
                best=min(best, price[t]*max(n+q,0)+float(np.interp(ee,g,v)))
            new[i]=best
        v=new
    return ConvexPiecewiseLinear(g,v)


def future_cost(price, load_scen, pv_scen, g_plan, terminal_value=None, *, grid=None):
    """反向计算每个场景的剩余紧急费用，并返回等权平均折线列表。"""
    p0=np.asarray(price,float)
    L=np.atleast_2d(load_scen); PV=np.atleast_2d(pv_scen)
    p = np.tile(p0[None, :], (L.shape[0], 1)) if p0.ndim == 1 else p0
    if p.shape[0] != L.shape[0] or p.shape[1] < L.shape[1]: raise ValueError("price/scenario shape mismatch")
    G=np.asarray(g_plan,float).ravel(); g=_grid() if grid is None else np.asarray(grid,float)
    if terminal_value is None: terminal_value = lambda z: np.zeros_like(np.asarray(z,float))
    out=[]
    for w in range(L.shape[0]):
        v=np.asarray(terminal_value(g),float); fs=[None]*(L.shape[1]+1)
        fs[-1]=ConvexPiecewiseLinear(g,v)
        for t in range(L.shape[1]-1,-1,-1):
            vals=[]; r=L[w,t]-PV[w,t]-G[t]
            for e in g:
                if r<=0:
                    c=min(-r,P_MAX_KWH,(SOC_MAX-e)/ETA); ee=e+ETA*c; val=float(np.interp(ee,g,v))
                else:
                    ylo=max(SOC_MIN, e-P_MAX_KWH/ETA)
                    ys=g[(g >= ylo-1e-9) & (g <= e+1e-9)]
                    if ys.size == 0: ys=np.array([ylo])
                    # 还要检查紧急购电恰好为零的折点 y=e-r/eta。
                    ys=np.unique(np.concatenate([ys, [np.clip(e-r/ETA, ylo, e)]]))
                    d=ETA*(e-ys)
                    cand=5*p[w,t]*np.maximum(r-d,0.0)+np.interp(ys,g,v)
                    val=float(np.min(cand))
                vals.append(val)
            v=np.asarray(vals); fs[t]=ConvexPiecewiseLinear(g,v)
        out.append(fs)
    return [average_functions([f[t] for f in out]) for t in range(L.shape[1]+1)]


def reserve_level(h_next, price, *, lo=SOC_MIN, hi=SOC_MAX):
    """缺口段的保留库存 argmin_z {5 p eta z + H(z)}。"""
    z=np.asarray(h_next.x,float); vals=5*float(price)*ETA*z+np.asarray(h_next(z),float)
    m=np.nanmin(vals[(z>=lo)&(z<=hi)]); return float(z[np.flatnonzero(vals<=m+1e-8)[-1]])


def evaluate_plan(g_plan, load_scen, pv_scen, price, soc_init, terminal_value=None, *, price_scen=None, base_plan=None):
    """按 DP 规则评估候选购电计划，返回平均（计划+紧急+终端）费用。"""
    p=np.asarray(price,float); G=np.asarray(g_plan,float); L=np.atleast_2d(load_scen); PV=np.atleast_2d(pv_scen)
    ps = np.asarray(price_scen,float) if price_scen is not None else np.tile(p[None, :], (L.shape[0], 1))
    if ps.ndim == 1: ps = np.tile(ps[None, :], (L.shape[0], 1))
    ps = ps[:, :G.size]
    tv = terminal_value or (lambda x: np.zeros_like(np.asarray(x,float)))
    hbar = future_cost(ps, L, PV, G, tv)
    base = None if base_plan is None else np.asarray(base_plan,float)
    plan_price = ps.mean(axis=0)
    total=[]
    for w in range(L.shape[0]):
        s=float(soc_init); em=0.0
        for t in range(len(G)):
            r=L[w,t]-PV[w,t]-G[t]
            if r<=0: c=min(-r,P_MAX_KWH,(SOC_MAX-s)/ETA); s+=ETA*c
            else:
                reserve=SOC_MIN if t+1 >= len(hbar) else reserve_level(hbar[t+1], ps[w,t])
                d=min(r,P_MAX_KWH,ETA*max(s-reserve,0)); s-=d/ETA; em+=5*ps[w,t]*(r-d)
        if base is None: cst=float(plan_price @ G)
        else: cst=float(plan_price @ np.minimum(base,G) + 0.5*plan_price @ np.maximum(base-G,0) + 1.5*plan_price @ np.maximum(G-base,0))
        total.append(cst+em+float(tv(np.array([s]))[0]))
    return float(np.mean(total))


__all__=["ConvexPiecewiseLinear","average_functions","lower_envelope","next_day_value","future_cost","reserve_level","evaluate_plan"]
