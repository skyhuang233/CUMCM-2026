"""统一的购电决策线性规划（优化核）。

全项目唯一的 LP 构建入口。内部按段数 T 与变量块偏移参数化装配稀疏矩阵，
后续单元在同一套变量与约束上加入场景维度、紧急购电、调整分裂变量与 48h 时域时，
只需传入不同的 T 与 offsets 复用下面的装配函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .data import EPS, ETA, P_MAX_KWH, SOC_INIT, SOC_MAX, SOC_MIN, T


@dataclass
class VarLayout:
    """变量向量的分块布局：块名 → 起始偏移，块长统一为 length。"""

    offsets: dict[str, int]
    length: int
    n_var: int

    def slice(self, name: str) -> slice:
        off = self.offsets[name]
        return slice(off, off + self.length)

    def take(self, x: np.ndarray, name: str) -> np.ndarray:
        return x[self.slice(name)]

    def block(self, name: str) -> dict[str, int]:
        """单块的偏移视图，供装配函数按 offsets[name] 定位列。"""
        return {name: self.offsets[name]}


def make_layout(names: list[str], length: int, start: int = 0) -> VarLayout:
    """按给定顺序为每个块分配长度为 length 的连续列区间。"""
    offsets = {name: start + i * length for i, name in enumerate(names)}
    return VarLayout(offsets=offsets, length=length, n_var=start + len(names) * length)


class EqualityBuilder:
    """按行装配等式约束的稀疏矩阵（COO 三元组累积，最后转 CSR）。"""

    def __init__(self, n_var: int):
        self.n_var = n_var
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []
        self._rhs: list[float] = []

    def add_row(self, coeffs: dict[int, float], rhs: float) -> int:
        r = len(self._rhs)
        for col, val in coeffs.items():
            self._rows.append(r)
            self._cols.append(col)
            self._vals.append(val)
        self._rhs.append(rhs)
        return r

    @property
    def n_row(self) -> int:
        return len(self._rhs)

    def build(self) -> tuple[sp.csr_matrix, np.ndarray]:
        mat = sp.coo_matrix(
            (self._vals, (self._rows, self._cols)), shape=(self.n_row, self.n_var)
        )
        return mat.tocsr(), np.asarray(self._rhs, dtype=float)


def balance_rows(
    builder: EqualityBuilder,
    n_seg: int,
    off: dict[str, int],
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
) -> list[int]:
    """能量平衡：G_t + PV_t + D_t (+ E_t) = L_t + C_t + W_t。

    off 需含 'G','C','D','W'；含 'E' 时把紧急购电量计入供给侧。
    """
    rows = []
    for t in range(n_seg):
        coeffs = {
            off["G"] + t: 1.0,
            off["D"] + t: 1.0,
            off["C"] + t: -1.0,
            off["W"] + t: -1.0,
        }
        if "E" in off:
            coeffs[off["E"] + t] = 1.0
        rows.append(builder.add_row(coeffs, float(load_kwh[t] - pv_kwh[t])))
    return rows


def soc_rows(
    builder: EqualityBuilder,
    n_seg: int,
    off: dict[str, int],
    soc_init: float,
    eta: float = ETA,
) -> list[int]:
    """储电量动态：S_t - S_{t-1} - eta*C_t + D_t/eta = 0，S_{-1} = soc_init。"""
    rows = []
    for t in range(n_seg):
        coeffs = {
            off["S"] + t: 1.0,
            off["C"] + t: -eta,
            off["D"] + t: 1.0 / eta,
        }
        if t == 0:
            rhs = soc_init
        else:
            coeffs[off["S"] + t - 1] = -1.0
            rhs = 0.0
        rows.append(builder.add_row(coeffs, rhs))
    return rows


def terminal_soc_row(
    builder: EqualityBuilder, n_seg: int, off: dict[str, int], soc_end: float
) -> int:
    """末段储电量约束：S_{n_seg-1} = soc_end（问 1 的周期约束即 soc_end = S_0）。"""
    return builder.add_row({off["S"] + n_seg - 1: 1.0}, float(soc_end))


def block_bounds(
    n_seg: int,
    p_max: float = P_MAX_KWH,
    soc_min: float = SOC_MIN,
    soc_max: float = SOC_MAX,
) -> dict[str, tuple[float, float | None]]:
    """各变量块的逐元素上下界。"""
    return {
        "G": (0.0, None),
        "E": (0.0, None),
        "C": (0.0, p_max),
        "D": (0.0, p_max),
        "W": (0.0, None),
        "S": (soc_min, soc_max),
    }


def stack_bounds(layout: VarLayout, spec: dict[str, tuple[float, float | None]]) -> list:
    """按布局展开成 linprog 需要的逐变量 bounds 列表。"""
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * layout.n_var
    for name, off in layout.offsets.items():
        for t in range(layout.length):
            bounds[off + t] = spec[name]
    return bounds


@dataclass
class LPResult:
    """优化核的结构化解。所有电量单位为 kWh。"""

    G: np.ndarray
    C: np.ndarray
    D: np.ndarray
    W: np.ndarray
    S: np.ndarray
    E: np.ndarray
    objective: float
    purchase_cost: float
    duals: dict[str, np.ndarray | float] = field(default_factory=dict)
    status: str = ""


def solve_deterministic(
    price: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    soc_init: float = SOC_INIT,
    *,
    periodic: bool = True,
    soc_end: float | None = None,
    eps: float = EPS,
) -> LPResult:
    """点预测曲线下的确定性购电 LP。

    periodic=True 时约束末段储电量回到 soc_init；soc_end 给出则改用该值。
    目标 = Σ p_t G_t + eps·Σ(C_t + D_t + W_t)，报告的购电费不含 eps 项。
    """
    price = np.asarray(price, dtype=float)
    load_kwh = np.asarray(load_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    n_seg = price.size
    assert load_kwh.size == n_seg and pv_kwh.size == n_seg

    layout = make_layout(["G", "C", "D", "W", "S"], n_seg)
    off = layout.offsets

    cost = np.zeros(layout.n_var)
    cost[layout.slice("G")] = price
    for name in ("C", "D", "W"):
        cost[layout.slice(name)] = eps

    builder = EqualityBuilder(layout.n_var)
    rows_balance = balance_rows(builder, n_seg, off, load_kwh, pv_kwh)
    rows_soc = soc_rows(builder, n_seg, off, soc_init)
    row_periodic = None
    if periodic or soc_end is not None:
        target = soc_init if soc_end is None else soc_end
        row_periodic = terminal_soc_row(builder, n_seg, off, target)
    a_eq, b_eq = builder.build()

    res = linprog(
        cost,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=stack_bounds(layout, block_bounds(n_seg)),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP 求解失败：{res.message}")

    x = res.x
    marginals = np.asarray(res.eqlin.marginals, dtype=float)
    duals: dict[str, np.ndarray | float] = {
        "balance": marginals[rows_balance],
        "soc": marginals[rows_soc],
    }
    if row_periodic is not None:
        duals["periodic"] = float(marginals[row_periodic])

    G = layout.take(x, "G")
    return LPResult(
        G=G,
        C=layout.take(x, "C"),
        D=layout.take(x, "D"),
        W=layout.take(x, "W"),
        S=layout.take(x, "S"),
        E=np.zeros(n_seg),
        objective=float(res.fun),
        purchase_cost=float(price @ G),
        duals=duals,
        status=res.message,
    )


# ---------------------------------------------------------------------------
# 问 2：SAA 两阶段购电 LP 与储能滚动 LP
# ---------------------------------------------------------------------------

BLOCK_BOUNDS = block_bounds(0)  # 与段数无关，仅按块名给出上下界


def _fill_bounds(lb: np.ndarray, ub: np.ndarray, name: str, off: int, n: int) -> None:
    lo, hi = BLOCK_BOUNDS[name]
    lb[off : off + n] = lo
    ub[off : off + n] = np.inf if hi is None else hi


@dataclass
class SAAStructure:
    """SAA LP 中与曲线数值无关的部分：稀疏矩阵、块偏移、边界。

    结构只取决于 (M, n_today, n_future)，按天缓存复用，避免逐日重复装配。
    """

    m: int
    n_today: int
    n_future: int
    n_h: int
    n_var: int
    g0_off: int
    blocks: list[dict[str, int]]
    a_eq: sp.csr_matrix
    bounds: np.ndarray  # (n_var, 2) 的 [下界, 上界]
    rows_per_scen: int


_SAA_CACHE: dict[tuple[int, int, int], SAAStructure] = {}


def _saa_structure(m: int, n_today: int, n_future: int) -> SAAStructure:
    """装配 SAA LP 的等式矩阵与边界（带缓存）。

    变量顺序：G0(n_today) ‖ 每个场景 [C, D, W, E, S](n_h) + G(n_future)。
    等式行顺序（每个场景连续 2*n_h 行）：当天平衡、次日平衡、SOC 链。
    """
    key = (m, n_today, n_future)
    hit = _SAA_CACHE.get(key)
    if hit is not None:
        return hit

    n_h = n_today + n_future
    lengths = [("C", n_h), ("D", n_h), ("W", n_h), ("E", n_h), ("S", n_h), ("G", n_future)]
    g0_off = 0
    cursor = n_today
    blocks: list[dict[str, int]] = []
    for _ in range(m):
        block = {}
        for name, ln in lengths:
            block[name] = cursor
            cursor += ln
        blocks.append(block)
    n_var = cursor

    lb = np.zeros(n_var)
    ub = np.full(n_var, np.inf)
    _fill_bounds(lb, ub, "G", g0_off, n_today)
    for block in blocks:
        for name, ln in lengths:
            _fill_bounds(lb, ub, name, block[name], ln)

    builder = EqualityBuilder(n_var)
    zeros_today = np.zeros(n_today)
    zeros_future = np.zeros(n_future)
    for block in blocks:
        off_today = {
            "G": g0_off,
            "C": block["C"],
            "D": block["D"],
            "W": block["W"],
            "E": block["E"],
        }
        balance_rows(builder, n_today, off_today, zeros_today, zeros_today)
        if n_future:
            off_next = {
                "G": block["G"],
                "C": block["C"] + n_today,
                "D": block["D"] + n_today,
                "W": block["W"] + n_today,
                "E": block["E"] + n_today,
            }
            balance_rows(builder, n_future, off_next, zeros_future, zeros_future)
        soc_rows(builder, n_h, {"S": block["S"], "C": block["C"], "D": block["D"]}, 0.0)
    a_eq, _ = builder.build()

    struct = SAAStructure(
        m=m,
        n_today=n_today,
        n_future=n_future,
        n_h=n_h,
        n_var=n_var,
        g0_off=g0_off,
        blocks=blocks,
        a_eq=a_eq,
        bounds=np.column_stack([lb, ub]),
        rows_per_scen=2 * n_h,
    )
    _SAA_CACHE[key] = struct
    return struct


@dataclass
class SAAResult:
    """SAA 两阶段 LP 的解。G0 为冻结的当天计划购电量。"""

    G0: np.ndarray
    objective: float
    plan_cost: float
    expected_recourse_cost: float
    point_solution: LPResult | None = None
    status: str = ""


def _solve_saa_core(
    price_h: np.ndarray,
    L_scen: np.ndarray,
    PV_scen: np.ndarray,
    L_future: np.ndarray,
    PV_future: np.ndarray,
    soc_init: float,
    eps: float,
    n_today: int,
) -> tuple[np.ndarray, SAAStructure, float, str]:
    price_h = np.asarray(price_h, dtype=float)
    L_scen = np.atleast_2d(np.asarray(L_scen, dtype=float))
    PV_scen = np.atleast_2d(np.asarray(PV_scen, dtype=float))
    L_future = np.asarray(L_future, dtype=float).ravel()
    PV_future = np.asarray(PV_future, dtype=float).ravel()

    m = L_scen.shape[0]
    n_future = price_h.size - n_today
    assert L_scen.shape == (m, n_today) and PV_scen.shape == (m, n_today)
    assert L_future.size == n_future and PV_future.size == n_future
    assert PV_scen.shape[0] == m

    struct = _saa_structure(m, n_today, n_future)

    cost = np.zeros(struct.n_var)
    cost[struct.g0_off : struct.g0_off + n_today] = price_h[:n_today]
    for block in struct.blocks:
        cost[block["E"] : block["E"] + struct.n_h] = 5.0 * price_h / m
        cost[block["C"] : block["C"] + struct.n_h] = eps
        cost[block["D"] : block["D"] + struct.n_h] = eps
        cost[block["W"] : block["W"] + struct.n_h] = eps
        if n_future:
            cost[block["G"] : block["G"] + n_future] = price_h[n_today:] / m

    net_future = L_future - PV_future
    soc_rhs = np.zeros(struct.n_h)
    soc_rhs[0] = float(soc_init)
    b_eq = np.empty(m * struct.rows_per_scen)
    for w in range(m):
        base = w * struct.rows_per_scen
        b_eq[base : base + n_today] = L_scen[w] - PV_scen[w]
        b_eq[base + n_today : base + struct.n_h] = net_future
        b_eq[base + struct.n_h : base + struct.rows_per_scen] = soc_rhs

    res = linprog(
        cost,
        A_eq=struct.a_eq,
        b_eq=b_eq,
        bounds=struct.bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"SAA LP 求解失败：{res.message}")
    return res.x, struct, float(res.fun), res.message


def solve_saa(
    price_h: np.ndarray,
    L_scen: np.ndarray,
    PV_scen: np.ndarray,
    L_future: np.ndarray,
    PV_future: np.ndarray,
    soc_init: float = SOC_INIT,
    eps: float = EPS,
    *,
    n_today: int = T,
    with_point: bool = False,
) -> SAAResult:
    """0:00 的 SAA 两阶段购电 LP。

    第一阶段：当天 G0（跨场景共享）；第二阶段：每场景的 C, D, W, E, S 与次日购电 G。
    目标 = Σ p_t G0_t + (1/M) Σ_ω [Σ_今 5p E + Σ_次日 (pG + 5pE)] + eps Σ_{ω,t}(C+D+W)。

    `with_point=True` 时另用点预测曲线（M=1、残差为零）解一次同结构 LP，
    返回其充放电与储电量轨迹作为「点预测解」，仅供论文对照，不进入执行。
    """
    x, struct, obj, status = _solve_saa_core(
        price_h, L_scen, PV_scen, L_future, PV_future, soc_init, eps, n_today
    )
    price_h = np.asarray(price_h, dtype=float)
    G0 = x[struct.g0_off : struct.g0_off + n_today]
    plan_cost = float(price_h[:n_today] @ G0)

    point = None
    if with_point:
        L0 = np.atleast_2d(np.asarray(L_scen, dtype=float)).mean(axis=0)
        PV0 = np.atleast_2d(np.asarray(PV_scen, dtype=float)).mean(axis=0)
        point = solve_saa_point(
            price_h, L0, PV0, L_future, PV_future, soc_init, eps, n_today=n_today
        )

    return SAAResult(
        G0=G0,
        objective=obj,
        plan_cost=plan_cost,
        expected_recourse_cost=obj - plan_cost,
        point_solution=point,
        status=status,
    )


def solve_saa_point(
    price_h: np.ndarray,
    L0: np.ndarray,
    PV0: np.ndarray,
    L_future: np.ndarray,
    PV_future: np.ndarray,
    soc_init: float = SOC_INIT,
    eps: float = EPS,
    *,
    n_today: int = T,
) -> LPResult:
    """点预测曲线下的 48h 确定性 LP（M=1 的 SAA），返回完整轨迹。"""
    price_h = np.asarray(price_h, dtype=float)
    L0 = np.asarray(L0, dtype=float).ravel()
    PV0 = np.asarray(PV0, dtype=float).ravel()
    x, struct, obj, status = _solve_saa_core(
        price_h, L0[None, :], PV0[None, :], L_future, PV_future, soc_init, eps, n_today
    )
    block = struct.blocks[0]
    n_h, n_future = struct.n_h, struct.n_future
    G = np.empty(n_h)
    G[:n_today] = x[struct.g0_off : struct.g0_off + n_today]
    if n_future:
        G[n_today:] = x[block["G"] : block["G"] + n_future]
    take = lambda name: x[block[name] : block[name] + n_h]  # noqa: E731
    return LPResult(
        G=G,
        C=take("C"),
        D=take("D"),
        W=take("W"),
        S=take("S"),
        E=take("E"),
        objective=obj,
        purchase_cost=float(price_h @ G),
        duals={},
        status=status,
    )


@dataclass
class RollingStep:
    """储能滚动 LP 的一次求解结果（全时域轨迹，只有首段会被执行）。"""

    C: np.ndarray
    D: np.ndarray
    W: np.ndarray
    E: np.ndarray
    S: np.ndarray
    G: np.ndarray
    objective: float


class StorageRollingSolver:
    """固定购电量下的储能滚动控制 LP。

    时域固定为 n_today + n_future 段，等式矩阵、代价向量、基础边界只装配一次；
    每段只更新右端项与边界：段 τ < t 的全部变量被上下界钉成 0（等价于把时域缩短到
    t..末段），SOC 链通过第 0 行的右端项从当前储电量起算。
    """

    def __init__(
        self,
        price_h: np.ndarray,
        *,
        n_today: int = T,
        eps: float = EPS,
        eta: float = ETA,
    ):
        self.price = np.asarray(price_h, dtype=float)
        self.n_today = int(n_today)
        self.n_h = self.price.size
        self.n_future = self.n_h - self.n_today
        assert self.n_future >= 0
        self.eta = eta

        self.layout = make_layout(["G", "C", "D", "W", "E", "S"], self.n_h)
        off = self.layout.offsets
        self.off = off
        n_var = self.layout.n_var

        cost = np.zeros(n_var)
        # 当天购电量被钉死为 G0，其费用是常数，这里置 0，目标只留紧急购电与次日购电。
        cost[off["G"] + self.n_today : off["G"] + self.n_h] = self.price[self.n_today :]
        cost[off["E"] : off["E"] + self.n_h] = 5.0 * self.price
        for name in ("C", "D", "W"):
            cost[off[name] : off[name] + self.n_h] = eps
        self.cost = cost

        lb = np.zeros(n_var)
        ub = np.full(n_var, np.inf)
        for name in ("G", "C", "D", "W", "E", "S"):
            _fill_bounds(lb, ub, name, off[name], self.n_h)
        self._bounds0 = np.column_stack([lb, ub])

        builder = EqualityBuilder(n_var)
        zeros = np.zeros(self.n_h)
        self._rows_balance = balance_rows(builder, self.n_h, off, zeros, zeros)
        self._rows_soc = soc_rows(builder, self.n_h, off, 0.0, eta=eta)
        self.a_eq, _ = builder.build()
        self._b_eq = np.zeros(builder.n_row)
        self._net = np.zeros(self.n_h)

    def solve(
        self,
        t: int,
        soc_prev: float,
        load_true_t: float,
        pv_true_t: float,
        load_pred_today: np.ndarray,
        pv_pred_today: np.ndarray,
        load_future: np.ndarray,
        pv_future: np.ndarray,
        g_today: np.ndarray,
    ) -> RollingStep:
        """段 t 开始时求解：段 t 用真值，t+1..末用点预测，当天购电量固定为 g_today。"""
        n_today, n_h = self.n_today, self.n_h
        off = self.off

        net = self._net
        net[:t] = 0.0
        net[t] = float(load_true_t) - float(pv_true_t)
        if t + 1 < n_today:
            net[t + 1 : n_today] = (
                np.asarray(load_pred_today[t + 1 : n_today], dtype=float)
                - np.asarray(pv_pred_today[t + 1 : n_today], dtype=float)
            )
        if self.n_future:
            net[n_today:] = (
                np.asarray(load_future, dtype=float) - np.asarray(pv_future, dtype=float)
            )
        b_eq = self._b_eq
        b_eq[:n_h] = net
        b_eq[n_h] = float(soc_prev)
        b_eq[n_h + 1 :] = 0.0

        bounds = self._bounds0.copy()
        for name in ("G", "C", "D", "W", "E"):
            base = off[name]
            bounds[base : base + t, :] = 0.0
        g = np.asarray(g_today, dtype=float)[t:n_today]
        bounds[off["G"] + t : off["G"] + n_today, 0] = g
        bounds[off["G"] + t : off["G"] + n_today, 1] = g

        res = linprog(
            self.cost,
            A_eq=self.a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"储能滚动 LP 求解失败（段 {t}）：{res.message}")
        x = res.x
        take = lambda name: x[off[name] : off[name] + n_h]  # noqa: E731
        return RollingStep(
            C=take("C"),
            D=take("D"),
            W=take("W"),
            E=take("E"),
            S=take("S"),
            G=take("G"),
            objective=float(res.fun),
        )
