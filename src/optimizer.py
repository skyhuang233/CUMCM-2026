"""统一的购电决策线性规划（优化核）。

全项目唯一的 LP 构建入口。内部按段数 T 与变量块偏移参数化装配稀疏矩阵，
后续单元在同一套变量与约束上加入场景维度、紧急购电、调整分裂变量与 48h 时域时，
只需传入不同的 T 与 offsets 复用下面的装配函数。
"""

from __future__ import annotations

from collections import OrderedDict
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


BALANCE_SIGNS = {
    "G": 1.0,  # 购电量（问 3 的当天段用 G0 常数 + 分裂变量代替）
    "DP": 1.0,  # 问 3 增购 Δ⁺
    "DM": -1.0,  # 问 3 减购 Δ⁻
    "D": 1.0,
    "E": 1.0,
    "C": -1.0,
    "W": -1.0,
}


def balance_rows(
    builder: EqualityBuilder,
    n_seg: int,
    off: dict[str, int],
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
) -> list[int]:
    """能量平衡：G_t + PV_t + D_t (+ E_t) = L_t + C_t + W_t。

    off 需含 'G','C','D','W'；含 'E' 时把紧急购电量计入供给侧。问 3 的重优化用
    'DP','DM' 代替 'G'（$G^a = G^0 + \\Delta^+ - \\Delta^-$，$G^0$ 并入右端项）。
    每个块名的系数由 `BALANCE_SIGNS` 给出，不在其中的块名（如 'S'）被忽略。
    """
    rows = []
    names = [name for name in off if name in BALANCE_SIGNS]
    for t in range(n_seg):
        coeffs = {off[name] + t: BALANCE_SIGNS[name] for name in names}
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
        "DP": (0.0, None),  # 问 3 增购量
        "DM": (0.0, None),  # 问 3 减购量，上界 G0_t 在求解时按段填入
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
    """SAA / 重优化 LP 中与曲线数值无关的部分：稀疏矩阵、块偏移、边界。

    结构只取决于 (M, n_today, n_future, 第一阶段块名)，按天缓存复用，避免逐日重复装配。
    问 2 的第一阶段块是 ('G',)；问 3 重优化换成 ('DP','DM')，其余完全相同。
    """

    m: int
    n_today: int
    n_future: int
    n_h: int
    n_var: int
    first: dict[str, int]
    blocks: list[dict[str, int]]
    a_eq: sp.csr_matrix
    bounds: np.ndarray  # (n_var, 2) 的 [下界, 上界]
    rows_per_scen: int

    @property
    def g0_off(self) -> int:
        """Backward-compatible view; callers should use ``first['G']``."""
        return self.first.get("G", 0)


_SAA_CACHE: OrderedDict[tuple[int, int, int, tuple[str, ...]], SAAStructure] = OrderedDict()
_SAA_CACHE_MAXSIZE = 32


def _saa_structure(
    m: int, n_today: int, n_future: int, first_stage: tuple[str, ...] = ("G",)
) -> SAAStructure:
    """装配 SAA / 重优化 LP 的等式矩阵与边界（带缓存）。

    变量顺序：第一阶段块(各 n_today) ‖ 每个场景 [C, D, W, E, S](n_h) + G(n_future)。
    等式行顺序（每个场景连续 2*n_h 行）：当天平衡、次日平衡、SOC 链。
    """
    key = (m, n_today, n_future, first_stage)
    hit = _SAA_CACHE.get(key)
    if hit is not None:
        _SAA_CACHE.move_to_end(key)
        return hit

    n_h = n_today + n_future
    lengths = [("C", n_h), ("D", n_h), ("W", n_h), ("E", n_h), ("S", n_h), ("G", n_future)]
    first = {name: i * n_today for i, name in enumerate(first_stage)}
    cursor = len(first_stage) * n_today
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
    for name, off in first.items():
        _fill_bounds(lb, ub, name, off, n_today)
    for block in blocks:
        for name, ln in lengths:
            _fill_bounds(lb, ub, name, block[name], ln)

    builder = EqualityBuilder(n_var)
    zeros_today = np.zeros(n_today)
    zeros_future = np.zeros(n_future)
    for block in blocks:
        off_today = dict(first)
        off_today.update(
            {"C": block["C"], "D": block["D"], "W": block["W"], "E": block["E"]}
        )
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
        first=first,
        blocks=blocks,
        a_eq=a_eq,
        bounds=np.column_stack([lb, ub]),
        rows_per_scen=2 * n_h,
    )
    _SAA_CACHE[key] = struct
    _SAA_CACHE.move_to_end(key)
    # Keep the process-wide structural cache bounded while preserving the
    # existing fast path for repeated daily solves.
    while len(_SAA_CACHE) > _SAA_CACHE_MAXSIZE:
        _SAA_CACHE.pop(next(iter(_SAA_CACHE)))
    return struct


@dataclass
class SAAResult:
    """SAA 两阶段 LP 的解。G0 为冻结的当天计划购电量。"""

    G0: np.ndarray
    objective: float
    plan_cost: float
    expected_recourse_cost: float
    status: str = ""


def _scen_price(price_scen: np.ndarray | None, m: int, n_h: int) -> np.ndarray | None:
    """校验按场景的电价矩阵 (M, n_h)；None 原样返回（确定电价路径）。"""
    if price_scen is None:
        return None
    ps = np.atleast_2d(np.asarray(price_scen, dtype=float))
    assert ps.shape == (m, n_h), f"price_scen 应为 {(m, n_h)}，实际 {ps.shape}"
    return ps


def _fill_scenario_cost(
    cost: np.ndarray,
    struct: SAAStructure,
    price_h: np.ndarray,
    ps: np.ndarray | None,
    eps: float,
) -> None:
    """填入第二阶段各场景块的 sample-average 代价。

    `ps` 为 None 时全部场景共用 `price_h`（确定电价）；否则场景 ω 用自己的 $p_{\\omega}$。
    """
    m, n_h, n_future = struct.m, struct.n_h, struct.n_future
    for w, block in enumerate(struct.blocks):
        p_w = price_h if ps is None else ps[w]
        cost[block["E"] : block["E"] + n_h] = 5.0 * p_w / m
        cost[block["C"] : block["C"] + n_h] = eps / m
        cost[block["D"] : block["D"] + n_h] = eps / m
        cost[block["W"] : block["W"] + n_h] = eps / m
        if n_future:
            cost[block["G"] : block["G"] + n_future] = p_w[struct.n_today :] / m


def first_stage_price(
    price_h: np.ndarray, ps: np.ndarray | None, n_first: int
) -> np.ndarray:
    """第一阶段（跨场景共享的 $G^0$ 或 $\\Delta^\\pm$）的计价电价。

    确定电价时就是 `price_h[:n_first]`；按场景电价时取场景均价
    $\\bar p^s_t = \\frac1M\\sum_\\omega p_{\\omega,t}$——因为第一阶段变量跨场景共享，
    「按场景计费再取期望」与「按均价计一次」完全等价。
    """
    if ps is None:
        return np.asarray(price_h, dtype=float)[:n_first]
    return ps[:, :n_first].mean(axis=0)


def saa_cost_vector(
    struct: SAAStructure,
    price_h: np.ndarray,
    ps: np.ndarray | None,
    eps: float,
) -> np.ndarray:
    """问 2 的 SAA LP 完整代价向量（第一阶段 G0 + 各场景块）。"""
    cost = np.zeros(struct.n_var)
    n_today = struct.n_today
    g0_off = struct.first["G"]
    cost[g0_off : g0_off + n_today] = first_stage_price(price_h, ps, n_today)
    _fill_scenario_cost(cost, struct, price_h, ps, eps)
    return cost


def _solve_saa_core(
    price_h: np.ndarray,
    L_scen: np.ndarray,
    PV_scen: np.ndarray,
    L_future: np.ndarray,
    PV_future: np.ndarray,
    soc_init: float,
    eps: float,
    n_today: int,
    price_scen: np.ndarray | None = None,
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
    ps = _scen_price(price_scen, m, struct.n_h)
    cost = saa_cost_vector(struct, price_h, ps, eps)

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
    price_scen: np.ndarray | None = None,
) -> SAAResult:
    """0:00 的 SAA 两阶段购电 LP。

    第一阶段：当天 G0（跨场景共享）；第二阶段：每场景的 C, D, W, E, S 与次日购电 G。
    目标 = Σ p_t G0_t + (1/M) Σ_ω [Σ_今 5p E + Σ_次日 (pG + 5pE)] + eps Σ_{ω,t}(C+D+W)。

    `price_scen` 给出 (M, price_h.size) 的按场景电价（问 4 的波动电价）时：紧急购电与
    次日购电按各场景自己的 $p_{\\omega,t}$ 计价，第一阶段 $G^0$ 按场景均价
    $\\bar p^s_t = \\frac1M\\sum_\\omega p_{\\omega,t}$ 计价（$G^0$ 共享，与按场景计费取期望等价）。
    缺省 None 时完全退化为单一 `price_h` 的确定电价路径。

    点预测解由调用方显式使用 :func:`solve_saa_point` 求解；此函数只返回
    SAA 场景解，避免把场景均值误标为「残差为零」的点预测解。
    """
    x, struct, obj, status = _solve_saa_core(
        price_h, L_scen, PV_scen, L_future, PV_future, soc_init, eps, n_today, price_scen
    )
    price_h = np.asarray(price_h, dtype=float)
    g0_off = struct.first["G"]
    G0 = x[g0_off : g0_off + n_today]
    plan_price = first_stage_price(
        price_h, _scen_price(price_scen, struct.m, struct.n_h), n_today
    )
    plan_cost = float(plan_price @ G0)

    return SAAResult(
        G0=G0,
        objective=obj,
        plan_cost=plan_cost,
        expected_recourse_cost=obj - plan_cost,
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
    g0_off = struct.first["G"]
    G[:n_today] = x[g0_off : g0_off + n_today]
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

    问 4 的电价逐段刷新（当段真值 + 剩余段点预测 + 次日曲线），`solve` 的 `price_h`
    参数只重算代价向量，结构与边界不变；缺省 None 时用构造时的电价。
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
        self.eps = eps

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
        self._cost_scratch = cost.copy()

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

    def cost_for(self, price_h: np.ndarray) -> np.ndarray:
        """按新电价重算代价向量（写入内部缓冲，不改结构与边界）。"""
        price = np.asarray(price_h, dtype=float)
        assert price.size == self.n_h, f"电价长度应为 {self.n_h}，实际 {price.size}"
        off = self.off
        cost = self._cost_scratch
        cost[off["G"] : off["G"] + self.n_today] = 0.0
        cost[off["G"] + self.n_today : off["G"] + self.n_h] = price[self.n_today :]
        cost[off["E"] : off["E"] + self.n_h] = 5.0 * price
        return cost

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
        price_h: np.ndarray | None = None,
    ) -> RollingStep:
        """段 t 开始时求解：段 t 用真值，t+1..末用点预测，当天购电量固定为 g_today。

        `price_h` 给出该段可得的 48h 电价（当段真值 + 剩余段点预测）时按它计价，
        缺省沿用构造时的电价。
        """
        n_today, n_h = self.n_today, self.n_h
        off = self.off
        cost = self.cost if price_h is None else self.cost_for(price_h)

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
            cost,
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


# ---------------------------------------------------------------------------
# 问 3：预报时刻的重优化 LP（调整购电量）
# ---------------------------------------------------------------------------

ADJUST_UP = 1.5  # 增购单价系数
ADJUST_DOWN = 0.5  # 减购违约金系数


@dataclass
class ReadjustResult:
    """重优化 LP 的解。数组均为当天 144 段的全长向量，$t<t_0$ 的部分保持 $G^0$。"""

    Ga: np.ndarray
    dplus: np.ndarray
    dminus: np.ndarray
    objective: float  # 含常数项 Σ_{t≥t0} p_t G0_t
    adjust_cost: float  # 1.5 p·Δ⁺ − 0.5 p·Δ⁻（相对计划费的净增量）
    status: str = ""


def solve_readjust(
    t0: int,
    price48: np.ndarray,
    G0: np.ndarray,
    L_scen: np.ndarray,
    PV_scen: np.ndarray,
    L_next: np.ndarray,
    PV_next: np.ndarray,
    soc_init: float,
    eps: float = EPS,
    *,
    n_day: int = T,
    price_scen: np.ndarray | None = None,
) -> ReadjustResult:
    """预报时刻 $t_0$ 的重优化：在冻结的 $G^0$ 上决定调整量 $\\Delta^\\pm$。

    时域 = 当天剩余段 $t_0..n_{day}-1$ + 次日全部段；`price48` 传当天 + 次日的完整电价，
    内部切成 `price48[t0:n_day] ‖ price48[n_day:]`。`L_scen`/`PV_scen` 形状 (M, n_day−t0)。

    第一阶段（跨场景共享）：$\\Delta^+_t\\ge0$、$0\\le\\Delta^-_t\\le G^0_t$，
    $G^a_t = G^0_t + \\Delta^+_t - \\Delta^-_t$ 代入当天平衡行（系数 ±1，$-G^0_t$ 进右端项）。
    目标在 $\\Delta^+$ 上取 $1.5p_t$、在 $\\Delta^-$ 上取 $-0.5p_t$，与结算式一致；
    两者同段净成本 $+p_t>0$，故不会共存。第二阶段与问 2 的 SAA 完全相同。

    `price_scen` 形状与 `price48` 一致，为 (M, price48.size) 的按场景电价：$\\Delta^\\pm$
    跨场景共享，按场景均价计费；紧急购电与次日购电按各场景电价计费。缺省退化为确定电价。
    """
    t0 = int(t0)
    price48 = np.asarray(price48, dtype=float)
    G0 = np.asarray(G0, dtype=float)
    L_scen = np.atleast_2d(np.asarray(L_scen, dtype=float))
    PV_scen = np.atleast_2d(np.asarray(PV_scen, dtype=float))
    L_next = np.asarray(L_next, dtype=float).ravel()
    PV_next = np.asarray(PV_next, dtype=float).ravel()

    n_rem = n_day - t0
    n_future = price48.size - n_day
    m = L_scen.shape[0]
    assert G0.size == n_day and 0 <= t0 < n_day
    assert L_scen.shape == (m, n_rem) and PV_scen.shape == (m, n_rem)
    assert L_next.size == n_future and PV_next.size == n_future

    price_h = np.concatenate([price48[t0:n_day], price48[n_day:]])
    g0_rem = G0[t0:]
    struct = _saa_structure(m, n_rem, n_future, ("DP", "DM"))
    dp_off, dm_off = struct.first["DP"], struct.first["DM"]

    ps48 = _scen_price(price_scen, m, price48.size)
    ps = None if ps48 is None else np.concatenate(
        [ps48[:, t0:n_day], ps48[:, n_day:]], axis=1
    )
    price_first = first_stage_price(price_h, ps, n_rem)

    cost = np.zeros(struct.n_var)
    cost[dp_off : dp_off + n_rem] = ADJUST_UP * price_first
    cost[dm_off : dm_off + n_rem] = -ADJUST_DOWN * price_first
    _fill_scenario_cost(cost, struct, price_h, ps, eps)

    bounds = struct.bounds.copy()
    bounds[dm_off : dm_off + n_rem, 1] = np.maximum(g0_rem, 0.0)  # Δ⁻_t ≤ G0_t ⇒ G^a ≥ 0

    net_future = L_next - PV_next
    soc_rhs = np.zeros(struct.n_h)
    soc_rhs[0] = float(soc_init)
    b_eq = np.empty(m * struct.rows_per_scen)
    for w in range(m):
        base = w * struct.rows_per_scen
        b_eq[base : base + n_rem] = L_scen[w] - PV_scen[w] - g0_rem
        b_eq[base + n_rem : base + struct.n_h] = net_future
        b_eq[base + struct.n_h : base + struct.rows_per_scen] = soc_rhs

    res = linprog(cost, A_eq=struct.a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"重优化 LP 求解失败（段 {t0}）：{res.message}")

    dplus = np.zeros(n_day)
    dminus = np.zeros(n_day)
    dplus[t0:] = np.maximum(res.x[dp_off : dp_off + n_rem], 0.0)
    dminus[t0:] = np.clip(res.x[dm_off : dm_off + n_rem], 0.0, np.maximum(g0_rem, 0.0))
    Ga = np.maximum(G0 + dplus - dminus, 0.0)
    price_day = np.empty(n_day)
    price_day[:t0] = price48[:t0]  # $t<t_0$ 段无调整量，取何电价都不影响费用
    price_day[t0:] = price_first[:n_rem]
    adjust_cost = float(
        ADJUST_UP * price_day @ dplus - ADJUST_DOWN * price_day @ dminus
    )
    return ReadjustResult(
        Ga=Ga,
        dplus=dplus,
        dminus=dminus,
        objective=float(res.fun) + float(price_first[:n_rem] @ g0_rem),
        adjust_cost=adjust_cost,
        status=res.message,
    )
