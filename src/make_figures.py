"""论文插图 F1–F13：只读 results/latest/m30 + results/latest/paper_aux + 附件，输出 PDF 矢量到 plot/。"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

from src import data

ROOT = Path(__file__).resolve().parents[1]
M30 = ROOT / "results/latest/m30"
AUX = ROOT / "results/latest/paper_aux"
OUT = ROOT / "plot"

# 统一低饱和度五色
C_BLUE = "#5b8db8"   # 主方法 / 计划购电量
C_GREEN = "#6aa88f"  # 光伏 / 储电量
C_RED = "#c0604d"    # 紧急购电 / 增费
C_PURPLE = "#8a7fae" # 次要分支 / 对照
C_GRAY = "#6e6c66"   # 真值 / 中性
PALETTE = [C_BLUE, C_GREEN, C_RED, C_PURPLE, C_GRAY]

for family in ("Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Songti SC"):
    if family in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = family
        break
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e8e7e2",
    "grid.linewidth": 0.6, "pdf.fonttype": 42, "axes.unicode_minus": False,
})

HOURS = np.arange(1, data.T + 1) / 6  # 每段末端小时


def read_json(path):
    return json.loads(path.read_text())


def read_summary(name):
    return read_json(M30 / f"{name}.summary.json")


def read_trace(name):
    with np.load(M30 / f"{name}.trace.npz") as z:
        return {k: z[k].copy() for k in z.files}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{name}.pdf"


def _box(ax, x, y, w, h, text, fc, fontsize=7.5, ec=C_GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=fc, edgecolor=ec, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, xy_from, xy_to, color=C_GRAY, lw=0.9, style="-|>", rad=0.0):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5,
                                connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- F1 框架图
def fig_f01():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    col_x = [1.7, 4.35, 7.0, 9.65]   # 每列左边界
    col_w = 2.45
    row_y = {"pred": 4.55, "opt": 2.75, "exec": 0.95}
    row_h = 1.35
    row_fc = {"pred": "#eaf1ee", "opt": "#e9eef5", "exec": "#f2eef5"}

    heads = ["问 1  确定性", "问 2  随机场景", "问 3  多预报时点", "问 4  波动电价"]
    infos = [r"$F_t(e)$", r"$\bar H_t(e)$", r"$\bar H_{t|\tau}(e)$", r"$\bar H_{t|\tau}(e;c)$"]
    cells = {
        "pred": ["附件 1 点预测曲线\n（无不确定性）",
                 "分解式负载预测\n配对残差场景 $M{=}30$",
                 "＋正式光伏预报\n预报时刻 0/6/12/18",
                 "＋电价预测\n联合价格场景"],
        "opt": ["确定性购电 LP\n价值标尺 $F_t(e)$",
                "SAA 两阶段代理模型\n候选计划重评 → 冻结 $G^0$",
                "滚动重优化\n提交调整购电量 $G^a$",
                "价格场景 SAA\n4-2 重订 $G^0$／4-3 重订 $G^a$"],
        "exec": ["DP 逐段执行\nLP–DP 互证",
                 "DP 价值执行器\n$\\bar H_t(e)\\to$ 动态保留水平 $R_t$",
                 "差额结算\n紧急购电按 5 倍电价",
                 "波动电价结算\n完美电价对照"],
    }
    # 左侧三层标签
    for key, label in [("pred", "预测／场景层"), ("opt", "采购优化层"), ("exec", "执行结算层")]:
        _box(ax, 0.12, row_y[key], 1.25, row_h, label, "#f5f4f0", fontsize=8)
    # 列头 + 信息状态递进链
    for j, x in enumerate(col_x):
        ax.text(x + col_w / 2, 7.25, heads[j], ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + col_w / 2, 6.62, infos[j], ha="center", va="center", fontsize=9, color=C_BLUE)
        if j < 3:
            _arrow(ax, (x + col_w + 0.06, 6.62), (col_x[j + 1] - 0.06, 6.62), color=C_BLUE, lw=1.1)
    ax.text(0.12, 6.62, "信息状态\n逐级扩展", ha="left", va="center", fontsize=7.5, color=C_BLUE)
    # 单元格与箭头
    for key in cells:
        for j, x in enumerate(col_x):
            _box(ax, x, row_y[key], col_w, row_h, cells[key][j], row_fc[key], fontsize=7.2)
    for j, x in enumerate(col_x):
        xc = x + col_w / 2
        _arrow(ax, (xc, row_y["pred"]), (xc, row_y["opt"] + row_h))       # 预测 → 优化
        _arrow(ax, (xc, row_y["opt"]), (xc, row_y["exec"] + row_h))       # 优化 → 执行
    ax.text(col_x[0] + col_w / 2 + 0.14, (row_y["pred"] + row_y["opt"] + row_h) / 2,
            "预测曲线／场景", ha="left", va="center", fontsize=6.8, color=C_GRAY)
    ax.text(col_x[0] + col_w / 2 + 0.14, (row_y["opt"] + row_y["exec"] + row_h) / 2,
            "购电量固定", ha="left", va="center", fontsize=6.8, color=C_GRAY)
    # 执行层反馈：真实储电量回到下一决策的信息状态
    _arrow(ax, (col_x[1] + col_w / 2, row_y["exec"]), (col_x[1] + col_w / 2, 0.32), lw=0.8)
    ax.text(col_x[1] + col_w / 2 + 0.12, 0.34, "真实账单与库存结转 → 次日 0:00 的信息状态",
            ha="left", va="center", fontsize=7.2, color=C_GRAY)
    # 横向递进箭头（问 i 优化层 → 问 i+1 优化层）
    for j in range(3):
        _arrow(ax, (col_x[j] + col_w, row_y["opt"] + row_h / 2),
               (col_x[j + 1], row_y["opt"] + row_h / 2), color=C_PURPLE, lw=0.9)
    return save(fig, "fig_f01_framework")


# ---------------------------------------------------- F2 价值函数与边际价值
def fig_f02():
    vf = read_json(AUX / "q1_value_functions.json")["picks"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    styles = {"0": ("-", C_BLUE), "36": ("--", C_GREEN), "72": ("-", C_RED), "108": (":", C_PURPLE)}
    names = {"0": "0:00", "36": "6:00", "72": "12:00", "108": "18:00"}
    for t in ["0", "36", "72", "108"]:
        p = vf[t]
        axes[0].plot(p["x"], np.asarray(p["y"]) / 1e4, styles[t][0], color=styles[t][1],
                     lw=1.3, label=f"$t=${names[t]}")
    axes[0].set_xlabel("段初储电量 $e$（kWh）")
    axes[0].set_ylabel("剩余最低费用 $F_t(e)$（万元）")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0].set_title("(a) 确定性价值函数（凸折线）", fontsize=8.5)

    p = vf["72"]
    axes[1].stairs(-np.asarray(p["slopes"]), p["x"], color=C_RED, lw=1.4, baseline=None)
    axes[1].set_xlabel("段初储电量 $e$（kWh）")
    axes[1].set_ylabel(r"边际价值 $-\partial F_t/\partial e$（元/kWh）")
    axes[1].set_title("(b) $t=$12:00 的边际价值（非增阶梯）", fontsize=8.5)
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    return save(fig, "fig_f02_value_function")


# ------------------------------------------------------------ F3 问1 典型日
def fig_f03():
    with np.load(M30 / "q1.trace.npz") as z:
        t1 = {k: z[k].copy() for k in z.files}
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.6), sharex=True)
    axes[0].step(HOURS, t1["price"], where="pre", color=C_GRAY, lw=1.1)
    axes[0].set_ylabel("电价（元/kWh）")
    axes[0].set_title("(a) 附件 1 电价", fontsize=8.5, loc="left")

    axes[1].plot(HOURS, t1["load"] * 6, color=C_GRAY, lw=1.2, label="负载")
    axes[1].plot(HOURS, t1["pv"] * 6, color=C_GREEN, lw=1.2, label="光伏")
    axes[1].set_ylabel("功率（kW）")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].set_title("(b) 负载与光伏点预测", fontsize=8.5, loc="left")

    axes[2].step(HOURS, t1["G"] * 6, where="pre", color=C_BLUE, lw=1.2, label="购电量")
    axes[2].set_ylabel("购电功率（kW）")
    ax2 = axes[2].twinx()
    ax2.plot(np.r_[0, HOURS], np.r_[data.SOC_INIT, t1["S"]], color=C_GREEN, lw=1.2, label="储电量")
    ax2.set_ylabel("储电量（kWh）")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, frameon=False, ncol=2, loc="lower right",
                   bbox_to_anchor=(1.0, 1.0))
    axes[2].set_title("(c) 计划购电与储电量轨迹", fontsize=8.5, loc="left")
    axes[2].set_xlabel("时刻（h）")
    axes[2].set_xlim(0, 24)
    axes[2].set_xticks(range(0, 25, 4))
    fig.tight_layout()
    return save(fig, "fig_f03_q1_day")


# ---------------------------------------------------------- F4 负载预测效果
def fig_f04():
    with np.load(AUX / "forecast_day_2025-03-20.npz") as z:
        pred, true = z["load_pred"] * 6, z["load_true"] * 6
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    axes[0].plot(HOURS, true, color=C_GRAY, lw=1.4, label="负载真值")
    axes[0].plot(HOURS, pred, color=C_BLUE, lw=1.2, ls="--", label="分解式负载预测（0:00 发布）")
    axes[0].set_ylabel("负载功率（kW）")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].set_title("2025-03-20", fontsize=8.5, loc="left")
    resid = pred - true
    axes[1].fill_between(HOURS, resid, 0, color=C_RED, alpha=0.35, step=None)
    axes[1].axhline(0, color=C_GRAY, lw=0.7)
    axes[1].set_ylabel("残差（kW）")
    axes[1].set_xlabel("时刻（h）")
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(range(0, 25, 4))
    mae = np.mean(np.abs(resid))
    axes[1].text(0.99, 0.92, f"预测 − 真值，MAE = {mae:.1f} kW", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=7.5, color=C_GRAY)
    fig.tight_layout()
    return save(fig, "fig_f04_load_forecast")


# ------------------------------------------------------------ F5 场景扇形
def fig_f05():
    with np.load(AUX / "forecast_day_2025-03-20.npz") as z:
        scen = (z["L_scen"] - z["PV_scen"]) * 6
        true = (z["load_true"] - z["pv_true"]) * 6
        point = (z["load_pred"] - z["pv_pred"]) * 6
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    lo, hi = np.percentile(scen, [10, 90], axis=0)
    ax.fill_between(HOURS, lo, hi, color=C_BLUE, alpha=0.18, label="场景 10–90 分位带")
    for s in scen:
        ax.plot(HOURS, s, color=C_BLUE, lw=0.35, alpha=0.28)
    ax.plot([], [], color=C_BLUE, lw=0.6, alpha=0.6, label="配对残差场景（$M=30$）")
    ax.plot(HOURS, point, color=C_PURPLE, lw=1.1, ls="--", label="点预测净负荷")
    ax.plot(HOURS, true, color=C_GRAY, lw=1.6, label="净负荷真值")
    ax.set_xlabel("时刻（h）")
    ax.set_ylabel("净负荷（负载 − 光伏，kW）")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title("2025-03-20", fontsize=8.5, loc="left")
    fig.tight_layout()
    return save(fig, "fig_f05_scenario_fan")


# ---------------------------------------------------------- F6 DP 执行机理
def fig_f06():
    with np.load(AUX / "forecast_day_2025-03-20.npz") as z:
        R, S, E = z["R"].copy(), z["S"].copy(), z["E"].copy()
    hbar = read_json(AUX / "hbar_day_2025-03-20.json")["picks"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2))

    ax = axes[0]
    ax.plot(np.r_[0, HOURS], np.r_[data.SOC_INIT, S], color=C_GREEN, lw=1.4, label="储电量 $S_t$")
    ax.plot(HOURS, R, color=C_PURPLE, lw=1.2, ls="--", label="动态保留水平 $R_t$（仅缺口段）")
    ax.axhline(data.SOC_MIN, color=C_GRAY, lw=0.6, ls=":")
    ax.text(23.8, data.SOC_MIN + 120, "库存下限 1200", ha="right", fontsize=7, color=C_GRAY)
    ax.set_ylabel("库存（kWh）")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax2 = ax.twinx()
    ax2.bar(HOURS - 1 / 12, E * 6, width=1 / 6, color=C_RED, alpha=0.85, label="紧急购电量 $E_t$")
    ax2.set_ylabel("紧急购电功率（kW）", color=C_RED)
    ax2.tick_params(axis="y", colors=C_RED)
    ax2.set_ylim(0, max(E.max() * 6 * 3.2, 1))
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    ax.set_title("(a) 2025-03-20：库存高于保留水平的部分才用于当段缺口，其余留给更贵的未来缺口",
                 fontsize=8, loc="left", pad=26)
    ax.set_xlabel("时刻（h）")

    ax = axes[1]
    for t, ls, color in [("36", "-", C_BLUE), ("72", "--", C_GREEN), ("108", ":", C_PURPLE)]:
        p = hbar[t]
        ax.plot(p["x"], np.asarray(p["y"]) / 1e4, ls, color=color, lw=1.3,
                label=f"$t=${int(t) // 6}:00")
    ax.set_xlabel("段初储电量 $e$（kWh）")
    ax.set_ylabel(r"未来费用函数 $\bar H_t(e)$（万元）")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("(b) 场景均值未来费用函数：斜率决定每一 kWh 库存留到未来的价值", fontsize=8, loc="left")
    fig.tight_layout()
    return save(fig, "fig_f06_dp_mechanism")


# ---------------------------------------------------------- F7 问2 月度费用
def fig_f07():
    s = read_summary("q2_candidates")
    months = sorted(s["monthly"])
    vals = np.array([s["monthly"][m]["total_cost"] for m in months]) / 1e4
    lb = read_json(AUX / "perfect_lowerbound.json")["lower_bound_const"] / 1e4 / len(months)
    ns = read_json(AUX / "nostorage_baseline.json")["totals"]["perfect_const"] / 1e4 / len(months)
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    x = np.arange(len(months))
    ax.bar(x, vals, width=0.62, color=C_BLUE, label="问 2 主方法（SAA＋DP 价值执行器）")
    ax.axhline(ns, color=C_GRAY, lw=1.2, ls="--", label=f"无储能直购基线（真值净负荷，月均 {ns:.1f}）")
    ax.axhline(lb, color=C_RED, lw=1.2, ls=":", label=f"完美信息离线下界（全年均摊，月均 {lb:.1f}）")
    for i, v in enumerate(vals):
        ax.text(i, v - 3, f"{v:.1f}", ha="center", va="top", fontsize=6.8, color="white")
    ax.set_xticks(x, [f"{int(m[-2:])}月" for m in months])
    ax.set_ylabel("月度实际费用（万元）")
    ax.set_ylim(0, max(vals.max(), ns) * 1.22)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1)
    fig.tight_layout()
    return save(fig, "fig_f07_q2_monthly")


# ---------------------------------------------------- F8 滚动重优化时间线
def fig_f08():
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    issues = [0, 6, 12, 18]
    ax.set_xlim(-0.3, 48.5)
    ax.set_ylim(-0.9, len(issues) + 0.4)
    for k, tau in enumerate(issues):
        y = len(issues) - 1 - k
        if tau > 0:  # 已执行段
            ax.add_patch(Rectangle((0, y), tau, 0.52, facecolor="#d8d6d0", edgecolor="none"))
        commit_end = tau + 6 if tau < 18 else 24
        ax.add_patch(Rectangle((tau, y), commit_end - tau, 0.52,
                               facecolor=C_BLUE, edgecolor="none", alpha=0.9))
        if commit_end < 24:  # 临时决策段（当天剩余，可被覆盖）
            ax.add_patch(Rectangle((commit_end, y), 24 - commit_end, 0.52,
                                   facecolor=C_BLUE, alpha=0.22, edgecolor=C_BLUE,
                                   linewidth=0.7, linestyle="--"))
        if tau > 0:  # 次日暂拟购电（跨午夜前瞻）
            ax.add_patch(Rectangle((24, y), tau, 0.52, facecolor=C_PURPLE, alpha=0.16,
                                   edgecolor=C_PURPLE, linewidth=0.7, linestyle=":"))
        # 24h 前瞻窗
        ax.annotate("", xy=(tau + 24, y + 0.78), xytext=(tau, y + 0.78),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.18,widthB=0.18", color=C_GRAY, lw=0.7))
        ax.text(tau + 12, y + 0.86, "24h 前瞻窗", ha="center", fontsize=6.5, color=C_GRAY)
        ax.text(-0.6, y + 0.26, f"{tau}:00", ha="right", va="center", fontsize=8)
        ax.plot([tau, tau], [y - 0.08, y + 0.6], color=C_RED, lw=1.0)
    ax.axvline(24, color=C_GRAY, lw=0.8, ls="--")
    ax.text(24, len(issues) + 0.28, "午夜（次日 0:00 重新制定计划）", ha="center", fontsize=7, color=C_GRAY)
    ax.set_xticks(range(0, 49, 6))
    ax.set_xlabel("自当天 0:00 起的小时")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    handles = [Rectangle((0, 0), 1, 1, facecolor="#d8d6d0"),
               Rectangle((0, 0), 1, 1, facecolor=C_BLUE, alpha=0.9),
               Rectangle((0, 0), 1, 1, facecolor=C_BLUE, alpha=0.22, edgecolor=C_BLUE, linestyle="--"),
               Rectangle((0, 0), 1, 1, facecolor=C_PURPLE, alpha=0.16, edgecolor=C_PURPLE, linestyle=":"),
               plt.Line2D([0], [0], color=C_RED, lw=1.0)]
    labels = ["已执行段", "本轮提交（0:00 为 $G^0$，其余为 $G^a$）", "临时决策段（可被覆盖）", "次日暂拟购电", "预报时刻"]
    ax.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.42), fontsize=7)
    fig.tight_layout()
    return save(fig, "fig_f08_rolling_timeline")


# ---------------------------------------------- F9 预报时刻边际贡献
def fig_f09():
    names = ["q3_0", "q3_06", "q3_0612", "q3_baseline"]
    costs = np.array([read_summary(n)["total_cost"] for n in names]) / 1e4
    savings = costs[:-1] - costs[1:]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(3)
    bars = ax.bar(x, savings, width=0.55, color=[C_GREEN, C_BLUE, C_GREEN])
    for i, v in enumerate(savings):
        ax.text(i, v + 0.5, f"+{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, ["新增 6:00", "新增 12:00", "新增 18:00"])
    ax.set_ylabel("相邻组合的全年费用节省（万元）")
    ax.set_ylim(0, savings.max() * 1.22)
    ax.text(0.02, 0.96, f"仅 0:00 时全年 {costs[0]:.2f} 万元\n四预报时刻齐备 {costs[-1]:.2f} 万元",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=C_GRAY)
    fig.tight_layout()
    return save(fig, "fig_f09_issue_marginal"), savings, costs


# ---------------------------------------------------- F10 价格预测效果
def fig_f10():
    days = read_json(AUX / "price_forecast_days.json")["days"]
    picks = ["2025-03-20", "2025-12-21"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, d, tag in zip(axes, picks, ["(a)", "(b)"]):
        rec = days[d]
        ax.step(HOURS, rec["truth"], where="pre", color=C_GRAY, lw=1.3, label="附件 4 电价真值")
        ax.step(HOURS, rec["pred"], where="pre", color=C_BLUE, lw=1.1, ls="--", label="0:00 发布的电价预测")
        ax.set_xlabel("时刻（h）")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.set_title(f"{tag} {d}　MAE = {rec['mae']:.3f} 元/kWh", fontsize=8.5, loc="left")
    axes[0].set_ylabel("电价（元/kWh）")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig_f10_price_forecast")


# ---------------------------------------------------- F11 问4 分支月度
def fig_f11():
    q3 = read_summary("q3_baseline")["monthly"]
    q42 = read_summary("q4_2_candidates")
    q43 = read_summary("q4_3_baseline")
    assert abs(q42["total_cost"] / 1e4 - 1427.06) < 0.01, q42["total_cost"]
    assert abs(q43["total_cost"] / 1e4 - 1371.70) < 0.01, q43["total_cost"]
    months = sorted(q3)
    d42 = np.array([q42["monthly"][m]["total_cost"] - q3[m]["total_cost"] for m in months]) / 1e4
    d43 = np.array([q43["monthly"][m]["total_cost"] - q3[m]["total_cost"] for m in months]) / 1e4
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    x = np.arange(len(months))
    ax.bar(x - 0.19, d42, width=0.36, color=C_PURPLE, label=f"问 4-2（全年 +{(q42['total_cost'] - read_summary('q3_baseline')['total_cost']) / 1e4:.1f} 万元）")
    ax.bar(x + 0.19, d43, width=0.36, color=C_BLUE, label=f"问 4-3（全年 +{(q43['total_cost'] - read_summary('q3_baseline')['total_cost']) / 1e4:.1f} 万元）")
    ax.axhline(0, color=C_GRAY, lw=0.8)
    ax.set_xticks(x, [f"{int(m[-2:])}月" for m in months])
    ax.set_ylabel("相对问 3 的月度费用变化（万元）")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    return save(fig, "fig_f11_q4_monthly"), d42, d43


# ---------------------------------------------- F12 指定日执行轨迹
def fig_f12():
    t3 = read_trace("q3_baseline")
    day = "2025-03-20"
    i = list(t3["dates"]).index(day)
    att2 = data.load_attachment2(str(ROOT / "data/附件2.xlsx"))
    j = [d.isoformat() for d in att2.dates].index(day)
    net = (att2.load_kwh[j] - att2.pv_kwh[j]) * 6
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(HOURS, net, color=C_GRAY, lw=1.3, label="净负荷真值")
    ax.step(HOURS, t3["Ga"][i] * 6, where="pre", color=C_BLUE, lw=1.2, label="调整购电量 $G^a$")
    ax.bar(HOURS - 1 / 12, t3["E"][i] * 6, width=1 / 6, color=C_RED, label="紧急购电量 $E$")
    ax.set_ylabel("功率（kW）")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.set_title(f"{day}（问 3 主链）", fontsize=8.5, loc="left", pad=24)
    ax = axes[1]
    ax.plot(np.r_[0, HOURS], np.r_[t3["S"][i - 1, -1], t3["S"][i]], color=C_GREEN, lw=1.3, label="储电量 $S_t$")
    ax.plot(HOURS, t3["R"][i], color=C_PURPLE, lw=1.0, ls="--", label="动态保留水平 $R_t$")
    ax.axhline(data.SOC_MIN, color=C_GRAY, lw=0.6, ls=":")
    ax.set_ylabel("库存（kWh）")
    ax.set_xlabel("时刻（h）")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout()
    return save(fig, "fig_f12_exec_day")


# ------------------------------------------------------ F13 灵敏度汇总
def fig_f13():
    rows = read_json(M30 / "ablation_comparison.json")
    branches = ["q2", "q3", "q4_2", "q4_3"]
    branch_names = {"q2": "问 2", "q3": "问 3", "q4_2": "问 4-2", "q4_3": "问 4-3"}
    variants = ["48h", "candidates", "same_type", "legacy_means", "legacy_price"]
    variant_names = {"48h": "48h 时域＋次日价值", "candidates": "候选计划重评",
                     "same_type": "同类型日场景池", "legacy_means": "旧均值预测",
                     "legacy_price": "旧价格预测"}
    variant_colors = {"48h": C_BLUE, "candidates": C_GREEN, "same_type": C_PURPLE,
                      "legacy_means": C_GRAY, "legacy_price": C_RED}
    lookup = {(r["branch"], r["variant"]): r["delta_yuan"] / 1e4 for r in rows}
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    group_w = 0.8
    for bi, b in enumerate(branches):
        avail = [v for v in variants if (b, v) in lookup]
        w = group_w / len(avail)
        for vi, v in enumerate(avail):
            xpos = bi - group_w / 2 + w * (vi + 0.5)
            val = lookup[(b, v)]
            ax.bar(xpos, val, width=w * 0.9, color=variant_colors[v],
                   label=variant_names[v] if bi == (0 if v != "legacy_price" else 2) else None)
            if abs(val) > 0.4:
                ax.text(xpos, val + (0.5 if val >= 0 else -0.5), f"{val:+.1f}",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=6.2, color=C_GRAY)
    vals_all = list(lookup.values())
    ax.set_ylim(min(vals_all) - 3.2, max(vals_all) + 3.2)
    ax.axhline(0, color=C_GRAY, lw=0.8)
    ax.set_xticks(range(4), [branch_names[b] for b in branches])
    ax.set_ylabel("相对分支基线的费用变化（万元）")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24), fontsize=7)
    ax.text(0.01, 0.97, "负值＝该组件单独启用后全年实际费用下降", transform=ax.transAxes,
            va="top", fontsize=7, color=C_GRAY)
    fig.tight_layout()
    return save(fig, "fig_f13_sensitivity")


def main():
    OUT.mkdir(exist_ok=True)
    made = []
    made.append(fig_f01())
    made.append(fig_f02())
    made.append(fig_f03())
    made.append(fig_f04())
    made.append(fig_f05())
    made.append(fig_f06())
    made.append(fig_f07())
    made.append(fig_f08())
    name, savings, costs = fig_f09()
    made.append(name)
    expect = np.array([3.83, 23.74, 5.09])
    assert np.allclose(savings, expect, atol=0.02), f"问3边际节省与锚定数字不符: {savings}"
    made.append(fig_f10())
    name, d42, d43 = fig_f11()
    made.append(name)
    made.append(fig_f12())
    made.append(fig_f13())
    print("生成插图：")
    for n in made:
        print(f"  plot/{n}")
    print(f"问3 相邻节省（万元）: {np.round(savings, 2).tolist()}，全套费用 {np.round(costs, 2).tolist()}")
    print(f"问4-2 相对问3 月度差范围（万元）: [{d42.min():.2f}, {d42.max():.2f}]，负值月 {int((d42 < 0).sum())} 个")
    print(f"问4-3 相对问3 月度差范围（万元）: [{d43.min():.2f}, {d43.max():.2f}]，负值月 {int((d43 < 0).sum())} 个")


if __name__ == "__main__":
    main()
