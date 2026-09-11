"""生成论文插图（PDF），输出到 plot/。

图 1 费用递进瀑布图；图 2 问 1 单日调度全景；图 3 问 3 指定日（3-20）调度全景；
图 4 问 3 四变体与各预报时刻边际价值；图 5 时域 / 场景数敏感性。
颜色：蓝 #2a78d6（购电/计划）、橙 #eb6834（紧急/调整）、青 #1baf7a（光伏/储能）、灰 #52514e（参考）。
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "plot"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#0b0b0b"

for cand in ["Songti SC", "PingFang SC", "STSong", "Arial Unicode MS"]:
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [cand, "Times New Roman"]
        break
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "pdf.fonttype": 42, "axes.unicode_minus": False,
})
HOURS = np.arange(1, 145) / 6.0


def hour_axis(ax, step=4):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, step))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, step)])


# ---------- 图 1 费用递进瀑布图 ----------
def fig_cascade():
    labels = ["问2\n日前承诺", "问3\n日内调整", "问4-3\n波动电价"]
    base = 1223
    vals = [1448, 1396, 1464]  # 万元
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.axhline(base, color=GRAY, lw=1.0, ls="--")
    ax.text(-0.45, base - 6, f"完美信息下界 {base}", ha="left", va="top", color=GRAY, fontsize=8)
    prev = base
    for i, v in enumerate(vals):
        d = v - prev
        bottom = prev if d >= 0 else v
        ax.bar(i, abs(d), bottom=bottom, color=ORANGE if d > 0 else AQUA, width=0.55)
        if i > 0:
            ax.plot([i - 1 + 0.275, i - 0.275], [prev, prev], color=GRAY, lw=0.8, ls=":")
        ytxt, va = (v + 6, "bottom") if d >= 0 else (v - 6, "top")
        ax.text(i, ytxt, f"{v}", ha="center", va=va, color=INK)
        ax.text(i, bottom + abs(d) / 2, f"{d:+d}\n({d / prev * 100:+.1f}%)", ha="center",
                va="center", color="white" if abs(d) > 60 else INK, fontsize=8)
        prev = v
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(1200, 1500)
    ax.set_ylabel("全年购电费用（万元）")
    ax.set_title("费用逐层递进（万元）：橙色增量、青色节省，百分比相对上一层")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cascade.pdf")
    plt.close(fig)


# ---------- 图 2 问 1 单日全景 ----------
def fig_q1():
    from src.data import SOC_INIT, load_attachment1
    from src.optimizer import solve_deterministic
    a = load_attachment1()
    r = solve_deterministic(a.price, a.load_kwh, a.pv_kwh, SOC_INIT, periodic=True)
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.3, 1]})
    ax = axes[0]
    ax.step(HOURS, a.price, where="post", color=GRAY)
    ax.set_ylabel("电价（元/kWh）")
    ax.set_title("附件 1 电价与负载/光伏")
    ax2 = axes[1]
    ax2.plot(HOURS, a.load_kwh * 6, color=INK, label="负载")
    ax2.plot(HOURS, a.pv_kwh * 6, color=AQUA, label="光伏")
    ax2.plot(HOURS, r.G * 6, color=BLUE, label="计划购电 $G_t$")
    ax2.set_ylabel("功率（kW）")
    ax2.legend(loc="upper left", frameon=False, ncol=3)
    ax2.set_title("问 1 最优购电：低价段购电充电，峰段由储能供电")
    ax3 = axes[2]
    ax3.plot(HOURS, r.S, color=AQUA)
    ax3.fill_between(HOURS, 1200, r.S, color=AQUA, alpha=0.15)
    ax3.axhline(1200, color=GRAY, lw=0.8, ls="--")
    ax3.axhline(10800, color=GRAY, lw=0.8, ls="--")
    ax3.set_ylabel("储电量（kWh）")
    ax3.set_ylim(0, 12000)
    ax3.set_title("储电量轨迹（0:00 = 24:00 = 6000 kWh）")
    hour_axis(ax3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_q1_day.pdf")
    plt.close(fig)


# ---------- 图 3 问 3 指定日全景 ----------
def fig_q3_day(day=date(2025, 3, 20)):
    from src.run_q3 import Params, load_bundle, run_period
    bundle = load_bundle()
    res = run_period(date(2025, 1, 1), day, Params(), bundle, record_from=day)
    dr = res.days[-1]
    L, PV = bundle.truth(day)
    price = bundle.att1.price
    G0, Ga, E, S = dr.G0, dr.Ga, dr.E, dr.S
    fig, axes = plt.subplots(4, 1, figsize=(6.2, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.8, 1.2, 1.1, 0.9]})
    ax = axes[0]
    ax.step(HOURS, price, where="post", color=GRAY)
    ax.set_ylabel("电价（元/kWh）")
    ax.set_title(f"{day.isoformat()}（春分）问 3 调度全景")
    ax = axes[1]
    ax.plot(HOURS, L * 6, color=INK, label="负载真值")
    ax.plot(HOURS, PV * 6, color=AQUA, label="光伏真值")
    ax.set_ylabel("功率（kW）")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax = axes[2]
    ax.plot(HOURS, G0 * 6, color=BLUE, ls="--", label="计划购电 $G^0$")
    ax.plot(HOURS, Ga * 6, color=BLUE, label="提交购电 $G^a$")
    ax.bar(HOURS - 1 / 12, E * 6, width=1 / 6, color=ORANGE, label="紧急购电 $E$")
    for h in (6, 12, 18):
        ax.axvline(h, color=GRAY, lw=0.8, ls=":")
    ax.set_ylabel("功率（kW）")
    ax.set_ylim(0, 11000)
    ax.legend(loc="upper center", frameon=False, ncol=3)
    ax.set_title("虚线为预报时刻 6:00 / 12:00 / 18:00；$G^a$ 仅在其后区间偏离 $G^0$")
    ax = axes[3]
    ax.plot(HOURS, S, color=AQUA)
    ax.fill_between(HOURS, 1200, S, color=AQUA, alpha=0.15)
    ax.axhline(1200, color=GRAY, lw=0.8, ls="--")
    ax.axhline(10800, color=GRAY, lw=0.8, ls="--")
    ax.set_ylim(0, 12000)
    ax.set_ylabel("储电量（kWh）")
    hour_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_q3_day.pdf")
    plt.close(fig)
    return dr


# ---------- 图 4 问 3 变体 ----------
def fig_q3_variants():
    names = ["仅 0:00", "+6:00", "+12:00", "+18:00"]
    total = np.array([14719213, 14321593, 14022531, 13959698]) / 1e4
    emerg = np.array([1714009, 1037621, 708256, 607583]) / 1e4
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.9))
    x = np.arange(4)
    a1.bar(x, emerg, color=ORANGE, width=0.6)
    for i, (e, v) in enumerate(zip(emerg, total)):
        a1.text(i, e + 3, f"紧急 {e:.0f}\n总费用 {v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a1.set_xticks(x)
    a1.set_xticklabels(names)
    a1.set_ylim(0, 215)
    a1.set_ylabel("全年紧急购电费（万元）")
    a1.set_title("逐步引入预报时刻")
    marg = -np.diff(total)
    a2.bar(np.arange(3), marg, color=AQUA, width=0.6)
    for i, v in enumerate(marg):
        a2.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    a2.set_xticks(range(3))
    a2.set_xticklabels(["6:00", "12:00", "18:00"])
    a2.set_ylabel("边际节省（万元）")
    a2.set_title("各预报时刻的边际价值")
    fig.tight_layout()
    fig.savefig(OUT / "fig_q3_variants.pdf")
    plt.close(fig)


# ---------- 图 5 敏感性 ----------
def fig_sensitivity():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.7))

    def rel_bars(ax, names, vals, base_idx, extra, title):
        base = vals[base_idx]
        rel = (np.asarray(vals) / base - 1) * 100
        colors = [ORANGE if r > 0.01 else (AQUA if r < -0.01 else BLUE) for r in rel]
        ax.bar(range(len(vals)), rel, color=colors, width=0.6)
        ax.axhline(0, color=GRAY, lw=0.8)
        for i, (r, v, e) in enumerate(zip(rel, vals, extra)):
            off = 0.15 if r >= 0 else -0.15
            ax.text(i, r + off, f"{r:+.3f}%\n{v / 1e4:.0f} 万元{e}" if abs(r) < 0.01 else f"{r:+.2f}%\n{v / 1e4:.0f} 万元{e}", ha="center",
                    va="bottom" if r >= 0 else "top", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names)
        ax.set_title(title)
        lim = max(abs(rel)) * 1.9 + 0.8
        ax.set_ylim(-lim if rel.min() < 0 else -0.8, lim)

    rel_bars(a1, ["24h", "48h（基准）", "72h"], [15199607, 14478531, 14479146], 1,
             ["\nSOC 1292", "\nSOC 7219", "\nSOC 7221"], "时域长度（问 2，全年）")
    a1.set_ylabel("相对基准的费用差（%）")
    rel_bars(a2, ["M=6", "M=12（基准）", "M=20"], [14530255, 14478531, 14428632], 1,
             ["", "", ""], "场景数（问 2，48h）")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_cascade(); print("fig_cascade")
    fig_q1(); print("fig_q1_day")
    fig_q3_variants(); print("fig_q3_variants")
    fig_sensitivity(); print("fig_sensitivity")
    dr = fig_q3_day(); print("fig_q3_day", "紧急 kWh", round(float(dr.E.sum()), 1))
