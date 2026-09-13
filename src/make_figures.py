"""Reproducible paper figures from frozen results; run python -m src.make_figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from src import data
ROOT = Path(__file__).resolve().parents[1]
M30 = ROOT / "results/latest/m30"
AUX = ROOT / "results/latest/paper_aux"
OUT = ROOT / "plot"
C_BLUE, C_GREEN, C_RED = "#244F73", "#298C82", "#B95541"
C_PURPLE, C_GRAY = "#81709C", "#747C83"
PALETTE = [C_BLUE, C_GREEN, C_RED, C_PURPLE]
STYLES = ["-", "--", "-.", ":"]
DAYS = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
BRANCHES = ["q2", "q3", "q4_2", "q4_3"]
BRANCH_LABELS = ["问 2", "问 3", "问 4-2", "问 4-3"]
VARIANTS = ["48h", "candidates", "same_type", "legacy_means", "legacy_price"]
VARIANT_LABELS = ["启用 48h 与次日价值", "启用候选计划重评", "改用同类型日场景池", "改用旧均值负载预测", "改用旧价格预测"]
HOURS = np.arange(1, data.T + 1) / 6
plt.rcParams.update({
    "font.family": "Songti SC", "font.size": 10, "axes.titlesize": 10,
    "legend.fontsize": 9, "axes.labelsize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#E5E9EC",
    "grid.linewidth": .55, "lines.linewidth": 1.5, "pdf.fonttype": 42,
    "axes.unicode_minus": False, "figure.constrained_layout.h_pad": .08,
    "figure.constrained_layout.w_pad": .06, "figure.constrained_layout.hspace": .12,
})


def read_json(path):
    return json.loads(path.read_text())


def read_summary(name):
    return read_json(M30 / f"{name}.summary.json")


def read_trace(name):
    with np.load(M30 / f"{name}.trace.npz") as z:
        return {k: z[k].copy() for k in z.files}


def save(fig, name):
    # One legend in its own reserved strip, never over a data axis.
    items = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                items.setdefault(label, handle)
    if items:
        fig.legend(items.values(), items.keys(), loc="outside upper center",
                   ncol=min(4, len(items)), frameon=False, handlelength=2.7,
                   columnspacing=1.4)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    return name


def panels(rows=1, cols=1, height=3, **kw):
    return plt.subplots(rows, cols, figsize=(6.8, height), layout="constrained", **kw)


def time_axis(ax):
    ax.set(xlim=(0, 24), xticks=np.arange(0, 25, 4), xlabel="时刻（h）")


def bounds(ax):
    for y in [data.SOC_MIN, data.SOC_MAX]:
        ax.axhline(y, color=C_GRAY, lw=.7, ls=":")
    ax.set(ylim=(0, 12000), yticks=[1200, 6000, 10800], ylabel="储电量（kWh）")


def soc(ax, trace, i, label="储电量", color=C_GREEN, ls="-"):
    initial = trace["S"][i-1, -1] if i else data.SOC_INIT
    ax.plot(np.r_[0, HOURS], np.r_[initial, trace["S"][i]], color=color, ls=ls, label=label)
    ax.scatter([0, 24], [initial, trace["S"][i, -1]], s=14, color=color, zorder=4)
    bounds(ax)


def _box(ax, x, y, w, h, text, fc, fontsize=7.5, ec=C_GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06", facecolor=fc, edgecolor=ec, linewidth=.8))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, xy_from, xy_to, color=C_GRAY, lw=.9, style="-|>", rad=0):
    ax.annotate("", xy=xy_to, xytext=xy_from, arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=1.5, shrinkB=1.5, connectionstyle=f"arc3,rad={rad}"))


def fig_f01():
    # The overview figure is deliberately a modular architecture rather than a
    # table: each question has the same input -> optimization -> execution
    # grammar, while the arrows show the information/value-function progression.
    fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained")
    ax.set(xlim=(0, 12.6), ylim=(0, 10.0)); ax.axis("off")

    # Unified kernel banner.
    _box(ax, .45, 8.72, 11.7, .7,
         "信息状态 $\\mathcal{I}_t$ 约束的统一优化核：库存 $e_t$  →  未来费用函数  →  购电决策与因果执行",
         "#E7EEF3", fontsize=9.2, ec=C_BLUE)
    ax.text(6.3, 9.67, "四问的预测—采购—执行总体框架", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#33434F")

    # Four question modules in the style of the reference architecture figures.
    modules = [
        ("问 1  ·  确定性", .45, 4.82, "#F2EAF4", "#81709C",
         "完整信息输入\n附件 1 曲线：负载、光伏、电价",
         "确定性线性规划\n全天购电计划 + 周期库存",
         "$F_t(e)$：凸折线价值\nDP 与 LP 最优值互证"),
        ("问 2  ·  随机场景", 6.55, 4.82, "#EAF3EF", "#298C82",
         "预测与场景输入\n分解式负载预测 + 配对残差",
         "样本平均近似\n0:00 冻结计划购电量",
         "$\\bar H_t(e)$：动态保留\n因果充放电执行"),
        ("问 3  ·  滚动调整", .45, .92, "#EAF0F6", "#244F73",
         "信息逐级发布\n正式光伏预报：0 / 6 / 12 / 18 时",
         "滚动重优化\n冻结计划 + 日内调整权限",
         "$\\bar H_{t\\mid\\tau}(e)$：\n调整、紧急费与下一轮更新"),
        ("问 4  ·  波动电价", 6.55, .92, "#F8EEE8", "#B95541",
         "新增价格场景\n水平 × 日内形状，三通道配对",
         "两分支价格优化\n沿用问题 2 / 3 交易权限",
         "$\\bar H_{t\\mid\\tau}(e;p)$：\n真值电价结算、末库存结转"),
    ]
    w, h = 5.6, 3.35
    for title, x, y, fc, accent, top, middle, bottom in modules:
        # Dashed grouping box and colored title strip.
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor="#69747C",
                               linewidth=.9, linestyle=(0, (3, 2))))
        ax.add_patch(Rectangle((x+.12, y+h-.55), w-.24, .4, facecolor=fc,
                               edgecolor=accent, linewidth=.8))
        ax.text(x+w/2, y+h-.35, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color="#33434F")
        # Three vertically aligned stages.
        bx, bw, bh = x+.55, w-1.1, .62
        _box(ax, bx, y+2.13, bw, bh, top, "#FBFCFC", fontsize=8.2, ec=accent)
        _box(ax, bx, y+1.28, bw, bh, middle, fc, fontsize=8.2, ec=accent)
        _box(ax, bx, y+.43, bw, bh, bottom, "#F5F2F7", fontsize=8.2, ec=accent)
        _arrow(ax, (x+w/2, y+2.13), (x+w/2, y+1.90), color=accent, lw=1.0)
        _arrow(ax, (x+w/2, y+1.28), (x+w/2, y+1.05), color=accent, lw=1.0)

    # Value-function chain and progression arrows between modules.
    ax.text(6.3, 4.46, r"价值函数链：$F_t(e)$  →  $\bar H_t(e)$  →  $\bar H_{t\mid\tau}(e)$  →  $\bar H_{t\mid\tau}(e;p)$",
            ha="center", va="center", fontsize=8.7, color="#4F5E66",
            bbox=dict(boxstyle="round,pad=.22", facecolor="#F1F5EE", edgecolor="#A9B9A6", linewidth=.7))
    _arrow(ax, (6.12, 6.48), (6.48, 6.48), color="#829B83", lw=1.2)
    _arrow(ax, (9.35, 4.78), (9.35, 4.35), color="#829B83", lw=1.2)
    _arrow(ax, (6.48, 2.58), (6.12, 2.58), color="#829B83", lw=1.2)
    _arrow(ax, (3.25, 4.35), (3.25, 4.78), color="#829B83", lw=1.2)
    return save(fig, "fig_f01_framework")


def _flow_canvas(height, ymax):
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
    ax.set(xlim=(0, 12), ylim=(ymax, 0))
    ax.axis("off")
    return fig, ax


def _flow_node(ax, x, y, text, w=3.8, h=.68, kind="process"):
    colors = {"process": ("#EAF1F7", "#66849A"),
              "action": ("#EAF2E4", "#829975"),
              "terminal": ("#FAF3D9", "#B8A766"),
              "decision": ("#F6E8E6", "#BA8B85")}
    fc, ec = colors[kind]
    if kind == "decision":
        patch = Polygon([(x, y-h/2), (x+w/2, y), (x, y+h/2), (x-w/2, y)],
                        facecolor=fc, edgecolor=ec, linewidth=1)
    else:
        patch = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0,rounding_size=.04",
                              facecolor=fc, edgecolor=ec, linewidth=1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=10,
            color="#26343D", linespacing=1.3)


def _flow_edge(ax, points, label=None, label_at=None):
    # Explicit orthogonal routes keep branches and feedback outside the nodes.
    color = "#4E5A61"
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, lw=1, solid_capstyle="butt")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1,
                                shrinkA=0, shrinkB=0, mutation_scale=10))
    if label:
        ax.text(*label_at, label, ha="center", va="center", fontsize=9.2,
                color=color, bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


def fig_f19():
    """Decision branches of DPValueExecutor.step and its execution loop."""
    fig, ax = _flow_canvas(6.8, 11.9)
    _flow_node(ax, 5.8, .48, "输入当前库存、购电计划与未来费用函数", w=7.1, kind="terminal")
    _flow_node(ax, 5.8, 1.56, "观测当段真实负载、光伏与电价\n计算净缺口 $r_t$", w=5.2, h=.78)
    _flow_node(ax, 5.8, 2.89, "$r_t>0$？", w=2.6, h=1.05, kind="decision")
    _flow_node(ax, 2.15, 4.15, "利用富余电量充电\n受功率上限与库存上限约束", w=3.45, h=.85, kind="action")
    _flow_node(ax, 8.25, 4.15, "由未来费用函数与当段电价\n计算动态保留水平 $R_t$", w=4.3, h=.85)
    _flow_node(ax, 8.25, 5.57, "$S_{t-1}>R_t$？", w=3.05, h=1.05, kind="decision")
    _flow_node(ax, 6.05, 6.88, "限额放电补缺口\n库存不低于保留水平", w=3.15, h=.85, kind="action")
    _flow_node(ax, 10.2, 6.88, "保留库存\n不放电", w=2.6, h=.85, kind="action")
    _flow_node(ax, 8.25, 8.15, "剩余缺口由紧急购电补足\n$E_t=r_t-D_t$", w=4.3, h=.78, kind="action")
    _flow_node(ax, 5.8, 9.3, "更新库存、富余电量与当段费用", w=5.3)
    _flow_node(ax, 5.8, 10.43, "本轮执行区间结束？", w=3.9, h=1.0, kind="decision")
    _flow_node(ax, 5.8, 11.48, "输出执行结果与末库存", w=4.2, h=.58, kind="terminal")
    for a, b in [((5.8,.82),(5.8,1.17)), ((5.8,1.95),(5.8,2.365)),
                 ((8.25,4.575),(8.25,5.045)), ((5.8,9.64),(5.8,9.93))]:
        _flow_edge(ax, [a,b])
    _flow_edge(ax, [(4.5,2.89),(2.15,2.89),(2.15,3.725)], "否（富余）", (2.85,2.68))
    _flow_edge(ax, [(7.1,2.89),(8.25,2.89),(8.25,3.725)], "是（缺口）", (8.8,3.23))
    _flow_edge(ax, [(6.725,5.57),(6.05,5.57),(6.05,6.455)], "是", (6.05,5.99))
    _flow_edge(ax, [(9.775,5.57),(10.2,5.57),(10.2,6.455)], "否", (10.2,5.99))
    _flow_edge(ax, [(6.05,7.305),(6.05,7.56),(8.25,7.56),(8.25,7.76)])
    _flow_edge(ax, [(10.2,7.305),(10.2,7.56),(8.25,7.56),(8.25,7.76)])
    _flow_edge(ax, [(8.25,8.54),(8.25,8.78),(5.8,8.78),(5.8,8.96)])
    _flow_edge(ax, [(2.15,4.575),(2.15,9.3),(3.15,9.3)])
    _flow_edge(ax, [(3.85,10.43),(.2,10.43),(.2,1.56),(3.2,1.56)],
               "否：进入下一段", (1.95,10.43))
    _flow_edge(ax, [(5.8,10.93),(5.8,11.19)], "是", (6.18,11.05))
    return save(fig, "fig_f19_dp_flowchart")


def fig_f20():
    """Issue-time loop: freeze at midnight, readjust later, execute one block."""
    fig, ax = _flow_canvas(6.1, 10.65)
    _flow_node(ax, 5.35, .49, "承接前一日末库存，设置发布时刻 $\\tau=0$", w=7, kind="terminal")
    _flow_node(ax, 5.35, 1.6, "接入最新预报与真实库存\n更新预测及配对残差场景", w=5.5, h=.8)
    _flow_node(ax, 5.35, 2.85, "$\\tau=0$？", w=2.65, h=1, kind="decision")
    _flow_node(ax, 2.6, 4.05, "求解日前采购模型\n冻结全天原计划 $G^0$", w=4.1, h=.82)
    _flow_node(ax, 8.1, 4.05, "以当前真实库存重优化\n相对 $G^0$ 调整当天剩余购电量", w=4.5, h=.82)
    _flow_node(ax, 5.35, 5.38, "提交本轮购电量：仅至下一发布时刻\n后续临时决策留待下轮覆盖", w=7.3, h=.82, kind="action")
    _flow_node(ax, 5.35, 6.61, "按本轮计划重建未来费用函数\n更新动态库存保留水平", w=5.5, h=.8)
    _flow_node(ax, 5.35, 7.84, "调用 DP 价值执行器\n逐段执行至下一发布时刻或 24:00", w=6.8, h=.8, kind="action")
    _flow_node(ax, 5.35, 9.09, "当天结束？", w=3.2, h=1, kind="decision")
    _flow_node(ax, 5.35, 10.17, "汇总实际账单，末库存结转次日", w=6, kind="terminal")
    for a,b in [((5.35,.83),(5.35,1.2)), ((5.35,2),(5.35,2.35)),
                ((5.35,5.79),(5.35,6.21)), ((5.35,7.01),(5.35,7.44)),
                ((5.35,8.24),(5.35,8.59))]:
        _flow_edge(ax,[a,b])
    _flow_edge(ax, [(4.025,2.85),(2.6,2.85),(2.6,3.64)], "是：0:00", (2.6,3.23))
    _flow_edge(ax, [(6.675,2.85),(8.1,2.85),(8.1,3.64)], "否：6 / 12 / 18 时", (8.1,3.23))
    _flow_edge(ax, [(2.6,4.46),(2.6,4.75),(5.35,4.75),(5.35,4.97)])
    _flow_edge(ax, [(8.1,4.46),(8.1,4.75),(5.35,4.75),(5.35,4.97)])
    _flow_edge(ax, [(6.95,9.09),(11.45,9.09),(11.45,1.6),(8.1,1.6)],
               "否：推进至下一发布时刻", (9.33,9.09))
    _flow_edge(ax, [(5.35,9.59),(5.35,9.83)], "是", (5.73,9.69))
    return save(fig, "fig_f20_rolling_flowchart")


def fig_f02():
    vf = read_json(AUX / "q1_value_functions.json")["picks"]
    fig, axes = panels(2, height=4.7, sharex=True)
    for t, color, ls in zip([0, 36, 72, 108], PALETTE, STYLES):
        p = vf[str(t)]
        axes[0].plot(p["x"], np.array(p["y"])/1e4, color=color, ls=ls, label=f"{t//6:02d}:00")
        axes[1].stairs(-np.array(p["slopes"]), p["x"], color=color, ls=ls, baseline=None)
    axes[0].set(ylabel="未来最低费用（万元）", title="(a) 库存价值函数")
    axes[1].set(ylabel="边际价值（元/kWh）", xlabel="段初储电量（kWh）", title="(b) 库存边际价值")
    return save(fig, "fig_f02_value_function")


def fig_f03():
    z = read_trace("q1")
    fig, axes = panels(2, height=4.1, sharex=True, gridspec_kw={"height_ratios":[2.5,1]})
    for key, label, color, ls in [("load","负载",C_GRAY,"-"),("pv","光伏",C_GREEN,"--"),("G","计划购电",C_BLUE,"-")]:
        axes[0].step(HOURS, z[key]*6, where="pre", label=label, color=color, ls=ls)
    axes[0].set(ylabel="功率（kW）", title="(a) 负载、光伏与购电")
    axes[1].step(HOURS, z["price"], where="pre", color=C_PURPLE)
    axes[1].set(ylabel="电价（元/kWh）", title="(b) 分时电价")
    time_axis(axes[1])
    return save(fig, "fig_f03_q1_day")


def fig_f14():
    z = read_trace("q1")
    fig, axes = panels(2, height=4.4, sharex=True)
    axes[0].bar(HOURS-1/12,z["C"]*6,width=1/6,color=C_BLUE,label="充电（正）")
    axes[0].bar(HOURS-1/12,-z["D"]*6,width=1/6,color=C_RED,label="放电（负）")
    axes[0].axhline(0,color=C_GRAY,lw=.8)
    axes[0].set(ylabel="充放电功率（kW）",title="(a) 充放电动作")
    axes[1].plot(np.r_[0,HOURS],np.r_[data.SOC_INIT,z["S"]],color=C_GREEN,label="储电量")
    axes[1].scatter([0,24],[6000,6000],s=24,color=C_GREEN,zorder=5)
    axes[1].annotate("首末均为 6000 kWh",(24,6000),xytext=(-8,12),textcoords="offset points",ha="right",fontsize=9)
    axes[1].set_title("(b) 库存与边界")
    bounds(axes[1]); time_axis(axes[1])
    return save(fig,"fig_f14_q1_storage")


def fig_f04():
    with np.load(AUX/"forecast_day_2025-03-20.npz") as f:
        z={k:f[k].copy() for k in f.files}
    fig=plt.figure(figsize=(6.8,4.7),layout="constrained")
    gs=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    for ax,key,title in zip(axes[:2],["load","pv"],["(a) 负载预测","(b) 光伏预测"]):
        ax.plot(HOURS,z[key+"_true"]*6,color=C_GRAY,label="真值")
        ax.plot(HOURS,z[key+"_pred"]*6,color=C_BLUE,ls="--",label="0:00 点预测")
        ax.set(ylabel="功率（kW）",title=title);time_axis(ax)
    scen=(z["L_scen"]-z["PV_scen"])*6
    lo,hi=np.percentile(scen,[10,90],axis=0)
    ax=axes[2]
    ax.fill_between(HOURS,lo,hi,color=C_GREEN,alpha=.2,label="场景 10%–90% 分位带")
    for path in scen: ax.plot(HOURS,path,color=C_GREEN,alpha=.2,lw=.5)
    ax.plot([],[],color=C_GREEN,lw=.7,label="30 条配对场景")
    ax.plot(HOURS,(z["load_pred"]-z["pv_pred"])*6,color=C_BLUE,ls="--")
    ax.plot(HOURS,(z["load_true"]-z["pv_true"])*6,color=C_GRAY)
    ax.set(ylabel="净负荷（kW）",title="(c) 净负荷场景");time_axis(ax)
    return save(fig,"fig_f04_load_forecast")


def fig_f06():
    z=read_trace("q2_48h");i=list(z["dates"]).index(DAYS[0])
    fig,axes=panels(2,height=4.6,sharex=True)
    soc(axes[0],z,i)
    axes[0].plot(HOURS,z["R"][i],color=C_PURPLE,ls="--",label="动态保留水平")
    axes[0].set_title("(a) 库存与保留水平")
    axes[1].bar(HOURS-1/12,z["E"][i]*6,width=1/6,color=C_RED,label="紧急购电")
    axes[1].set(ylabel="紧急购电功率（kW）",title="(b) 紧急购电与当段电价")
    axes[1].yaxis.label.set_color(C_RED);axes[1].tick_params(axis="y",colors=C_RED)
    ax2=axes[1].twinx();ax2.step(HOURS,z["price"][i],where="pre",color=C_BLUE,ls="--",label="当段电价")
    ax2.set_ylabel("电价（元/kWh）",color=C_BLUE);ax2.tick_params(axis="y",colors=C_BLUE);ax2.grid(False)
    time_axis(axes[1])
    return save(fig,"fig_f06_dp_mechanism")


def fig_f07():
    monthly=read_summary("q2_48h")["monthly"];months=sorted(monthly);x=np.arange(len(months))
    plan=np.array([monthly[m]["plan_cost"] for m in months])/1e4
    emergency=np.array([monthly[m]["emergency_cost"] for m in months])/1e4
    fig,axes=panels(2,height=4.7,sharex=True,gridspec_kw={"height_ratios":[2,1]})
    axes[0].bar(x,plan,color=C_BLUE,label="计划购电费",width=.65)
    axes[0].bar(x,emergency,bottom=plan,color=C_RED,label="紧急购电费",width=.65)
    for a,b in zip(x,plan+emergency):axes[0].text(a,b+2,f"{b:.1f}",ha="center",fontsize=8)
    axes[0].set(ylabel="费用（万元）",ylim=(0,max(plan+emergency)*1.15),title="(a) 月度账单分解")
    axes[1].bar(x,emergency,color=C_RED,width=.65)
    axes[1].set(ylabel="紧急费用（万元）",xticks=x,xticklabels=[f"{int(m[-2:])}月" for m in months],title="(b) 紧急购电费细节")
    return save(fig,"fig_f07_q2_monthly")


def fig_f08():
    fig,ax=panels(height=3.8)
    ax.broken_barh([(0,24)],(4.7,.5),facecolors=C_PURPLE,label="全天冻结计划")
    ax.text(12,4.95,"0:00 冻结全天 $G^0$",ha="center",va="center",color="white",fontsize=9)
    for k,t in enumerate([0,6,12,18]):
        y=3.5-k
        if t:ax.broken_barh([(0,t)],(y,.5),facecolors="#DCE1E5",label="已执行" if k==1 else None)
        ax.broken_barh([(t,6)],(y,.5),facecolors=C_BLUE,label="本轮提交区间" if k==0 else None)
        if t+6<24:ax.broken_barh([(t+6,24-t-6)],(y,.5),facecolors="#DCE9F2",edgecolors=C_BLUE,linestyles="--",label="当天临时决策" if k==0 else None)
        ax.broken_barh([(24,24)],(y,.5),facecolors="#E9E3F0",edgecolors=C_PURPLE,linestyles=":",label="次日前瞻" if k==0 else None)
        ax.text(-1,y+.25,f"{t:02d}:00",ha="right",va="center",fontsize=9)
    ax.axvline(24,color=C_GRAY,lw=1,ls="--")
    ax.set(xlim=(-.2,48.2),ylim=(.1,5.5),xticks=np.arange(0,49,6),yticks=[],xlabel="自当天 0:00 起的小时")
    ax.grid(axis="y",visible=False)
    return save(fig,"fig_f08_rolling_timeline")


def fig_f09():
    costs=np.array([read_summary(n)["total_cost"] for n in ["q3_0","q3_06","q3_0612","q3_baseline"]])/1e4
    savings=costs[:-1]-costs[1:]
    fig,axes=panels(2,height=4.5)
    axes[0].plot(range(4),costs,"o-",color=C_BLUE)
    for x,v in enumerate(costs):axes[0].annotate(f"{v:.2f}",(x,v),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9)
    axes[0].set(xticks=range(4),xticklabels=["0","0/6","0/6/12","0/6/12/18"],ylabel="全年费用（万元）",ylim=(min(costs)-5,max(costs)+12),title="(a) 预报时刻组合（24h 基线）")
    axes[1].bar(range(3),savings,width=.48,color=[C_GREEN,C_BLUE,C_GREEN])
    for x,v in enumerate(savings):axes[1].text(x,v+.6,f"{v:.2f}",ha="center",fontsize=9)
    axes[1].set(xticks=range(3),xticklabels=["新增 6:00","新增 12:00","新增 18:00"],ylabel="相邻节省（万元）",ylim=(0,max(savings)*1.25),title="(b) 按固定顺序增加时刻的节省")
    return save(fig,"fig_f09_issue_marginal")


def fig_f10():
    records=read_json(AUX/"price_forecast_days.json")["branches"]
    fig,axes=panels(2,2,height=4.9,sharex=True,sharey=True)
    for row,b in enumerate(["q4_2","q4_3"]):
        for col,d in enumerate([DAYS[0],DAYS[-1]]):
            rec=records[b][d];ax=axes[row,col]
            ax.step(HOURS,rec["truth"],where="pre",color=C_GRAY,label="电价真值")
            ax.step(HOURS,rec["pred"],where="pre",color=C_BLUE,ls="--",label="0:00 电价预测")
            ax.set_title(f"{d[5:]}  ·  MAE {rec['mae']:.3f}",loc="left")
            ax.set_ylabel(f"问 4-{row+2}\n电价（元/kWh）")
            time_axis(ax)
    return save(fig,"fig_f10_price_forecast")


def fig_f11():
    sums={b:read_summary(b+"_48h")["monthly"] for b in BRANCHES}
    months=sorted(sums["q2"]);x=np.arange(len(months))
    def diff(a,b):return np.array([sums[a][m]["total_cost"]-sums[b][m]["total_cost"] for m in months])/1e4
    fig,axes=panels(2,height=4.6,sharex=True)
    axes[0].bar(x-.18,diff("q4_2","q2"),width=.35,color=C_PURPLE,label="问 4-2 − 问 2")
    axes[0].bar(x+.18,diff("q4_3","q3"),width=.35,color=C_BLUE,label="问 4-3 − 问 3")
    axes[0].set(ylabel="费用差（万元）",title="(a) 相对对应常数电价分支")
    axes[1].bar(x,diff("q4_2","q4_3"),width=.6,color=C_GREEN,label="问 4-2 − 问 4-3")
    axes[1].set(ylabel="账单差（万元）",title="(b) 两分支实际账单差",xticks=x,xticklabels=[f"{int(m[-2:])}月" for m in months])
    for ax in axes:ax.axhline(0,color=C_GRAY,lw=.8)
    return save(fig,"fig_f11_q4_monthly")


def fig_f12():
    z=read_trace("q3_48h");i=list(z["dates"]).index(DAYS[0]);delta=z["Ga"][i]-z["G0"][i]
    fig,axes=panels(3,height=6,sharex=True)
    axes[0].step(HOURS,z["G0"][i]*6,where="pre",color=C_GRAY,ls="--",label="原计划")
    axes[0].step(HOURS,z["Ga"][i]*6,where="pre",color=C_BLUE,label="最终交付购电")
    axes[0].set(ylabel="购电功率（kW）",title="(a) 原计划与最终提交量")
    axes[1].bar(HOURS-1/12,np.maximum(delta,0)*6,width=1/6,color=C_BLUE,label="调增")
    axes[1].bar(HOURS-1/12,np.minimum(delta,0)*6,width=1/6,color=C_RED,label="调减")
    axes[1].set(ylabel="调整功率（kW）",title="(b) 相对原计划的调整")
    soc(axes[2],z,i)
    axes[2].plot(HOURS,z["R"][i],color=C_PURPLE,ls="--",label="动态保留水平")
    axes[2].set_title("(c) 库存响应")
    for ax in axes:
        for t in [6,12,18]:ax.axvline(t,color=C_GRAY,ls=":",lw=.7)
    time_axis(axes[2])
    return save(fig,"fig_f12_exec_day")


def specified_soc(branches,name):
    fig,axes=panels(len(branches),height=3 if len(branches)==1 else 5.2,sharex=True,squeeze=False)
    for ax,b in zip(axes.flat,branches):
        z=read_trace(b+"_48h")
        for d,color,ls in zip(DAYS,PALETTE,STYLES):soc(ax,z,list(z["dates"]).index(d),label=d[5:],color=color,ls=ls)
        time_axis(ax)
        if len(branches)>1:ax.set_title(BRANCH_LABELS[BRANCHES.index(b)],loc="left")
    return save(fig,name)


def fig_f13():
    rec=read_json(AUX/"sensitivity_m_horizon.json")
    vals=[rec[f"M{m}_48h"]["total_cost"]/1e4 for m in [6,12,20]]+[read_summary("q2_48h")["total_cost"]/1e4]
    fig,axes=panels(1,2,height=3.3)
    axes[0].plot([6,12,20,30],vals,"o-",color=C_BLUE)
    for x,y in zip([6,12,20,30],vals):axes[0].annotate(f"{y:.2f}",(x,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8)
    axes[0].set(xticks=[6,12,20,30],xlabel="场景数 M",ylabel="全年费用（万元）",ylim=(1344,1385),title="(a) 场景数（48h）")
    for m,c,ls in [(12,C_GREEN,"--"),(30,C_BLUE,"-")]:
        v=[rec[f"M{m}_24h"]["total_cost"]/1e4,rec[f"M{m}_48h"]["total_cost"]/1e4] if m==12 else [read_summary("q2_baseline")["total_cost"]/1e4,vals[-1]]
        axes[1].plot([0,1],v,marker="o",ls=ls,color=c,label=f"M = {m}")
        for x,y in enumerate(v):axes[1].annotate(f"{y:.2f}",(x,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8)
    axes[1].set(xticks=[0,1],xticklabels=["24h","48h"],xlim=(-.25,1.25),ylim=(1350,1367),ylabel="全年费用（万元）",title="(b) 前瞻时域")
    return save(fig,"fig_f13_sensitivity")


def fig_f18():
    rows=read_json(M30/"ablation_comparison.json")
    lookup={(r["branch"],r["variant"]):r["delta_yuan"]/1e4 for r in rows}
    fig,ax=panels(height=3.6)
    for bi,(b,c,marker,label) in enumerate(zip(BRANCHES,PALETTE,["o","s","D","^"],BRANCH_LABELS)):
        for vi,v in enumerate(VARIANTS):
            if (b,v) in lookup:ax.scatter(lookup[b,v],vi+(bi-1.5)*.14,color=c,marker=marker,s=32,label=label if vi==0 else None,zorder=3)
    ax.axvline(0,color=C_GRAY,lw=.9)
    ax.set(yticks=range(5),yticklabels=VARIANT_LABELS,xlabel="费用增量：变体 − 对应 24h 基线（万元）",ylim=(4.6,-.6))
    ax.grid(axis="y",visible=False)
    return save(fig,"fig_f18_components")


def main():
    OUT.mkdir(exist_ok=True)
    made=[fn() for fn in [fig_f01,fig_f02,fig_f03,fig_f14,fig_f04,fig_f19,fig_f06,fig_f07,fig_f20,fig_f08,fig_f09,fig_f12,fig_f10,fig_f11,fig_f13,fig_f18]]
    made += [specified_soc(["q2"],"fig_f15_q2_soc"),specified_soc(["q3"],"fig_f16_q3_soc"),specified_soc(["q4_2","q4_3"],"fig_f17_q4_soc")]
    (OUT/"paper_figures.json").write_text(json.dumps(made,ensure_ascii=False,indent=2)+"\n")
    print("Generated",len(made),"paper figures")


if __name__=="__main__":main()
