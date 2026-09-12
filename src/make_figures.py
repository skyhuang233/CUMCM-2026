"""Generate paper tables and figures exclusively from saved M30 results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "results/m30", ROOT / "plot"
TABLES = ROOT / "paper/CUMCMThesis-master/generated"
BRANCHES = {"q2": "问2", "q3": "问3", "q4_2": "问4-2", "q4_3": "问4-3"}
VARIANTS = {"48h": "48h+$V$", "candidates": "候选重评", "legacy_means": "旧均值", "same_type": "同类型池"}
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
for family in ("Arial Unicode MS", "PingFang SC", "Songti SC"):
    if family in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = family
        break
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0",
                     "pdf.fonttype": 42, "axes.unicode_minus": False})

def summary(name):
    path = DATA / f"{name}.summary.json"
    if not path.exists(): return None
    item = json.loads(path.read_text())
    if name != "q1" and not item.get("complete_year"):
        raise ValueError(f"Incomplete scoring year: {path}")
    return item

def trace(name):
    path = DATA / f"{name}.trace.npz"
    if not path.exists(): return None
    with np.load(path) as data: return {key: data[key].copy() for key in data.files}

def save(fig, *names):
    fig.tight_layout()
    for name in names: fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def table(caption, label, headers, rows, columns):
    return "\n".join([r"\begin{table}[!htbp]", r"\centering\small", "\\caption{"+caption+"}\\label{"+label+"}",
        "\\begin{tabular}{"+columns+"}", r"\toprule", " & ".join(headers)+r" \\", r"\midrule",
        *[" & ".join(map(str,row))+r" \\" for row in rows], r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

def write_tables(name, blocks):
    (TABLES / (name+".tex")).write_text("\n".join(blocks),encoding="utf-8")

def main():
    OUT.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    missing, macros = [], []
    q1, t1 = summary("q1"), trace("q1")
    if q1 and t1:
        write_tables("q1_tables", [
            table("问1指定时段购电量（kWh）", "tab:q1purchase", ["时段起点","10:00","12:00","14:00","16:00","18:00","20:00"],
                  [["购电量",*[f"{v:.2f}" for v in q1["seg_purchase"]]]], "lrrrrrr"),
            table("问1四小时充放电汇总（kWh）", "tab:q1storage", ["时段","充电量","放电量"],
                  [[a.replace("-","--"),f"{c:.2f}",f"{d:.2f}"] for a,c,d in zip(q1["interval_labels"],q1["charge"],q1["discharge"])], "lrr")])
        macros += [rf"\newcommand{{\QoneCost}}{{{q1['purchase_cost']:,.2f}}}",rf"\newcommand{{\QoneEnergy}}{{{q1['daily_purchase']:,.2f}}}"]
        h=np.arange(1,145)/6; fig, axes=plt.subplots(3,1,figsize=(6.1,5.6),sharex=True)
        axes[0].step(h,t1["price"],where="pre",color=GRAY); axes[0].set_ylabel("电价（元/kWh）")
        for key,label,color in [("load","负载",GRAY),("pv","光伏",AQUA),("G","购电",BLUE)]:
            axes[1].plot(h,t1[key]*6,label=label,color=color)
        axes[1].set_ylabel("功率（kW）"); axes[1].legend(frameon=False,ncol=3)
        axes[2].plot(np.r_[0,h],np.r_[6000,t1["S"]],color=AQUA)
        axes[2].set_ylabel("储电量（kWh）"); axes[2].set_xlabel("小时"); axes[2].set_xticks(range(0,25,4))
        save(fig,"fig_q1_day")
    else: missing.append("q1")
    bases={b:summary(b+"_baseline") for b in BRANCHES}; blocks=[]
    if all(bases.values()):
        rows=[[BRANCHES[b],f"{x['total_cost']/1e4:.4f}",f"{x['emergency_cost']/1e4:.4f}",f"{x['emergency_kwh']:.2f}",f"{x['soc_end']:.2f}"] for b,x in bases.items()]
        blocks.append(table("M=30基线：334个评分日的实际账单与末库存","tab:baseline",
            ["分支","总费/万元","紧急费/万元","紧急量/kWh","末库存/kWh"],rows,"lrrrr"))
        reference={"q2":13566395.37,"q3":13024699.25,"q4_2":14269748.14,"q4_3":13716889.66}
        rows=[[BRANCHES[b],f"{reference[b]:.2f}",f"{x['total_cost']:.2f}",f"{x['total_cost']-reference[b]:+.2f}",f"{100*(x['total_cost']/reference[b]-1):+.5f}"] for b,x in bases.items()]
        blocks.append(table("与四问完整稿报告费用的对照","tab:reference",["分支","参考稿/元","本次/元","差额/元","差额/\\%"],rows,"lrrrr"))
        vals=[x["total_cost"]/1e4 for x in bases.values()]; fig,ax=plt.subplots(figsize=(6.1,3))
        ax.bar(list(BRANCHES.values()),vals,color=[BLUE,AQUA,ORANGE,"#7357a5"])
        for i,v in enumerate(vals): ax.text(i,v+8,f"{v:.2f}",ha="center")
        ax.set_ylim(0,max(vals)*1.13); ax.set_ylabel("评分期费用（万元）"); save(fig,"fig_branch_costs","fig_cascade")
        fig,ax=plt.subplots(figsize=(6.3,3.1))
        for b,x in bases.items():
            months=sorted(x["monthly"])
            ax.plot([int(m[-2:]) for m in months],[x["monthly"][m]["total_cost"]/1e4 for m in months],marker="o",ms=3,label=BRANCHES[b])
        ax.set_xticks(range(2,13)); ax.set_xlabel("月份"); ax.set_ylabel("月费用（万元）"); ax.legend(frameon=False,ncol=4)
        save(fig,"fig_monthly_costs")
        for b,m in [("q2","QtwoCost"),("q3","QthreeCost"),("q4_2","QfourTwoCost"),("q4_3","QfourThreeCost")]:
            macros.append(rf"\newcommand{{\{m}}}{{{bases[b]['total_cost']/1e4:.4f}}}")
    else: missing += [b+"_baseline" for b,x in bases.items() if x is None]
    write_tables("baseline_tables",blocks)
    names=["q3_0","q3_06","q3_0612","q3_baseline"]; issues=[summary(n) for n in names]; blocks=[]
    if all(issues):
        rows=[[["0","0,6","0,6,12","0,6,12,18"][i],f"{x['total_cost']:.2f}",f"{x['emergency_cost']:.2f}",
               "---" if i==0 else f"{issues[i-1]['total_cost']-x['total_cost']:.2f}"] for i,x in enumerate(issues)]
        blocks.append(table("问3发布时刻组合与相邻组合的费用变化","tab:issues",["发布时刻/h","总费用/元","紧急费/元","相邻节省/元"],rows,"lrrr"))
        costs=np.array([x["total_cost"] for x in issues])/1e4; fig,axes=plt.subplots(1,2,figsize=(6.3,2.8))
        axes[0].bar(range(4),costs,color=BLUE); axes[0].set_xticks(range(4),["0","+6","+12","+18"])
        axes[0].set_ylabel("总费用（万元）"); axes[0].set_xlabel("逐步增加发布时刻")
        savings=costs[:-1]-costs[1:]; axes[1].bar(range(3),savings,color=AQUA)
        axes[1].set_xticks(range(3),["6:00","12:00","18:00"]); axes[1].set_ylabel("相邻组合节省（万元）")
        for i,v in enumerate(savings): axes[1].text(i,v+.3,f"{v:.2f}",ha="center")
        axes[1].set_ylim(0,max(savings)*1.18); save(fig,"fig_q3_issues","fig_q3_variants")
    else: missing += [n for n,x in zip(names,issues) if x is None]
    write_tables("issues_tables",blocks)
    blocks=[]; ablations=[]; fig,axes=plt.subplots(2,2,figsize=(6.6,5.1))
    for ax,(b,label) in zip(axes.flat,BRANCHES.items()):
        variants=dict(VARIANTS)
        if b.startswith("q4"): variants["legacy_price"]="旧价格"
        rows=[]; labels=[]; deltas=[]
        for v,description in variants.items():
            name=b+"_"+v; x=summary(name); base=bases[b]
            if x is None or base is None: missing.append(name); continue
            delta=x["total_cost"]-base["total_cost"]; percent=100*delta/base["total_cost"]
            rows.append([description,f"{x['total_cost']/1e4:.4f}",f"{delta:+.2f}",f"{percent:+.3f}",f"{x['soc_end']:.1f}"])
            labels.append(description); deltas.append(percent)
            ablations.append(dict(branch=b,variant=v,total_cost=x["total_cost"],delta_yuan=delta,delta_percent=percent))
        if rows:
            blocks.append(table(label+"单因子消融（与本分支基线比较）","tab:ablation-"+b.replace("_","-"),
                ["仅改变组件","总费/万元","增量/元","增量/\\%","末库存/kWh"],rows,"lrrrr"))
            ax.bar(range(len(deltas)),deltas,color=[ORANGE if d>0 else AQUA for d in deltas]); ax.set_xticks(range(len(labels)),labels,rotation=18)
        ax.axhline(0,color=GRAY,lw=.7); ax.set_title(label); ax.set_ylabel("费用变化（%）")
    if ablations: save(fig,"fig_ablations","fig_sensitivity")
    else: plt.close(fig)
    write_tables("ablation_tables",blocks)
    (DATA/"ablation_comparison.json").write_text(json.dumps(ablations,ensure_ascii=False,indent=2))
    findings=[]
    for b,label in BRANCHES.items():
        rows=[r for r in ablations if r["branch"]==b]
        if not rows: continue
        improvements=[r for r in rows if r["delta_yuan"]<0]
        descriptions=dict(VARIANTS,legacy_price="旧价格")
        if improvements:
            details="；".join(descriptions[r["variant"]]+f"减少{-r['delta_yuan']:.2f}元（{-r['delta_percent']:.3f}\\%）" for r in improvements)
            findings.append(label+"中费用低于基线的单因子对照为："+details+"。这仅支持在该分支、该评分期继续保留相应组件作为候选，不表示组件叠加后仍有同样收益。")
        else:
            findings.append(label+"的已完成单因子对照均未降低实际费用，因此本轮结果不支持用这些组件替换该分支参考基线。")
    write_tables("ablation_findings",findings)
    blocks=[]
    for b,label in BRANCHES.items():
        if bases[b] is None: continue
        rows=[[d[5:],f"{x.get('daily_plan', x['daily_purchase']):.2f}",f"{x['purchase_cost']:.2f}",f"{x['emergency_kwh']:.2f}",f"{x['soc_start']:.2f}",f"{x['soc_end']:.2f}"] for d,x in bases[b]["report_days"].items()]
        blocks.append(table(label+"指定日摘要（同一次连续运行）","tab:days-"+b.replace("_","-"),
            ["日期","计划量/kWh","费用/元","紧急量/kWh","初库存/kWh","末库存/kWh"],rows,"lrrrrr"))
    write_tables("report_day_tables",blocks)
    t3=trace("q3_baseline")
    if t3:
        i=list(t3["dates"]).index("2025-03-20"); h=np.arange(1,145)/6
        fig,axes=plt.subplots(3,1,figsize=(6.2,5.6),sharex=True)
        axes[0].step(h,t3["G0"][i]*6,where="pre",label="$G^0$",ls="--",color=BLUE)
        axes[0].step(h,t3["Ga"][i]*6,where="pre",label="$G^a$",color=AQUA)
        axes[0].set_ylabel("购电功率（kW）"); axes[0].legend(frameon=False,ncol=2)
        axes[1].bar(h,t3["E"][i]*6,width=1/6,color=ORANGE); axes[1].set_ylabel("紧急购电（kW）")
        axes[2].plot(np.r_[0,h],np.r_[t3["S"][i-1,-1],t3["S"][i]],color=AQUA,label="实际储电量")
        axes[2].plot(h,t3["R"][i],color=GRAY,lw=.8,ls=":",label="动态保留水平")
        axes[2].set_ylabel("库存（kWh）"); axes[2].set_xlabel("小时"); axes[2].set_xlim(0,24)
        axes[2].set_xticks(range(0,25,4)); axes[2].legend(frameon=False,ncol=2); save(fig,"fig_q3_day")
    else: missing.append("q3_baseline.trace")
    write_tables("m30_numbers",macros)
    (OUT/"m30-data-status.json").write_text(json.dumps({"missing":sorted(set(missing))},ensure_ascii=False,indent=2))
    print(json.dumps({"missing":sorted(set(missing)),"ablation_count":len(ablations)},ensure_ascii=False))

if __name__ == "__main__": main()
