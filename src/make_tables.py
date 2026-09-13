"""Generate the paper's numerical tables from frozen traces and summaries.

Run ``python -m src.make_tables``. Monetary settlement and interval aggregation
are shared with the official result exporters; no optimization is rerun.
"""
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from . import data
from .results import TABLE1_LABELS, TABLE1_SEGS, interval_sums, summarize_day, summarize_day_q3

ROOT = Path(__file__).resolve().parents[1]
M30 = ROOT / 'results/latest/m30'
OUT = ROOT / 'paper/CUMCMThesis-master/generated'
DAYS = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
BRANCHES = ['q2', 'q3', 'q4_2', 'q4_3']
NAMES = ['问 2', '问 3', '问 4-2', '问 4-3']


def write_table(name, text):
    (OUT / name).write_text(text.rstrip() + '\n')


def summary(name):
    return json.loads((M30 / f'{name}.summary.json').read_text())


def trace(name):
    with np.load(M30 / f'{name}.trace.npz') as z:
        return {k: z[k].copy() for k in z.files}


def num(x, precision=2):
    if abs(x) < .5*10**(-precision): x = 0.
    return f'{x:.{precision}f}'


def row(cells):
    return ' & '.join(str(c) for c in cells) + r' \\' + '\n'


def tabular(head, rows, spec=None):
    spec = spec or 'L' + 'R'*(len(head)-1)
    return (r'\begin{tabularx}{\textwidth}{@{}'+spec+r'@{}}'+'\n'+r'\toprule'+'\n'
            +r'\rowcolor{cumcmgray} '+row(head)+r'\midrule'+'\n'
            +''.join(row(r) for r in rows)+r'\bottomrule'+'\n'+r'\end{tabularx}'+'\n')


def table(caption, label, head=None, rows=None, body=None, note=None):
    return (r'\begin{table}[H]\centering'+'\n'
            +r'\caption{'+caption+r'}\label{'+label+'}\n'
            +r'\small\renewcommand{\arraystretch}{1.12}\setlength{\tabcolsep}{4pt}'+'\n'
            +(body if body is not None else tabular(head,rows))
            +(r'\par\vspace{3pt}{\footnotesize '+note+'}\n' if note else '')
            +r'\end{table}'+'\n\n')


def storage_body(records):
    parts=[]
    for title, charge, discharge, start, end in records:
        lines=[r'\multicolumn{6}{@{}l}{\textbf{'+title+r'}} \\',r'\toprule',
               r'\rowcolor{cumcmgray} 时段 & 充电量 & 放电量 & 时段 & 充电量 & 放电量 \\',r'\midrule']
        for k in range(3):
            cells=[]
            for j in [k,k+3]:
                cells += [f'{j*4:02d}:00--{(j+1)*4:02d}:00',num(charge[j]),num(discharge[j])]
            lines.append(row(cells).rstrip())
        lines += [r'\midrule',row([r'\multicolumn{2}{l}{0:00 储电量}',num(start),r'\multicolumn{2}{l}{24:00 储电量}',num(end)]).rstrip(),r'\bottomrule']
        parts.append(r'\begin{tabularx}{\textwidth}{@{}LRRLRR@{}}'+'\n'+'\n'.join(lines)+'\n'+r'\end{tabularx}')
    return '\n'+(r'\par\vspace{5pt}'+'\n').join(parts)+'\n'


def build_q1():
    z=trace('q1')
    rows=[[label.replace('-','--'),num(z['G'][i])] for label,i in zip(TABLE1_LABELS,TABLE1_SEGS)]
    rows += [['全天购电量（kWh）',num(z['G'].sum())],['全天购电费（元）',num(z['price']@z['G'])]]
    text=table('问题一指定时段购电量及全天结果','tab:q1purchase',['时段／指标','购电量（kWh）／费用（元）'],rows)
    body=storage_body([('附件 1 典型日',interval_sums(z['C']),interval_sums(z['D']),data.SOC_INIT,z['S'][-1])])
    text+=table('问题一分段充放电与首末库存（kWh）','tab:q1storage',body=body)
    write_table('q1_tables.tex', text)


def build_days(branch, name):
    z=trace(branch+'_48h');adjusted=branch in ['q3','q4_3'];records=[]
    for d in DAYS:
        i=list(z['dates']).index(d)
        day=SimpleNamespace(**{k:z[k][i] for k in ['G0','Ga','C','D','S','E']},day=date.fromisoformat(d),soc_start=z['S'][i-1,-1] if i else summary(branch+'_48h')['initial_soc'])
        s=(summarize_day_q3 if adjusted else summarize_day)(day,z['price'][i])
        s['daily_plan']=float(day.G0.sum())
        s['increase']=float(np.maximum(day.Ga-day.G0,0).sum()) if adjusted else 0.
        s['decrease']=float(np.maximum(day.G0-day.Ga,0).sum()) if adjusted else 0.
        s['date_label']=d[5:]
        records.append(s)
    slug=branch.replace('_','-')
    parts=[]
    for field,label,suffix in ([('seg_plan','原计划购电量','plan'),('seg_purchase','最终提交购电量','purchase')] if adjusted else [('seg_purchase','计划购电量','purchase')]):
        rows=[[t.replace('-','--')]+[num(s[field][k]) for s in records] for k,t in enumerate(TABLE1_LABELS)]
        parts.append(table(f'{name}指定时段{label}（kWh，48h 主方法）',f'tab:{slug}-{suffix}',['时段']+[d[5:] for d in DAYS],rows))
    if adjusted:
        head=['日期','原计划','调增','调减','最终提交','紧急购电','合计购电']
        rows=[[s['date_label']]+[num(s[k]) for k in ['daily_plan','increase','decrease','daily_purchase','emergency_kwh']]+[num(s['daily_purchase']+s['emergency_kwh'])] for s in records]
        parts.append(table(f'{name}全天购电与调整量（kWh）',f'tab:days-{slug}',head,rows))
        head=['日期','计划费','减购违约费','增购费','紧急费','总费用']
        rows=[[s['date_label']]+[num(s['cost'][k]) for k in ['plan','curtail_penalty','extra','emergency','total']] for s in records]
        note=r'计划费按 $p\min(G^0,G^a)$ 结算；总费用为四项之和。'
    else:
        head=['日期','计划购电','紧急购电','合计购电','计划费','紧急费','总费用']
        rows=[[s['date_label'],num(s['daily_purchase']),num(s['emergency_kwh']),num(s['daily_purchase']+s['emergency_kwh']),num(s['plan_cost']),num(s['emergency_cost']),num(s['purchase_cost'])] for s in records]
        note='电量单位为 kWh，费用单位为元；合计购电为计划与紧急购电之和。'
    parts.append(table(f'{name}全天购电与费用' if not adjusted else f'{name}全天费用分解（元）',f'tab:{slug}-cost',head,rows,note=note))
    # Events precede storage so the inventory figure can immediately follow storage.
    eventrows=[]
    for s in records:
        if not s['emergency']: eventrows.append([s['date_label'],'无紧急购电','0.00'])
        for j,(label,kwh) in enumerate(s['emergency']):eventrows.append([s['date_label'] if j==0 else '',label.replace('-','--'),num(kwh)])
    parts.append(table(f'{name}指定日紧急购电事件',f'tab:{slug}-events',['日期','连续事件区间','电量（kWh）'],eventrows))
    body=storage_body([(s['date_label'],s['charge'],s['discharge'],s['soc_start'],s['soc_end']) for s in records])
    parts.append(table(f'{name}指定日充放电与首末库存（kWh）',f'tab:{slug}-storage',body=body,note='指定日初值取自全年连续运行的前一日末库存。'))
    write_table(f'{branch}_report_tables.tex', ''.join(parts))
    return {s['date_label']:{'cost':s['purchase_cost'],'emergency':s['emergency_kwh'],'start':s['soc_start'],'end':s['soc_end'],'increase':s['increase'],'decrease':s['decrease']} for s in records}


def build_annual():
    head=['分支','24h 总费','48h 总费','节省额','48h 紧急费']
    rows=[]
    for b,n in zip(BRANCHES,NAMES):
        base=summary(b+'_baseline');main=summary(b+'_48h')
        rows.append([n]+[num(v/1e4) for v in [base['total_cost'],main['total_cost'],base['total_cost']-main['total_cost'],main['emergency_cost']]])
    text=table('四分支综合性能比较（万元，334 个评分日）','tab:main-results',head,rows,note='24h 为单日时域配置，48h 为跨日前瞻配置；节省额为两者总费之差。')
    write_table('annual_table.tex', text)


def build_components():
    records=json.loads((M30/'ablation_comparison.json').read_text());lookup={(r['branch'],r['variant']):r['delta_yuan'] for r in records}
    variants=['48h','candidates','same_type','legacy_means','legacy_price']
    labels=['启用 48h 与次日价值','启用候选计划重评','改用同类型日场景池','改用旧均值负载预测','改用旧价格预测']
    head=['组件变化']+[r'\shortstack{'+n+r'\\基线 '+num(summary(b+'_baseline')['total_cost'])+'}' for b,n in zip(BRANCHES,NAMES)]
    rows=[[label]+[num(lookup[b,v]) if (b,v) in lookup else '---' for b in BRANCHES] for v,label in zip(variants,labels)]
    body=tabular(head,rows,spec='>{\\hsize=1.5\\hsize\\linewidth=\\hsize}L'+('>{\\hsize=.875\\hsize\\linewidth=\\hsize}R'*4))
    write_table('component_table.tex', table('组件变化相对对应 24h 基线的费用增量（元）','tab:components',body=body,note='费用增量为变体减基线，负值表示节省；“---”表示该分支不适用。'))


def build_issues():
    labels = ['0', '0/6', '0/6/12', '0/6/12/18']
    sums = [summary(n) for n in ['q3_0', 'q3_06', 'q3_0612', 'q3_baseline']]
    rows = []
    for i, (label, rec) in enumerate(zip(labels, sums)):
        saving = '---' if i == 0 else num(sums[i-1]['total_cost']-rec['total_cost'])
        rows.append([label, num(rec['total_cost']), num(rec['emergency_cost']), saving])
    write_table('issues_tables.tex', table('预报时刻组合的费用与相邻节省（元，24h 基线）', 'tab:issues', ['预报时刻', '总费用', '紧急费用', '相邻节省'], rows))


def main():
    OUT.mkdir(exist_ok=True)
    build_q1();build_annual();build_components();build_issues()
    values={b:build_days(b,n) for b,n in zip(BRANCHES,NAMES)}
    (OUT/'report_values.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    print('Generated Q1, four branch table groups, annual and component tables.')


if __name__=='__main__':main()
