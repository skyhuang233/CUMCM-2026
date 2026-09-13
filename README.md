# 微网购电与储能调度优化

2026 年全国大学生数学建模竞赛 C 题：带储能的非孤岛微网向外网购电与储能充放电的日前计划、滚动调整与执行结算。

## 方法概要

采用 **SAA 两阶段购电 LP + DP 价值执行器** 的两层架构：

1. **优化核**：每日 0:00 用经验残差场景的样本平均近似（SAA）线性规划冻结计划购电量。候选计划重评是可选增量，不是参考基线。
2. **执行层**：购电量固定后，用同批场景反推的未来费用函数得到每段动态保留水平，逐段按当段真值决定充放电——富余尽量充，缺口只放到保留水平为止，紧急购电补足余额。
3. **问 3 扩展**：在 6:00/12:00/18:00 用新光伏预报滚动重优化，只提交到下一预报时刻。
4. **问 4 扩展**：4-2 在 0:00 冻结采购、在四个发布时刻更新价格与库存价值；4-3 同时重订剩余采购。价格残差与负载/光伏按完整发布路径配对。

1 月重建历史预测和完整残差，储能待机；2--12 月为连续评分期。当前冻结的 M=30 完整精度矩阵保存在 `results/latest/m30/`，每项摘要的 `complete_year` 标记全年完成状态，`verification.json` 记录整组产物的独立物理及账单核验。旧稿备份位于 `.scratch/m30-review/`。

## 快速开始

### 环境搭建

```bash
# 需要 Python >= 3.13 和 uv
uv venv .venv
uv pip install -e .
```

### 运行四问

```bash
# 问 1：确定性日前计划（秒级）
.venv/bin/python -m src.run_q1

# 问 2：基线全年回测（默认不调参）
.venv/bin/python -m src.run_q2 --no-write

# 问 3：0/6/12/18 时发布后的完整 24h 滚动重优化
.venv/bin/python -m src.run_q3 --skip-tuning --no-write

# 问 4：波动电价（分别计算 4-2 和 4-3，不写结果）
.venv/bin/python -m src.run_q4 --which both --no-write
```

带 `--no-write` 的问2--4命令只作不落盘检查；去掉该参数才写入 `results/`。问1默认直接写出确定性结果。

### 运行测试

```bash
.venv/bin/python -m pytest
```

### 生成论文图表

```bash
.venv/bin/python -m src.paper_aux priceday  # 按正式预测器重放两分支的价格对照
MPLCONFIGDIR=/tmp/cumcm-mpl .venv/bin/python -m src.make_figures
.venv/bin/python -m src.make_tables
/opt/homebrew/bin/tectonic -X compile --keep-logs paper/CUMCMThesis-master/paper.tex
```

图形与表格只读 `results/latest/m30/` 的冻结实验，不重跑采购或执行模型。19 张矢量图输出到 `plot/`，当前论文使用的清单见 `plot/paper_figures.json`；按问分组的结果表、年度汇总和组件比较表输出到 `paper/CUMCMThesis-master/generated/`。主方法运行图与指定日表读取对应 48h 结果；预报时刻与组件比较明确使用 24h 基线。价格辅助数据按两个正式分支分别重放，其中 4-2 绑定历史时点因果净负荷特征。

### 保存完整精度实验

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m src.run_experiments --branch q2 --variant baseline --m 30 --name q2_baseline
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m src.run_experiments --branch q3 --variant candidates --m 30 --name q3_candidates --candidate-workers 5
```

长时全年任务可在每个完整日结束时原子保存检查点；中断后使用相同的模型、数据与日期参数恢复。检查点严格绑定源码/数据指纹，配置不一致会拒绝恢复：

```bash
.venv/bin/python -u -m src.run_experiments --branch q3 --variant legacy_means --m 30 --name q3_legacy_means --checkpoint results/latest/m30/checkpoints/q3_legacy_means.pkl --progress 1
.venv/bin/python -u -m src.run_experiments --branch q3 --variant legacy_means --m 30 --name q3_legacy_means --checkpoint results/latest/m30/checkpoints/q3_legacy_means.pkl --resume --progress 1
```

分支为 `q2/q3/q4_2/q4_3`；变体为 `baseline/48h/candidates/legacy_means/same_type`，问4另支持 `legacy_price/perfect`。每项生成 XLSX、逐日 JSON、全年 summary JSON、完整十分钟 NPZ；配置、数值源码与数据指纹均匹配且产物齐全时才复用。旧均值消融固定 K_load=6、K_pv=7，未在本轮重新调参。候选并行只分配五个独立候选，逐日库存演进仍顺序执行。

论文在当前 macOS 环境可用 `/opt/homebrew/bin/tectonic paper.tex` 编译（工作目录 `paper/CUMCMThesis-master`）。中文正文显式选取宋体集合中的 SC Regular 字形并嵌入 PDF，避免误选不可嵌入的 Black 字形。

## 实验产物与协作

| 位置 | 用途 | 合作者应如何使用 |
|---|---|---|
| `results/latest/m30/` | 当前冻结的最终 M=30 矩阵：22 组完整全年实验、`verification.json`、消融比较 JSON | 论文图表、统计和复核唯一读取此目录；不要混入临时试跑文件 |
| `results/latest/m30/*.summary.json` | 每组配置、源码/数据指纹、汇总费用与月度统计 | 先检查 `complete_year=true`，再引用费用 |
| `results/latest/m30/*.trace.npz` | 334 天 × 144 段完整轨迹 | 用于物理平衡、账单与图表复核 |
| `results/latest/m30/*.daily.json` 与 `*.xlsx` | 日度指标与便于人工查看的工作簿 | 用于日粒度分析和展示 |
| `results/legacy/` | 旧默认导出、历史日志、远端下载副本和被替代的试跑 | 仅供追溯，不作为论文最新结论的数据源 |

最终矩阵的 22 组由问2、问3、问4-2、问4-3各自的 `baseline/48h/candidates/legacy_means/same_type`，以及问4两个 `legacy_price` 对照组成。远端复用的 10 组保留各自产物内的原始源码指纹；它们与当前性能优化后的源码指纹不同，但已通过相同的 M=30、334 日、SOC、能量平衡与账单独立核验。运行 `PYTHONPATH=. .venv/bin/python .scratch/m30-review/validate_matrix.py` 可重新生成并检查 `results/latest/m30/verification.json`。

## 代码结构

```
src/
├── data.py              # 附件读取与全局常量（kW→kWh 转换）
├── optimizer.py         # 统一购电决策 LP（确定性/SAA/重优化）
├── forecast.py          # 分解式负载与三日光伏点预测（旧均值为增量）
├── forecast_pv.py       # 附件 3 整点光伏预报插值
├── forecast_price.py    # 电价预测（问 4）
├── scenarios.py         # 完整发布24h的配对残差场景库
├── value_dp.py          # 凸分段线性价值函数与 DP 递推
├── executor.py          # DP 价值执行器（逐段因果控制）
├── settlement.py        # 费用结算（问 2/问 3 公式）
├── results.py           # Excel 结果写出与摘要
├── tuning.py            # 超参数（K_L/K_P/K_p）网格搜索
├── parallel_eval.py     # 候选计划并行重评
├── make_figures.py      # 论文插图生成
├── run_q1.py            # 问 1 入口
├── run_q2.py            # 问 2 入口（全年 walk-forward）
├── run_q3.py            # 问 3 入口（滚动重优化）
└── run_q4.py            # 问 4 入口（波动电价）

data/                    # 赛题附件（附件 1–4）
results/latest/m30/      # 当前冻结的最终 22 组 M=30 结果
results/legacy/          # 旧导出、历史日志、远端副本与被替代试跑
tests/                   # 测试
plot/                    # 论文插图（PDF）
paper/                   # 论文 LaTeX 源码
docs/adr/                # 架构决策记录
CONTEXT.md               # 领域术语表
```

## 数据

赛题附件置于 `data/` 目录：

| 文件 | 内容 |
|---|---|
| `附件1.xlsx` | 问 1 的确定性负载、光伏、电价曲线 |
| `附件2.xlsx` | 问 2/3 全年每日负载、光伏（真值） |
| `附件3.xlsx` | 问 3 的 0/6/12/18 时整点光伏预报 |
| `附件4.xlsx` | 问 4 的全年波动电价 |

## 关键参数

| 参数 | 含义 | 选定值 |
|---|---|---|
| M | SAA 场景数 | 30（可用完整路径不足时退化） |
| 基线时域 | 每次发布后的完整 24h | 线性续存费用 |
| 增量时域 | 48h + 次日自由采购价值函数 | 须独立消融 |

参考基线固定复现完整稿：负载为“近3同类型形状×日电量切换倍率”，光伏为此前3日均值；M=30 条等权、完整实现且同发布时刻配对的路径。问2/4-2为24h线性续存价值，问3/4-3每次发布向前完整24h；问4-2使用净负荷回归且冻结采购、不读取附件3，问4-3使用 AR(1) 并可调整采购。问3、4-3按 B 口径对最终调整量结算。连续 DP 使用精确凸分段线性函数，不使用 SOC/action 网格或巨大终端罚项。48h+$V(e)$、候选重评、旧均值与同类型场景池均为显式增量；Q2/Q3/Q4均提供 `--horizon-days 1/2`、`--candidate-reeval`、`--legacy-means`、`--same-type-scenarios`，问4另有 `--legacy-price`。默认不调参；`--tune` 在问2/3选择旧负载/光伏窗口，在问4仅选择旧电价窗口。`--perfect-price` 是价格信息参照，非离线下界。官方附件5不在仓库，完整全年消融尚未运行。
