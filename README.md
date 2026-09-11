# 微网购电与储能调度优化

2026 年全国大学生数学建模竞赛 C 题：带储能的非孤岛微网向外网购电与储能充放电的日前计划、滚动调整与执行结算。

## 方法概要

采用 **SAA 两阶段购电 LP + DP 价值执行器** 的两层架构：

1. **决策层**：每日 0:00 用经验残差场景的样本平均近似（SAA）线性规划冻结计划购电量；候选计划经 DP 执行器重评后取费用最优者。
2. **执行层**：购电量固定后，用同批场景反推的未来费用函数得到每段动态保留水平，逐段按当段真值决定充放电——富余尽量充，缺口只放到保留水平为止，紧急购电补足余额。
3. **问 3 扩展**：在 6:00/12:00/18:00 用新光伏预报滚动重优化，只提交到下一预报时刻。
4. **问 4 扩展**：电价由近期同类型日均值预测，残差与负载/光伏取同日保留相关性。

全年 walk-forward 回测，1 月冷启动预热，2–12 月为正式结果。

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

# 问 2：SAA + DP 执行全年回测（约 5 分钟）
.venv/bin/python -m src.run_q2

# 问 3：滚动重优化（约 8 分钟）
.venv/bin/python -m src.run_q3

# 问 4：波动电价（分别产出 result4-2 和 result4-3）
.venv/bin/python -m src.run_q4
.venv/bin/python -m src.run_q4 --q3
```

结果输出到 `results/` 目录（.xlsx 格式）。

### 运行测试

```bash
.venv/bin/python -m pytest      # 122 个测试
```

### 生成论文插图

```bash
.venv/bin/python -m src.make_figures   # PDF 输出到 plot/
```

## 代码结构

```
src/
├── data.py              # 附件读取与全局常量（kW→kWh 转换）
├── optimizer.py         # 统一购电决策 LP（确定性/SAA/重优化）
├── forecast.py          # 负载与光伏点预测（K 近邻同类型日均值）
├── forecast_pv.py       # 附件 3 整点光伏预报插值
├── forecast_price.py    # 电价预测（问 4）
├── scenarios.py         # 经验残差场景库（同类型日、同日三通道）
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
results/                 # 输出结果（gitignored）
tests/                   # 测试（122 个）
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
| K_L | 负载预测近邻数 | 6 |
| K_P | 光伏预测近邻数 | 7 |
| K_p | 电价预测近邻数（问 4） | 8 |
| M | SAA 场景数 | 12 |
| 时域 | 优化窗口 | 48h（当天 + 次日） |

参数仅用 1 月实际费用选定，避免信息泄漏。
