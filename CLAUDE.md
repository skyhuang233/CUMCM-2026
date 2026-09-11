# 微网购电与储能调度（2026 国赛 C 题）

## 项目概况

带储能的非孤岛微网日前购电计划、滚动调整与执行结算。四问分别覆盖确定性、随机、多预报时点、波动电价场景。全年（2.1–12.31）walk-forward 回测，1 月为冷启动预热。

## 环境

- Python 3.13，虚拟环境 `.venv/bin/python`
- 依赖管理：`uv`（`uv pip install -e .`）
- 测试：`pytest`（122 个测试，约 12 秒）

## 运行

```bash
# 问 1：确定性日前计划
.venv/bin/python -m src.run_q1

# 问 2：SAA 购电 + DP 执行（全年约 5 分钟）
.venv/bin/python -m src.run_q2

# 问 3：滚动重优化（--issues 控制预报时点子集）
.venv/bin/python -m src.run_q3
.venv/bin/python -m src.run_q3 --issues 0       # 仅 0:00
.venv/bin/python -m src.run_q3 --issues 0 6 12 18

# 问 4：波动电价
.venv/bin/python -m src.run_q4           # result4-2
.venv/bin/python -m src.run_q4 --q3      # result4-3
.venv/bin/python -m src.run_q4 --perfect-price  # 完美电价对照

# 插图
.venv/bin/python -m src.make_figures
```

## 架构

三层结构，四问共用同一套组件：

| 层 | 文件 | 职责 |
|---|---|---|
| 优化核 | `optimizer.py` | 统一 LP（确定性 / SAA / 重优化），稀疏矩阵装配 |
| 预测/场景 | `forecast.py`, `forecast_pv.py`, `forecast_price.py`, `scenarios.py` | 点预测与经验残差场景生成 |
| 执行 | `executor.py`, `value_dp.py` | DP 价值执行器：未来费用函数 → 动态保留水平 → 逐段因果控制 |
| 结算 | `settlement.py` | 费用分解（问 2 / 问 3 公式不同） |
| 数据 | `data.py` | 附件读取，全局常量（kW → kWh 转换在此完成） |
| 结果 | `results.py` | Excel 写出与摘要打印 |
| 参数选择 | `tuning.py` | K_L / K_P / K_p 网格搜索 |

## 术语

使用 `CONTEXT.md` 中定义的领域语言。关键术语速查：
- **计划购电量** G⁰：0:00 冻结的每段购电量
- **调整购电量** Gᵃ：问 3 起重优化后提交的购电量
- **紧急购电量** E：5 倍电价补缺，禁止用于充电
- **未来费用函数** H̄ₜ(e)：场景均值的剩余段最低费用
- **动态保留水平** Rₜ：缺口时不再放电的 SOC 下限

## 约束

- **不要用 2–12 月数据选参数**——构成信息泄漏，只能用 1 月实际费用选。
- **执行层用 DP 价值执行器**，不要换回储能滚动 LP（ADR `docs/adr/0001-dp-value-executor.md` 记录了原因和数据）。
- **优化核变量命名**遵循 `VarLayout` 的块名约定（`G`, `C`, `D`, `S`, `E`, `W`, `dP`, `dN`），不要重命名。
- `results/` 和 `.scratch/` 已 gitignore，是产出物和草稿。
- 所有功率在 `data.py` 读取时已转为 kWh，全流程不再出现 kW。
