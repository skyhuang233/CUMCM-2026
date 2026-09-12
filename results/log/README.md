# 2025 全年回测实验日志

本目录保存本次四条并行全年回测的完整标准输出日志。

运行配置：

- 回测区间：`2025-01-01` 至 `2025-12-31`，统计区间从 `2025-02-01` 开始
- 场景数：`M=12`
- 候选评估进程：每条回测 `--candidate-workers 2`
- 原生数值库线程：`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`
- Q2：`--skip-tuning`
- Q3：`--skip-tuning`，预报时刻 `0,6,12,18`
- Q4-2/Q4-3：`--skip-tuning`，电价参数 `K_p=4`

日志文件：

- `q2_2025_full.log`
- `q3_2025_full.log`
- `q4_2_2025_full.log`
- `q4_3_2025_full.log`

对应结果文件位于上级 `results/` 目录：`result2.xlsx`、`result3.xlsx`、`result4-2.xlsx`、`result4-3.xlsx`。
