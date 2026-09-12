# 实验结果索引

## 当前结论唯一来源

`latest/m30/` 是当前冻结的 M=30 全年实验矩阵。论文、图表脚本和独立校验都只读取此目录。

- 22 组完整实验：问2、问3、问4-2、问4-3各五个变体，外加问4的两个 `legacy_price` 对照。
- 每组包含同名的 `.summary.json`、`.daily.json`、`.trace.npz` 和 `.xlsx`。
- `verification.json` 是全矩阵的独立物理与账单复核记录；使用结果前应确认 `complete_count=22`、`missing=[]`。
- `ablation_comparison.json` 为生成论文消融图表的汇总输入。

复核命令：

```bash
PYTHONPATH=. .venv/bin/python .scratch/m30-review/validate_matrix.py
```

## 历史归档

`legacy/` 仅用于追溯，不是论文最新结论的数据源。

- `legacy/remote-restore/`：远端已验证结果的下载副本和原始归档结构。
- `legacy/old-m30-exports/`：早期命名的工作簿导出。
- `legacy/superseded-m30/`：未纳入最终矩阵的重复或被替代试跑。
- `legacy/old-m30-logs/` 与 `legacy/log/`：历史运行日志及队列 manifest。
- `legacy/result*.xlsx`：旧的默认入口导出。

最新实验与旧实验不得混放；新增正式矩阵产物应先写入 `latest/m30/`，并通过全矩阵校验后才能用于图表或论文。
