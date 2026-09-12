# 论文补充数据（paper_aux）

由 `src/paper_aux.py` 从附件真值、冻结 trace 与因果预测重放**只读推导**，供论文图表与命题使用；不属于 m30 实验矩阵，不改变任何冻结结论数字。

| 文件 | 内容 | 消费方 |
|---|---|---|
| nostorage_baseline.json | 无储能直购基线（perfect/forecast × const/att4） | 结果定位区间 |
| perfect_lowerbound.json | 完美信息离线下界（全年一次 LP） | 结果定位区间 |
| q1_value_functions.json | 问 1 F_t(e) 折线与边际价值 | 命题 1 / 图 F2 |
| forecast_day_*.npz | 典型日预测、场景、冻结 G0/R/S/E | 图 F4/F5/F6 |
| hbar_day_*.json | 典型日 H̄_t(e) 折线样本 | 图 F6 |
| price_forecast_days.json | 问 4 电价预测 vs 真值 | 图 F10 |

重建：`.venv/bin/python -m src.paper_aux all`（每个 json 内含脚本与数据指纹）。
