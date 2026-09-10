"""附件读取与全局常量。

所有功率（kW）在读取时立即乘 DT_H 转成该段电量（kWh），此后全流程只用 kWh。
唯一例外是附件 3 的整点光伏预报，保持 kW 瞬时功率，留待后续单元按整点插值。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np
import openpyxl

# 时间与单位
T = 144  # 一天的段数（10 分钟一段）
DT_H = 1 / 6  # 段长（小时）：kW → kWh 的换算系数

# 储能参数（题目附录 1）
P_MAX_KWH = 5000 / 6  # 每段充/放电量上限（交流侧 5000 kW）
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_INIT = 6000.0  # 问 1 的 0:00 储电量
ETA = 0.9  # 充电乘 0.9、放电除以 0.9

EPS = 1e-4  # 目标中吞吐惩罚系数（元/kWh），用于排除退化解


def _seg_labels(n: int = T) -> list[str]:
    """段末时刻标签：'0:10' … '23:50', '0:00+1'。"""
    labels = [f"{(10 * (i + 1)) // 60}:{(10 * (i + 1)) % 60:02d}" for i in range(n - 1)]
    return labels + ["0:00+1"]


SEG_LABELS = _seg_labels()

# 四小时区间（表 2 的行）
INTERVAL_LABELS = [f"{h}:00-{h + 4}:00" for h in range(0, 24, 4)]
INTERVAL_SEGS = 24  # 每个四小时区间含 24 段


@dataclass
class Attachment1:
    """附件 1：问 1 的点预测曲线（电价 元/kWh，负载与光伏 kWh/段）。"""

    price: np.ndarray  # (T,)
    load_kwh: np.ndarray  # (T,)
    pv_kwh: np.ndarray  # (T,)


@dataclass
class DailySeries:
    """附件 2：全年逐日逐段的负载与光伏真值（kWh/段）。"""

    dates: list[date]
    load_kwh: np.ndarray  # (n_days, T)
    pv_kwh: np.ndarray  # (n_days, T)


def _sheet_rows(path: str, sheet: str | None = None) -> list[tuple]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    rows = [row for row in ws.iter_rows(values_only=True)]
    wb.close()
    return rows


def _day_matrix(rows: list[tuple], n_seg: int = T) -> tuple[list[date], np.ndarray]:
    """把「首列日期 + n_seg 列数值」的表读成日期列表与 (n_days, n_seg) 矩阵。"""
    dates: list[date] = []
    values = np.empty((len(rows) - 1, n_seg), dtype=float)
    for i, row in enumerate(rows[1:]):
        stamp = row[0]
        dates.append(stamp.date() if isinstance(stamp, datetime) else stamp)
        cells = row[1 : n_seg + 1]
        assert all(c is not None for c in cells), f"第 {i + 1} 行存在空值"
        values[i] = [float(c) for c in cells]
    return dates, values


def load_attachment1(path: str = "data/附件1.xlsx") -> Attachment1:
    """读附件 1：144 段的电价、小区负载、光伏发电预测功率。"""
    rows = _sheet_rows(path)[1:]
    assert len(rows) == T, f"附件 1 应有 {T} 段，实际 {len(rows)}"
    price = np.array([float(r[1]) for r in rows])
    load_kw = np.array([float(r[2]) for r in rows])
    pv_kw = np.array([float(r[3]) for r in rows])
    return Attachment1(price=price, load_kwh=load_kw * DT_H, pv_kwh=pv_kw * DT_H)


def load_attachment2(path: str = "data/附件2.xlsx") -> DailySeries:
    """读附件 2：全年逐日的小区负载与光伏发电实际功率。"""
    load_dates, load_kw = _day_matrix(_sheet_rows(path, "小区负载"))
    pv_dates, pv_kw = _day_matrix(_sheet_rows(path, "光伏发电实际功率"))
    assert load_dates == pv_dates, "附件 2 两张表的日期不一致"
    return DailySeries(dates=load_dates, load_kwh=load_kw * DT_H, pv_kwh=pv_kw * DT_H)


def load_attachment4(path: str = "data/附件4.xlsx") -> tuple[list[date], np.ndarray]:
    """读附件 4：全年逐日逐段的实际电价（元/kWh，不换算）。"""
    return _day_matrix(_sheet_rows(path))


def load_attachment3(path: str = "data/附件3.xlsx") -> dict[tuple[date, int], np.ndarray]:
    """读附件 3：预报时刻的未来 24 小时整点光伏预报。

    键为 (日期, 预报时刻小时 ∈ {0,6,12,18})，值为「预报 k 小时」k=1..24 的 kW 瞬时功率。
    日期列在同一天的后三行为空串，需前向填充。
    """
    rows = _sheet_rows(path)[1:]
    forecasts: dict[tuple[date, int], np.ndarray] = {}
    current: date | None = None
    for row in rows:
        cell = row[0]
        if isinstance(cell, datetime):
            current = cell.date()
        elif isinstance(cell, date):
            current = cell
        elif isinstance(cell, str) and cell.strip():
            current = date.fromisoformat("-".join(f"{int(p):02d}" for p in cell.split("-")))
        assert current is not None, "附件 3 首行日期缺失，无法前向填充"
        hour = int(str(row[1]).split(":")[0])
        forecasts[(current, hour)] = np.array([float(v) for v in row[2:26]])
    return forecasts


def day_type(d: date) -> int:
    """日类型：周五、周六为 1（高负载日），其余为 0。"""
    return 1 if d.weekday() in (4, 5) else 0


def to_kwh(power_kw: np.ndarray) -> np.ndarray:
    """kW 瞬时功率 → 段电量 kWh。"""
    return np.asarray(power_kw, dtype=float) * DT_H


__all__ = [
    "T",
    "DT_H",
    "P_MAX_KWH",
    "SOC_MIN",
    "SOC_MAX",
    "SOC_INIT",
    "ETA",
    "EPS",
    "SEG_LABELS",
    "INTERVAL_LABELS",
    "INTERVAL_SEGS",
    "Attachment1",
    "DailySeries",
    "load_attachment1",
    "load_attachment2",
    "load_attachment3",
    "load_attachment4",
    "day_type",
    "to_kwh",
]
