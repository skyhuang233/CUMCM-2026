"""论文补充数据（U1）：只读推导，产物写入 results/latest/paper_aux/。

不进 m30 矩阵、不改任何冻结结论数字。五个子命令：

  baseline    无储能直购基线（完美信息版 + 因果预测版），常数电价与附件 4 电价各一套
  lowerbound  完美信息离线下界：全年真值一次性 LP（储能可用、终端库存自由）
  valuefn     问 1 确定性价值函数 F_t(e) 折线与边际价值导出（命题 1 配图）
  forecastday 典型日的负载/光伏预测、M=30 配对场景与 H̄ 曲线导出（问 2 配图）
  priceday    典型日的电价预测 vs 真值导出（问 4 配图）
  all         依次全部运行

用法：.venv/bin/python -m src.paper_aux all
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path

import numpy as np

from .data import SOC_INIT, T, load_attachment1, load_attachment2, load_attachment4
from .forecast import PointForecaster
from .forecast_price import K_PRICE, PriceForecaster
from .optimizer import solve_deterministic
from .scenarios import M_SCEN, IssuedPathLibrary
from .value_dp import ConvexPiecewiseLinear, _free_step, future_cost, next_day_value

OUT_DIR = Path("results/latest/paper_aux")
RECORD_START = date(2025, 2, 1)
PERIOD_END = date(2025, 12, 31)
WARMUP_START = date(2025, 1, 1)
REPORT_DAYS = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]
TERMINAL_RATE = 0.481548  # 与 run_q2 基线一致的线性续存费率


def _fingerprint() -> dict:
    """产物指纹：本脚本与两份附件的 sha256，供 README 与复核。"""
    out = {}
    for label, p in [
        ("paper_aux.py", Path(__file__)),
        ("附件2.xlsx", Path("data/附件2.xlsx")),
        ("附件4.xlsx", Path("data/附件4.xlsx")),
    ]:
        out[label] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _write_json(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"已写出 {path}")
    return path


def _year_rows(dates: list[date], start: date = RECORD_START, end: date = PERIOD_END) -> list[int]:
    return [i for i, d in enumerate(dates) if start <= d <= end]


def _replay_library(daily, att1, until: date) -> tuple[PointForecaster, IssuedPathLibrary]:
    """把 run_q2 常数电价分支的 0:00 发布路径与残差库重放到 until（不含当天残差）。

    只复现预测层，与 LP / 执行无关，因此对 baseline/candidates 等变体通用。
    """
    forecaster = PointForecaster(daily, att1, reference_baseline=True)
    library = IssuedPathLibrary()
    row = {d: i for i, d in enumerate(daily.dates)}
    pending: list[tuple[datetime, np.ndarray, np.ndarray]] = []
    d = WARMUP_START
    while d <= until:
        asof = datetime.combine(d, clock_time.min)
        keep = []
        for issued, lp, vp in pending:
            if issued + timedelta(days=1) <= asof:
                nd = issued.date() + timedelta(days=1)
                if issued.date() != WARMUP_START and nd in row:
                    actual_l = daily.load_kwh[row[issued.date()]]
                    actual_v = daily.pv_kwh[row[issued.date()]]
                    library.update(issued.date(), 0, actual_l - lp, actual_v - vp, None)
            else:
                keep.append((issued, lp, vp))
        pending = keep
        load_pred, pv_pred = forecaster.predict(d)
        pending.append((asof, load_pred.copy(), pv_pred.copy()))
        d += timedelta(days=1)
    return forecaster, library


def run_baseline() -> None:
    """无储能直购基线，两种信息假设 × 两种电价。"""
    att1 = load_attachment1()
    daily = load_attachment2()
    p4_dates, p4 = load_attachment4()
    rows = _year_rows(daily.dates)
    row4 = {d: i for i, d in enumerate(p4_dates)}
    forecaster = PointForecaster(daily, att1, reference_baseline=True)

    def day_costs(i: int) -> dict:
        d = daily.dates[i]
        load, pv = daily.load_kwh[i], daily.pv_kwh[i]
        net_pos = np.maximum(load - pv, 0.0)
        lhat, vhat = forecaster.predict(d)
        g0 = np.maximum(lhat - vhat, 0.0)
        shortfall = np.maximum(load - pv - g0, 0.0)
        out = {}
        for tag, price in (("const", att1.price), ("att4", p4[row4[d]])):
            out[f"perfect_{tag}"] = float(price @ net_pos)
            out[f"forecast_{tag}"] = float(price @ g0 + 5.0 * price @ shortfall)
        return out

    totals: dict[str, float] = {}
    monthly: dict[str, dict[str, float]] = {}
    for i in rows:
        c = day_costs(i)
        mk = daily.dates[i].strftime("%Y-%m")
        for k, v in c.items():
            totals[k] = totals.get(k, 0.0) + v
            monthly.setdefault(mk, {})[k] = monthly.get(mk, {}).get(k, 0.0) + v
    _write_json("nostorage_baseline.json", {
        "说明": "无储能直购基线（2.1–12.31）。perfect=已知真值净负荷按需购电（下侧参照）；"
                "forecast=与主方法同一 0:00 因果预测定购、缺口按 5 倍电价紧急购电（可比参照）。"
                "const=附件1常数电价（问2/3 口径），att4=附件4真实电价（问4 口径）。富余电量按弃掉处理。",
        "totals": totals,
        "monthly": monthly,
        "fingerprint": _fingerprint(),
    })
    for k, v in sorted(totals.items()):
        print(f"  {k:16} {v:16.2f} 元")


def run_lowerbound() -> None:
    """完美信息离线下界：真值曲线上的全年单次 LP（含储能，终端库存自由）。

    任何因果策略的实际账单不低于该值：策略轨迹本身是该 LP 的可行解，
    而紧急购电（5 倍）与问 3 调整费只会更贵。
    """
    att1 = load_attachment1()
    daily = load_attachment2()
    p4_dates, p4 = load_attachment4()
    rows = _year_rows(daily.dates)
    row4 = {d: i for i, d in enumerate(p4_dates)}
    load = np.concatenate([daily.load_kwh[i] for i in rows])
    pv = np.concatenate([daily.pv_kwh[i] for i in rows])
    out = {"说明": "完美信息离线下界（2.1–12.31 一次 LP，初始库存 6000，终端库存自由，"
                   "普通电价购电、无紧急/调整费）。const=附件1常数电价，att4=附件4真实电价。",
           "n_seg": int(load.size)}
    for tag, price in (
        ("const", np.tile(att1.price, len(rows))),
        ("att4", np.concatenate([p4[row4[daily.dates[i]]] for i in rows])),
    ):
        res = solve_deterministic(price, load, pv, SOC_INIT, periodic=False)
        out[f"lower_bound_{tag}"] = res.purchase_cost
        print(f"  lower_bound_{tag} {res.purchase_cost:16.2f} 元")
    out["fingerprint"] = _fingerprint()
    _write_json("perfect_lowerbound.json", out)


def _polyline(fn: ConvexPiecewiseLinear) -> dict:
    return {"x": fn.x.tolist(), "y": fn.y.tolist(), "slopes": fn.slopes.tolist()}


def run_valuefn() -> None:
    """问 1 的 F_t(e) 折线族与边际价值，命题 1 的配图数据。"""
    att = load_attachment1()
    values: list[ConvexPiecewiseLinear] = [None] * (T + 1)
    values[T] = ConvexPiecewiseLinear.point(SOC_INIT)
    for t in range(T - 1, -1, -1):
        values[t] = _free_step(values[t + 1], att.price[t], att.load_kwh[t] - att.pv_kwh[t])
    picks = {t: _polyline(values[t]) for t in (0, 36, 72, 108, 132)}
    seg_counts = [len(values[t].x) for t in range(T + 1)]
    _write_json("q1_value_functions.json", {
        "说明": "问 1 确定性价值函数 F_t(e)（终端为 S_144=6000 的单点约束）。"
                "picks 键为段号 t（0=0:00, 36=6:00, 72=12:00, 108=18:00, 132=22:00），"
                "x 为库存断点（kWh）、y 为最低剩余费用（元）、slopes 为区间斜率（-边际价值）。",
        "picks": {str(k): v for k, v in picks.items()},
        "n_breakpoints_by_t": seg_counts,
        "F0_at_6000": float(values[0](SOC_INIT)),
        "fingerprint": _fingerprint(),
    })
    print(f"  F_0(6000) = {values[0](SOC_INIT):.4f} 元（应与问 1 LP 目标一致）")


def run_forecastday(target: date) -> None:
    """典型日：0:00 预测、M=30 配对场景、当日 H̄_t 曲线样本（用冻结 G0）。"""
    att1 = load_attachment1()
    daily = load_attachment2()
    row = {d: i for i, d in enumerate(daily.dates)}
    forecaster, library = _replay_library(daily, att1, target)
    load_pred, pv_pred = forecaster.predict(target)
    L, PV, _ = library.scenarios(datetime.combine(target, clock_time.min), 0,
                                 load_pred, pv_pred, None, M_SCEN)
    z = np.load("results/latest/m30/q2_candidates.trace.npz", allow_pickle=True)
    dates = list(z["dates"])
    di = dates.index(target.isoformat())
    g0 = z["G0"][di]
    terminal = ConvexPiecewiseLinear.constant(0.0).add_linear(-TERMINAL_RATE)
    hbar = future_cost(att1.price, L, PV, g0, terminal)
    picks = {t: _polyline(hbar[t]) for t in (6, 36, 72, 108, 138)}
    np.savez_compressed(
        OUT_DIR / f"forecast_day_{target.isoformat()}.npz",
        load_pred=load_pred, pv_pred=pv_pred,
        load_true=daily.load_kwh[row[target]], pv_true=daily.pv_kwh[row[target]],
        L_scen=L, PV_scen=PV, G0=g0,
        R=z["R"][di], S=z["S"][di], E=z["E"][di],
    )
    print(f"已写出 {OUT_DIR / f'forecast_day_{target.isoformat()}.npz'}"
          f"（场景 {L.shape[0]} 条）")
    _write_json(f"hbar_day_{target.isoformat()}.json", {
        "说明": f"{target} 的场景均值未来费用函数 H̄_t(e) 样本（冻结 G0 + 重放场景 + "
                "基线线性续存 v=0.481548），键为段号 t。执行层保留水平 R 存于同名 npz。",
        "picks": {str(k): v for k, v in picks.items()},
        "fingerprint": _fingerprint(),
    })


def run_priceday(targets: list[date]) -> None:
    """问 4 典型日：0:00 电价点预测 vs 附件 4 真值。"""
    att1 = load_attachment1()
    daily = load_attachment2()
    p4_dates, p4 = load_attachment4()
    forecaster = PointForecaster(daily, att1, reference_baseline=True)
    ps = PriceForecaster(p4_dates, p4, att1, k=K_PRICE, branch="q4_2")

    def net(d: date) -> float:
        lh, vh = forecaster.predict(d)
        return float(np.sum(lh - vh))

    out = {}
    row4 = {d: i for i, d in enumerate(p4_dates)}
    for d in targets:
        today_p, _ = ps.predict(d, 0, net_load_pred=net(d), net_load_next=net(d + timedelta(days=1)))
        truth = p4[row4[d]]
        out[d.isoformat()] = {
            "pred": np.asarray(today_p, float).tolist(),
            "truth": truth.tolist(),
            "mae": float(np.mean(np.abs(np.asarray(today_p) - truth))),
        }
    _write_json("price_forecast_days.json", {
        "说明": "问 4 电价预测（0:00 发布，K_p=35 基线管线）与附件 4 真值对照，含逐日 MAE。",
        "days": out,
        "fingerprint": _fingerprint(),
    })
    for d, v in out.items():
        print(f"  {d} 电价 MAE = {v['mae']:.4f} 元/kWh")


def _write_readme() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README.md").write_text(
        "# 论文补充数据（paper_aux）\n\n"
        "由 `src/paper_aux.py` 从附件真值、冻结 trace 与因果预测重放**只读推导**，"
        "供论文图表与命题使用；不属于 m30 实验矩阵，不改变任何冻结结论数字。\n\n"
        "| 文件 | 内容 | 消费方 |\n|---|---|---|\n"
        "| nostorage_baseline.json | 无储能直购基线（perfect/forecast × const/att4） | 结果定位区间 |\n"
        "| perfect_lowerbound.json | 完美信息离线下界（全年一次 LP） | 结果定位区间 |\n"
        "| q1_value_functions.json | 问 1 F_t(e) 折线与边际价值 | 命题 1 / 图 F2 |\n"
        "| forecast_day_*.npz | 典型日预测、场景、冻结 G0/R/S/E | 图 F4/F5/F6 |\n"
        "| hbar_day_*.json | 典型日 H̄_t(e) 折线样本 | 图 F6 |\n"
        "| price_forecast_days.json | 问 4 电价预测 vs 真值 | 图 F10 |\n\n"
        "重建：`.venv/bin/python -m src.paper_aux all`（每个 json 内含脚本与数据指纹）。\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="论文补充数据（只读推导）")
    parser.add_argument("cmd", choices=("baseline", "lowerbound", "valuefn",
                                        "forecastday", "priceday", "all"))
    parser.add_argument("--date", type=date.fromisoformat, default=REPORT_DAYS[0])
    args = parser.parse_args(argv)
    _write_readme()
    if args.cmd in ("baseline", "all"):
        run_baseline()
    if args.cmd in ("lowerbound", "all"):
        run_lowerbound()
    if args.cmd in ("valuefn", "all"):
        run_valuefn()
    if args.cmd in ("forecastday", "all"):
        targets = REPORT_DAYS if args.cmd == "all" else [args.date]
        for t in targets:
            run_forecastday(t)
    if args.cmd in ("priceday", "all"):
        run_priceday(REPORT_DAYS)


if __name__ == "__main__":
    main()
