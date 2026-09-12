"""Reproducible full-year experiment runner for the four approved branches."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .data import SOC_INIT, load_attachment4
from .forecast_price import K_PRICE
from .results import summarize_day, summarize_day_q3, write_result2, write_result3
from .run_q2 import Params as Q2Params
from .run_q2 import RECORD_START, WARMUP_START, load_bundle as load_q2, run_period as run_q2, validate as validate_q2
from .run_q3 import Params as Q3Params
from .run_q3 import load_bundle as load_q3, run_period as run_q3, validate as validate_q3
from .run_q4 import make_price_source
from .settlement import cost_q2, cost_q3

REPORT_DAYS = {"2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"}


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def _variant_params(branch: str, variant: str, m: int, k_load: int, k_pv: int):
    flags = {
        "horizon_days": 1,
        "candidate_reeval": False,
        "reference_baseline": True,
        "scenario_same_type": False,
    }
    if variant == "48h":
        flags["horizon_days"] = 2
    elif variant == "candidates":
        flags["candidate_reeval"] = True
    elif variant == "legacy_means":
        flags["reference_baseline"] = False
    elif variant == "same_type":
        flags["scenario_same_type"] = True
    elif variant not in {"baseline", "legacy_price", "perfect"}:
        raise ValueError(f"unsupported variant: {variant}")
    if branch in {"q2", "q4_2"}:
        return Q2Params(k_load=k_load, k_pv=k_pv, m_scen=m, **flags), flags
    return Q3Params(k_load=k_load, k_pv=k_pv, m_scen=m, **flags), flags


def _run(
    branch: str, variant: str, end: date, m: int, k_load: int, k_pv: int,
    k_price: int, progress: int, candidate_workers: int = 1, *,
    checkpoint_path: Path | None = None, resume: bool = False,
    checkpoint_tag: str | None = None,
):
    params, flags = _variant_params(branch, variant, m, k_load, k_pv)
    params.candidate_workers = candidate_workers
    if branch == "q2":
        bundle = load_q2()
        result = run_q2(WARMUP_START, end, params, bundle, record_from=RECORD_START,
                        soc_init=SOC_INIT, progress=progress,
                        checkpoint_path=checkpoint_path, resume=resume,
                        checkpoint_tag=checkpoint_tag)
        validate_q2(result, bundle)
        return result, bundle, params, None
    if branch == "q3":
        bundle = load_q3()
        result = run_q3(WARMUP_START, end, params, bundle, record_from=RECORD_START,
                        soc_init=SOC_INIT, progress=progress,
                        checkpoint_path=checkpoint_path, resume=resume,
                        checkpoint_tag=checkpoint_tag)
        validate_q3(result, bundle, params.issues)
        return result, bundle, params, None

    is_q3 = branch == "q4_3"
    bundle = load_q3() if is_q3 else load_q2()
    att4 = load_attachment4()
    source = make_price_source(
        att4, bundle.att1, k_price, variant == "perfect",
        branch="q4_3" if is_q3 else "q4_2",
        mode="legacy_level" if variant == "legacy_price" else "baseline",
        reference_baseline=flags["reference_baseline"],
    )
    runner = run_q3 if is_q3 else run_q2
    result = runner(WARMUP_START, end, params, bundle, record_from=RECORD_START,
                    soc_init=SOC_INIT, price_source=source, progress=progress,
                    checkpoint_path=checkpoint_path, resume=resume,
                    checkpoint_tag=checkpoint_tag)
    if is_q3:
        validate_q3(result, bundle, params.issues)
    else:
        validate_q2(result, bundle)
    return result, bundle, params, source


def _summary(result, branch: str, default_price: np.ndarray) -> dict:
    q3 = branch in {"q3", "q4_3"}
    monthly: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    reports = {}
    for day in result.days:
        month = day.day.strftime("%Y-%m")
        row = monthly[month]
        row["total_cost"] += day.total_cost
        row["plan_cost"] += day.plan_cost
        row["emergency_cost"] += day.emergency_cost
        row["emergency_kwh"] += float(day.E.sum())
        row["plan_kwh"] += float(day.G0.sum())
        if q3:
            row["curtail_cost"] += day.curtail_cost
            row["extra_cost"] += day.extra_cost
            row["adjust_kwh"] += day.adjust_kwh
        if day.day.isoformat() in REPORT_DAYS:
            reports[day.day.isoformat()] = (summarize_day_q3 if q3 else summarize_day)(
                day, default_price if day.price is None else day.price
            )
    total = {
        "total_cost": result.total_cost,
        "plan_cost": result.plan_cost,
        "emergency_cost": result.emergency_cost,
        "emergency_kwh": result.emergency_kwh,
        "emergency_days": result.emergency_days,
        "plan_kwh": result.plan_kwh,
        "soc_end": result.soc_end,
        "monthly": dict(monthly),
        "report_days": reports,
    }
    if q3:
        total.update({"curtail_cost": result.curtail_cost, "extra_cost": result.extra_cost,
                      "adjust_kwh": sum(day.adjust_kwh for day in result.days)})
    return total


def _write_trace(days, default_price: np.ndarray, path: Path) -> None:
    fields = ("G0", "C", "D", "S", "E", "W", "R")
    arrays = {field: np.stack([np.asarray(getattr(day, field)) for day in days]) for field in fields}
    arrays["Ga"] = np.stack([np.asarray(getattr(day, "Ga", day.G0)) for day in days])
    arrays["dates"] = np.array([day.day.isoformat() for day in days])
    arrays["price"] = np.stack([np.asarray(default_price if day.price is None else day.price) for day in days])
    np.savez_compressed(path, **arrays)


def _validate_artifacts(summary: dict, daily: list[dict], trace_path: Path, q3: bool) -> None:
    """Ensure the persisted arrays, daily rows, monthly rows and total agree."""
    trace = np.load(trace_path)
    n = len(daily)
    assert n > 0 and trace["G0"].shape == (n, 144)
    total = 0.0
    monthly = defaultdict(float)
    for i, row in enumerate(daily):
        price, g0, ga, emergency = trace["price"][i], trace["G0"][i], trace["Ga"][i], trace["E"][i]
        cost = cost_q3(price, g0, ga, emergency) if q3 else None
        expected = cost["total"] if q3 else sum(cost_q2(price, g0, emergency))
        assert abs(float(row["purchase_cost"]) - expected) < 1e-6
        total += expected
        monthly[str(row["date"])[:7]] += expected
    assert abs(total - float(summary["total_cost"])) < 1e-6
    for month, value in monthly.items():
        assert abs(value - float(summary["monthly"][month]["total_cost"])) < 1e-6


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run one reproducible microgrid experiment")
    parser.add_argument("--branch", required=True, choices=("q2", "q3", "q4_2", "q4_3"))
    parser.add_argument("--variant", default="baseline", choices=("baseline", "48h", "candidates", "legacy_means", "same_type", "legacy_price", "perfect"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default="results/latest/m30")
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--k-load", type=int, default=6)
    parser.add_argument("--k-pv", type=int, default=7)
    parser.add_argument("--k-price", type=int, default=K_PRICE)
    parser.add_argument("--progress", type=int, default=30)
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path,
                        help="atomic per-calendar-day checkpoint file")
    parser.add_argument("--resume", action="store_true",
                        help="resume a matching --checkpoint after interruption")
    args = parser.parse_args(argv)
    if args.variant in {"legacy_price", "perfect"} and args.branch not in {"q4_2", "q4_3"}:
        parser.error("legacy_price and perfect apply only to q4 branches")
    if not RECORD_START <= args.end <= date(2025, 12, 31):
        parser.error("end must be within 2025-02-01 through 2025-12-31")
    if args.m < 1:
        parser.error("m must be positive")
    if not 1 <= args.candidate_workers <= 5:
        parser.error("candidate-workers must be between 1 and 5")
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"{args.name}.summary.json"
    config = {key: getattr(args, key) for key in ("branch", "variant", "end", "m", "k_load", "k_pv", "k_price")}
    config["end"] = args.end.isoformat()
    numeric_sources = [
        "checkpoint.py", "data.py", "forecast.py", "forecast_pv.py", "forecast_price.py", "scenarios.py",
        "optimizer.py", "value_dp.py", "executor.py", "settlement.py", "parallel_eval.py",
        "run_q2.py", "run_q3.py", "run_q4.py", "run_experiments.py", "results.py",
    ]
    source_hash = _sha256([Path("src") / name for name in numeric_sources])
    data_hash = _sha256(sorted(Path("data").glob("*.xlsx")))
    fingerprint = hashlib.sha256(json.dumps({"config": config, "source": source_hash, "data": data_hash}, sort_keys=True).encode()).hexdigest()
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        required = [out / f"{args.name}{suffix}" for suffix in (".xlsx", ".daily.json", ".trace.npz")]
        if prior.get("complete_year") and prior.get("fingerprint") == fingerprint and all(path.exists() for path in required):
            print(f"reuse complete matching experiment: {summary_path}")
            return prior

    wall = time.perf_counter()
    result, bundle, params, source = _run(
        args.branch, args.variant, args.end, args.m, args.k_load, args.k_pv,
        args.k_price, args.progress, args.candidate_workers,
        checkpoint_path=args.checkpoint, resume=args.resume,
        checkpoint_tag=fingerprint,
    )
    q3 = args.branch in {"q3", "q4_3"}
    xlsx = out / f"{args.name}.xlsx"
    (write_result3 if q3 else write_result2)(result.days, str(xlsx))
    default_price = bundle.att1.price
    _write_trace(result.days, default_price, out / f"{args.name}.trace.npz")
    daily = [(summarize_day_q3 if q3 else summarize_day)(
        day, default_price if day.price is None else day.price
    ) for day in result.days]
    (out / f"{args.name}.daily.json").write_text(json.dumps(daily, ensure_ascii=False, indent=2, default=_jsonable))
    payload = {
        "fingerprint": fingerprint, "config": config, "params": _jsonable(params),
        "source_sha256": source_hash, "data_sha256": data_hash, "runtime_s": time.perf_counter() - wall,
        "initial_soc": SOC_INIT, "complete_year": args.end == date(2025, 12, 31) and len(result.days) == 334,
        "source": None if source is None else type(source).__name__,
        **_summary(result, args.branch, default_price),
    }
    _validate_artifacts(payload, daily, out / f"{args.name}.trace.npz", q3)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable))
    print(json.dumps({"summary": str(summary_path), "complete_year": payload["complete_year"], "total_cost": result.total_cost}))
    return payload


if __name__ == "__main__":
    main()
