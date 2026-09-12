"""Bounded queue for the remaining M30 full-year experiments.

The queue is intentionally a foreground process: it starts ordinary Python
children and keeps going when one child fails.  This makes it suitable for a
long-lived Windows SSH session without depending on shell-specific daemons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from .forecast_price import K_PRICE
from .run_experiments import _sha256

END = date(2025, 12, 31)
TASK_VARIANTS = {
    "q3": ("48h", "legacy_means", "same_type"),
    "q4_2": ("legacy_means", "same_type", "legacy_price"),
    "q4_3": ("48h", "legacy_means", "same_type", "legacy_price"),
}
NUMERIC_SOURCES = [
    "checkpoint.py", "data.py", "forecast.py", "forecast_pv.py", "forecast_price.py", "scenarios.py",
    "optimizer.py", "value_dp.py", "executor.py", "settlement.py", "parallel_eval.py",
    "run_q2.py", "run_q3.py", "run_q4.py", "run_experiments.py", "results.py",
]


def task_specs() -> list[dict[str, Any]]:
    return [
        {"name": f"{branch}_{variant}", "branch": branch, "variant": variant}
        for branch, variants in TASK_VARIANTS.items()
        for variant in variants
    ]


def _fingerprint(spec: dict[str, Any], root: Path) -> str:
    config = {
        "branch": spec["branch"], "variant": spec["variant"], "end": END.isoformat(),
        "m": 30, "k_load": 6, "k_pv": 7, "k_price": K_PRICE,
    }
    # Match run_experiments exactly: its digest includes relative path names.
    source = _sha256([Path("src") / name for name in NUMERIC_SOURCES])
    data = _sha256(sorted(Path("data").glob("*.xlsx")))
    return hashlib.sha256(json.dumps({"config": config, "source": source, "data": data}, sort_keys=True).encode()).hexdigest()


def _inspect_existing(spec: dict[str, Any], out: Path, root: Path) -> tuple[str, dict[str, Any] | None, str | None]:
    """Return (reuse|launch|conflict, summary, reason) without changing files."""
    summary_path = out / f"{spec['name']}.summary.json"
    if not summary_path.exists():
        return "launch", None, None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return "conflict", None, f"unreadable summary: {exc}"
    cfg = summary.get("config", {})
    expected = {"branch": spec["branch"], "variant": spec["variant"], "end": END.isoformat(), "m": 30, "k_load": 6, "k_pv": 7, "k_price": K_PRICE}
    mismatch = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
    required = [out / f"{spec['name']}{suffix}" for suffix in (".xlsx", ".daily.json", ".trace.npz")]
    if summary.get("complete_year"):
        if mismatch:
            return "conflict", summary, f"completed summary config mismatch: {mismatch}"
        if summary.get("fingerprint") != _fingerprint(spec, root):
            return "conflict", summary, "completed summary fingerprint does not match current sources/data"
        if not all(p.exists() for p in required):
            return "conflict", summary, "completed summary is missing required artifacts"
        return "reuse", summary, None
    return "launch", summary, "existing result incomplete or stale"


def _run_one(spec: dict[str, Any], out: Path, log_dir: Path, root: Path) -> dict[str, Any]:
    started = time.time()
    log_path = log_dir / f"{spec['name']}.log"
    checkpoint = out / "checkpoints" / f"{spec['name']}.pkl"
    cmd = [
        sys.executable, "-u", "-m", "src.run_experiments",
        "--branch", spec["branch"], "--variant", spec["variant"],
        "--name", spec["name"], "--out-dir", str(out), "--end", END.isoformat(),
        "--m", "30", "--k-load", "6", "--k-pv", "7", "--k-price", str(K_PRICE),
        "--candidate-workers", "1", "--progress", "1", "--checkpoint", str(checkpoint),
    ]
    if checkpoint.exists():
        cmd.append("--resume")
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env)
        pid = proc.pid
        rc = proc.wait()
    summary_path = out / f"{spec['name']}.summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            pass
    return {"status": "succeeded" if rc == 0 and summary and summary.get("complete_year") else "failed", "returncode": rc, "pid": pid, "summary": summary, "log": str(log_path), "started": started, "ended": time.time()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the remaining bounded M30 experiment queue")
    parser.add_argument("--out-dir", default="results/latest/m30")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_workers <= 32:
        parser.error("max-workers must be between 1 and 32")
    root = Path.cwd()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out / "remote-queue-logs"; log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out / "remote_queue_manifest.json"
    specs = task_specs()
    manifest = {"queue": "m30-remote", "created": time.time(), "max_workers": args.max_workers, "tasks": []}
    launch: list[dict[str, Any]] = []
    for spec in specs:
        action, summary, reason = _inspect_existing(spec, out, root)
        item = {**spec, "config": {"end": END.isoformat(), "m": 30, "k_load": 6, "k_pv": 7, "k_price": K_PRICE, "candidate_workers": 1}, "status": "planned"}
        if action == "reuse":
            item.update(status="reused", summary_identity={"fingerprint": summary.get("fingerprint"), "complete_year": summary.get("complete_year"), "total_cost": summary.get("total_cost")})
        elif action == "conflict":
            item.update(status="conflict", error=reason)
        elif args.dry_run:
            item.update(status="dry-run", reason=reason)
        else:
            item["status"] = "queued"; launch.append(item)
        manifest["tasks"].append(item)
    if not args.dry_run and launch:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(_run_one, item, out, log_dir, root): item for item in launch}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result(); item.update(result, summary_identity=None if not result.get("summary") else {"fingerprint": result["summary"].get("fingerprint"), "complete_year": result["summary"].get("complete_year"), "total_cost": result["summary"].get("total_cost")})
                except Exception as exc:
                    item.update(status="failed", error=repr(exc), ended=time.time())
        manifest["finished"] = time.time()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({"manifest": str(manifest_path), "tasks": [{"name": t["name"], "status": t["status"]} for t in manifest["tasks"]]}, ensure_ascii=False))
    return 1 if any(t["status"] in {"failed", "conflict"} for t in manifest["tasks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
