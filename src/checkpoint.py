"""Atomic, versioned checkpoints for sequential walk-forward backtests.

The checkpoint payload deliberately contains only the state that evolves with
the walk-forward clock: submitted-result days, SOC, issued-but-not-delivered
paths, and the residual-path library.  Input workbooks and deterministic
forecasters are rebuilt on resume, so a checkpoint cannot silently carry a
different input data bundle into a new run.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
import os
from pathlib import Path
import pickle
from typing import Any


_SCHEMA = 1


def identity(
    kind: str,
    start: date,
    end: date,
    record_from: date,
    params: Any,
    *,
    varies_price: bool,
    tag: str | None = None,
) -> dict[str, Any]:
    """Return the complete compatibility key for a resumable run."""
    return {
        "schema": _SCHEMA,
        "kind": kind,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "record_from": record_from.isoformat(),
        "params": asdict(params) if is_dataclass(params) else dict(params),
        "varies_price": bool(varies_price),
        # The experiment runner supplies its source/data fingerprint here.
        # Direct callers retain strict configuration compatibility without
        # pretending that an unknown external source revision is identical.
        "tag": tag,
    }


def load(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any]:
    """Load a trusted local checkpoint and reject incompatible runs early."""
    try:
        with path.open("rb") as stream:
            state = pickle.load(stream)
    except (OSError, pickle.PickleError, EOFError, AttributeError) as exc:
        raise ValueError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("identity") != expected_identity:
        raise ValueError(
            f"checkpoint {path} does not match this experiment's configuration, "
            "data/source fingerprint, or date range"
        )
    if not isinstance(state.get("last_day"), date):
        raise ValueError(f"checkpoint {path} has no valid last completed day")
    return state


def save(path: Path, *, run_identity: dict[str, Any], last_day: date,
         soc: float, result: Any, pending: Any, paths: Any,
         elapsed_s: float) -> None:
    """Persist a completed calendar-day boundary atomically.

    A temporary sibling and ``os.replace`` mean an interruption leaves either
    the preceding valid checkpoint or the whole new one, never a partial
    pickle.  Checkpoints are local trusted artifacts; pickle is appropriate
    here because the state includes project dataclasses and NumPy arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "identity": run_identity,
        "last_day": last_day,
        "soc": float(soc),
        "result": result,
        "pending": pending,
        "paths": paths,
        "elapsed_s": float(elapsed_s),
    }
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # ``replace`` removes it on success; this only cleans an interrupted
        # write before it could become a valid checkpoint.
        if temporary.exists():
            temporary.unlink()

