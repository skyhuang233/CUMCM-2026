"""Process-parallel evaluation of independent purchase-plan candidates.

The walk-forward state of a backtest (SOC and residual libraries) remains
strictly sequential.  The five convex-combination candidates at a decision
time, however, are fully independent and expensive enough to justify process
parallelism.  Keeping an executor alive for the whole run amortises worker
startup over all days and forecast blocks.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Iterable

import numpy as np

from .value_dp import evaluate_plan


def _evaluate_candidate(args):
    """Pickle-friendly process entry point; preserves ``evaluate_plan`` API."""
    candidate, load_scen, pv_scen, price, soc_init, terminal_value, price_scen, base_plan, n_fixed = args
    return evaluate_plan(
        candidate, load_scen, pv_scen, price, soc_init, terminal_value,
        price_scen=price_scen, base_plan=base_plan, n_fixed=n_fixed, return_hbar=True,
    )


class CandidateEvaluator:
    """Reusable candidate evaluator.

    ``workers=1`` is deliberately the default: it preserves deterministic,
    low-overhead behaviour for short runs and tests.  Positive values above
    one start spawn-based workers, avoiding unsafe fork interactions with the
    LP/BLAS stack on macOS.
    """

    def __init__(self, workers: int = 1):
        requested_workers = int(workers)
        if requested_workers < 1:
            raise ValueError("candidate_workers must be at least 1")
        # The policy currently supplies five convex combinations.  Do not
        # create idle child processes when a caller passes the machine's full
        # core count; those cores are better left for another full backtest.
        self.workers = min(requested_workers, 5)
        self._pool = (
            ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"))
            if self.workers > 1 else None
        )

    def evaluate(
        self,
        candidates: Iterable[np.ndarray],
        load_scen,
        pv_scen,
        price,
        soc_init: float,
        terminal_value=None,
        *,
        price_scen=None,
        base_plan=None,
        n_fixed=None,
    ) -> list[tuple[float, list]]:
        """Return ordered ``(score, future_cost_functions)`` pairs.

        At most five candidate tasks exist in the current policies, so callers
        should reserve remaining cores for independent full-year backtests
        rather than requesting more than five workers for one run.
        ``terminal_value`` must be pickleable when workers exceed one; the
        production callers use ``ConvexPiecewiseLinear``, which is pickleable.
        """
        tasks = [
            (candidate, load_scen, pv_scen, price, soc_init, terminal_value,
             price_scen, base_plan, n_fixed)
            for candidate in candidates
        ]
        if self._pool is None or len(tasks) < 2:
            return [_evaluate_candidate(task) for task in tasks]
        return list(self._pool.map(_evaluate_candidate, tasks))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __del__(self):  # defensive cleanup for exceptions in a backtest
        if getattr(self, "_pool", None) is not None:
            self.close()
