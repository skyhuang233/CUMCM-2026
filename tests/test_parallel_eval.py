import numpy as np

from src.parallel_eval import CandidateEvaluator
from src.value_dp import ConvexPiecewiseLinear


def test_process_candidate_evaluation_matches_serial_execution():
    """Parallel work returns the same ordered scores and DP value functions."""
    candidates = [np.full(4, amount) for amount in (0.0, 200.0, 500.0)]
    load = np.array([[1000.0, 700.0, 1200.0, 900.0], [800.0, 1500.0, 600.0, 1100.0]])
    pv = np.array([[100.0, 300.0, 200.0, 100.0], [200.0, 100.0, 400.0, 300.0]])
    price = np.array([0.5, 1.2, 0.7, 0.9])
    terminal = ConvexPiecewiseLinear.constant(0.0)

    serial = CandidateEvaluator(1)
    parallel = CandidateEvaluator(2)
    try:
        expected = serial.evaluate(candidates, load, pv, price, 6000.0, terminal)
        actual = parallel.evaluate(candidates, load, pv, price, 6000.0, terminal)
    finally:
        serial.close()
        parallel.close()

    assert np.allclose([score for score, _ in actual], [score for score, _ in expected])
    for (_, actual_hbar), (_, expected_hbar) in zip(actual, expected):
        for actual_fn, expected_fn in zip(actual_hbar, expected_hbar):
            assert np.allclose(actual_fn.breakpoints, expected_fn.breakpoints)
            assert np.allclose(actual_fn.values, expected_fn.values)
