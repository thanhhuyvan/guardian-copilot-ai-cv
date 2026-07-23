from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "experiment_lightweight_improvements.py"
SPEC = importlib.util.spec_from_file_location("experiment_lightweight_improvements", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)


def robust_variant(window: int = 11, max_speed: float = 20.0):
    return EXPERIMENT.Variant("test", "official_median", window, "theil_sen", max_speed)


class CausalDepthPolicyTests(unittest.TestCase):
    def test_linear_approach_has_expected_ttc(self) -> None:
        policy = EXPERIMENT.CausalDepthPolicy(robust_variant())
        prediction = math.inf
        for index in range(11):
            timestamp = index * 0.05
            prediction = policy.update(timestamp, 10.0 - 5.0 * timestamp)
        self.assertAlmostEqual(prediction, 1.5, places=6)

    def test_pairwise_median_resists_single_mid_history_outlier(self) -> None:
        policy = EXPERIMENT.CausalDepthPolicy(robust_variant())
        prediction = math.inf
        for index in range(11):
            timestamp = index * 0.05
            depth = 10.0 - 5.0 * timestamp
            if index == 5:
                depth += 12.0
            prediction = policy.update(timestamp, depth)
        self.assertAlmostEqual(prediction, 1.5, places=6)

    def test_unphysical_closing_speed_is_rejected(self) -> None:
        policy = EXPERIMENT.CausalDepthPolicy(robust_variant(window=2, max_speed=20.0))
        self.assertTrue(math.isinf(policy.update(0.0, 10.0)))
        self.assertTrue(math.isinf(policy.update(0.05, 1.0)))

    def test_receding_depth_returns_infinity(self) -> None:
        policy = EXPERIMENT.CausalDepthPolicy(robust_variant(window=3))
        policy.update(0.0, 5.0)
        policy.update(0.05, 5.5)
        self.assertTrue(math.isinf(policy.update(0.10, 6.0)))

    def test_future_change_cannot_alter_prefix_predictions(self) -> None:
        prefix = [(index * 0.05, 10.0 - 2.0 * index * 0.05) for index in range(8)]
        future_a = [(0.40, 9.2), (0.45, 9.1), (0.50, 9.0)]
        future_b = [(0.40, 30.0), (0.45, 2.0), (0.50, 50.0)]

        def run(sequence):
            policy = EXPERIMENT.CausalDepthPolicy(robust_variant())
            return [policy.update(timestamp, depth) for timestamp, depth in sequence]

        predictions_a = run(prefix + future_a)
        predictions_b = run(prefix + future_b)
        self.assertEqual(predictions_a[: len(prefix)], predictions_b[: len(prefix)])


if __name__ == "__main__":
    unittest.main()
