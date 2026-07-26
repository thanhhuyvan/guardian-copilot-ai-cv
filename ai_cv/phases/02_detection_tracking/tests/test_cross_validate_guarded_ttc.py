from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
MODULE_PATH = SRC / "cross_validate_guarded_ttc.py"
SPEC = importlib.util.spec_from_file_location(
    "cross_validate_guarded_ttc",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
CV = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CV
SPEC.loader.exec_module(CV)


class GuardCrossValidationTests(unittest.TestCase):
    def test_grid_contains_the_current_guard(self) -> None:
        configs = CV.guard_grid()

        index = CV.current_guard_index(configs)

        self.assertEqual(configs[index], CV.GuardConfig(**CV.CURRENT_GUARD))
        self.assertEqual(len(configs), 2916)

    def test_corridor_widens_toward_image_bottom(self) -> None:
        centers = np.asarray([0.70, 0.70])
        bottoms = np.asarray([0.36, 1.0])

        accepted = CV.corridor_membership(centers, bottoms, 0.10, 0.50)

        np.testing.assert_array_equal(accepted, [False, True])

    def test_prediction_uses_minimum_ttc_of_accepted_tracks(self) -> None:
        data = CV.TripData(
            trip_id="T01-Sample",
            frame_ids=np.asarray([0, 1]),
            ground_truth=np.asarray([1.0, math.inf]),
            candidate_frame_index=np.asarray([0, 0, 1]),
            center_x=np.asarray([0.5, 0.5, 0.9]),
            bottom_y=np.asarray([0.8, 0.8, 0.8]),
            height=np.asarray([0.2, 0.2, 0.2]),
            confidence=np.asarray([0.9, 0.9, 0.9]),
            closing_speed=np.asarray([4.0, 5.0, 4.0]),
            depth=np.asarray([8.0, 5.0, 8.0]),
            residual=np.asarray([0.1, 0.1, 0.1]),
            ttc=np.asarray([2.0, 1.0, 1.0]),
        )

        predictions = CV.predict(data, CV.GuardConfig(**CV.CURRENT_GUARD))

        self.assertEqual(predictions[0], 1.0)
        self.assertTrue(math.isinf(predictions[1]))

    def test_f1_and_confusion_are_computed_at_two_seconds(self) -> None:
        metrics = CV.score(
            np.asarray([1.0, 1.5, math.inf, math.inf]),
            np.asarray([1.0, math.inf, 1.5, math.inf]),
        )

        self.assertEqual((metrics.tp, metrics.fp, metrics.fn), (1, 1, 1))
        self.assertAlmostEqual(metrics.f1, 0.5)


if __name__ == "__main__":
    unittest.main()
