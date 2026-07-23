from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "analyze_stereo_confidence.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_stereo_confidence", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


class LeftRightConsistencyTests(unittest.TestCase):
    def test_consistent_correspondence_passes(self) -> None:
        left = np.full((2, 8), -1.0, dtype=np.float32)
        right = np.full((2, 8), -97.0, dtype=np.float32)
        left[:, 3:] = 2.0
        right[:, 1:6] = -2.0

        valid, consistent, residual = ANALYSIS.left_right_consistency(left, right)

        self.assertTrue(np.all(valid[:, 3:]))
        self.assertTrue(np.all(consistent[:, 3:]))
        self.assertTrue(np.allclose(residual[:, 3:], 0.0))

    def test_mismatch_and_out_of_bounds_are_rejected(self) -> None:
        left = np.array([[4.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        right = np.array([[-4.0, -4.0, -4.0, -4.0]], dtype=np.float32)

        _, consistent, residual = ANALYSIS.left_right_consistency(left, right)

        self.assertFalse(consistent[0, 0])  # x_right = -4
        self.assertFalse(consistent[0, 2])  # residual |2 + -4| = 2
        self.assertTrue(math.isnan(float(residual[0, 0])))

    def test_threshold_is_inclusive(self) -> None:
        left = np.array([[1.0, 1.0]], dtype=np.float32)
        right = np.array([[-2.0, -2.0]], dtype=np.float32)
        _, consistent, _ = ANALYSIS.left_right_consistency(
            left, right, threshold_px=1.0
        )
        self.assertTrue(consistent[0, 1])

    def test_shape_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes"):
            ANALYSIS.left_right_consistency(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((3, 2), dtype=np.float32),
            )


class DepthConversionTests(unittest.TestCase):
    def test_metric_depth_and_range_filter(self) -> None:
        disparity = np.array([[10.0, 1.0, 0.0, 96.0]], dtype=np.float32)
        depth = ANALYSIS.disparity_to_depth(disparity, 320.0, 0.3)
        self.assertAlmostEqual(float(depth[0, 0]), 9.6, places=6)
        self.assertTrue(math.isnan(float(depth[0, 1])))  # 96 m > maximum
        self.assertTrue(math.isnan(float(depth[0, 2])))
        self.assertTrue(math.isnan(float(depth[0, 3])))  # 1 m < minimum


if __name__ == "__main__":
    unittest.main()
