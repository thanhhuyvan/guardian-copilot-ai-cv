from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "classical_geometry.py"
SPEC = importlib.util.spec_from_file_location("classical_geometry", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
GEOMETRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEOMETRY
SPEC.loader.exec_module(GEOMETRY)


class GroundFitTests(unittest.TestCase):
    def test_robust_line_fit_ignores_mode_outliers(self) -> None:
        rows = np.arange(180, 360, dtype=np.float32)
        disparities = 0.12 * rows - 18.0
        disparities[::9] += 15.0
        weights = np.full_like(rows, 30.0)

        model = GEOMETRY.fit_ground_line(rows, disparities, weights)

        self.assertIsNotNone(model)
        assert model is not None
        self.assertAlmostEqual(model.disparity_per_row, 0.12, places=3)
        self.assertAlmostEqual(model.intercept, -18.0, places=2)
        self.assertGreater(model.confidence, 0.5)

    def test_too_few_rows_returns_none(self) -> None:
        rows = np.arange(10, dtype=np.float32)
        disparities = rows * 0.1
        self.assertIsNone(GEOMETRY.fit_ground_line(rows, disparities))


class MaskTests(unittest.TestCase):
    def test_ground_and_closer_obstacle_are_separated(self) -> None:
        height, width = 100, 120
        rows = np.arange(height, dtype=np.float32)[:, None]
        plane = 0.10 * rows + 2.0
        disparity = np.broadcast_to(plane, (height, width)).copy()
        disparity[45:75, 50:70] += 5.0
        model = GEOMETRY.GroundModel(0.10, 2.0, 1.0, 0.0, 50, 50)

        ground, obstacle, _ = GEOMETRY.ground_and_obstacle_masks(
            disparity, model
        )

        self.assertTrue(np.all(ground[80:, 10:40]))
        self.assertTrue(np.all(obstacle[45:75, 50:70]))
        self.assertFalse(np.any(ground[45:75, 50:70]))

    def test_corridor_is_narrower_at_top(self) -> None:
        mask = GEOMETRY.collision_corridor_mask((100, 200))
        self.assertLess(np.count_nonzero(mask[40]), np.count_nonzero(mask[99]))
        self.assertTrue(mask[99, 100])
        self.assertFalse(mask[99, 0])

    def test_vertical_support_rejects_thin_road_band(self) -> None:
        disparity = np.full((100, 120), 10.0, dtype=np.float32)
        evidence = np.zeros_like(disparity, dtype=bool)
        evidence[80:84, 20:100] = True
        evidence[42:80, 54:70] = True
        consistent = np.ones_like(evidence)

        components, _, _ = GEOMETRY.extract_obstacle_components(
            disparity,
            evidence,
            consistent,
            focal_length_px=320.0,
            baseline_m=0.3,
        )

        self.assertEqual(len(components), 1)
        self.assertGreater(components[0].height, components[0].width)


if __name__ == "__main__":
    unittest.main()
