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


def legacy_v_disparity_histogram(
    disparity: np.ndarray,
    *,
    max_disparity: int = 96,
    bin_size: float = 0.5,
    x_margin_fraction: float = 0.10,
) -> np.ndarray:
    height, width = disparity.shape
    bins = int(max_disparity / bin_size)
    histogram = np.zeros((height, bins), dtype=np.float32)
    x0 = int(width * x_margin_fraction)
    x1 = int(width * (1.0 - x_margin_fraction))
    for row in range(height):
        values = disparity[row, x0:x1]
        values = values[
            np.isfinite(values)
            & (values > 0.5)
            & (values < max_disparity)
        ]
        if values.size:
            indices = np.clip(
                (values / bin_size).astype(np.int32),
                0,
                bins - 1,
            )
            histogram[row] = np.bincount(indices, minlength=bins)
    return histogram


def legacy_component_labels(
    obstacle_evidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corridor = GEOMETRY.collision_corridor_mask(obstacle_evidence.shape)
    binary = (obstacle_evidence & corridor).astype(np.uint8) * 255
    for operation, shape in (
        (GEOMETRY.cv2.MORPH_OPEN, (3, 7)),
        (GEOMETRY.cv2.MORPH_CLOSE, (9, 5)),
        (GEOMETRY.cv2.MORPH_OPEN, (3, 3)),
    ):
        binary = GEOMETRY.cv2.morphologyEx(
            binary,
            operation,
            GEOMETRY.cv2.getStructuringElement(
                GEOMETRY.cv2.MORPH_RECT,
                shape,
            ),
            iterations=1,
        )
    _, labels, _, _ = GEOMETRY.cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    return labels, corridor


class HistogramTests(unittest.TestCase):
    def test_vectorized_histogram_is_exactly_equivalent(self) -> None:
        generator = np.random.default_rng(20260726)
        disparity = generator.uniform(-2.0, 90.0, (37, 53)).astype(np.float32)
        disparity[0, :6] = [np.nan, np.inf, -np.inf, 0.5, 64.0, 63.999]

        expected = legacy_v_disparity_histogram(
            disparity,
            max_disparity=64,
            bin_size=0.25,
            x_margin_fraction=0.13,
        )
        actual = GEOMETRY.v_disparity_histogram(
            disparity,
            max_disparity=64,
            bin_size=0.25,
            x_margin_fraction=0.13,
        )

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)

    def test_vectorized_histogram_preserves_empty_rows(self) -> None:
        disparity = np.full((4, 8), np.nan, dtype=np.float32)

        expected = legacy_v_disparity_histogram(disparity)
        actual = GEOMETRY.v_disparity_histogram(disparity)

        np.testing.assert_array_equal(actual, expected)


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

    def test_vectorized_ground_line_matches_reference(self) -> None:
        rows = np.arange(180, dtype=np.float32)
        disparities = 0.10 * rows + 2.0
        weights = np.linspace(1.0, 5.0, rows.size, dtype=np.float32)
        self.assertEqual(
            GEOMETRY.fit_ground_line(rows, disparities, weights),
            GEOMETRY.fit_ground_line_vectorized(rows, disparities, weights),
        )

    def test_corridor_is_narrower_at_top(self) -> None:
        mask = GEOMETRY.collision_corridor_mask((100, 200))
        self.assertLess(np.count_nonzero(mask[40]), np.count_nonzero(mask[99]))
        self.assertTrue(mask[99, 100])
        self.assertFalse(mask[99, 0])

    def test_cached_corridor_still_returns_caller_owned_arrays(self) -> None:
        GEOMETRY._cached_collision_corridor_mask.cache_clear()

        first = GEOMETRY.collision_corridor_mask((100, 200))
        first[99, 100] = False
        second = GEOMETRY.collision_corridor_mask((100, 200))

        self.assertTrue(second[99, 100])
        self.assertIsNot(first, second)
        self.assertEqual(
            GEOMETRY._cached_collision_corridor_mask.cache_info().hits,
            1,
        )

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

    def test_component_roi_metrics_match_full_frame_reference(self) -> None:
        disparity = np.full((120, 160), 10.0, dtype=np.float32)
        disparity[45:100, 70:95] = np.linspace(
            8.0,
            16.0,
            25,
            dtype=np.float32,
        )[None, :]
        evidence = np.zeros_like(disparity, dtype=bool)
        evidence[45:100, 70:95] = True
        consistent = np.indices(evidence.shape).sum(axis=0) % 3 != 0

        components, labels, corridor = GEOMETRY.extract_obstacle_components(
            disparity,
            evidence,
            consistent,
            focal_length_px=320.0,
            baseline_m=0.3,
        )
        reference_labels, reference_corridor = legacy_component_labels(evidence)

        np.testing.assert_array_equal(labels, reference_labels)
        np.testing.assert_array_equal(corridor, reference_corridor)
        self.assertEqual(len(components), 1)
        component = components[0]
        component_region = labels == component.component_id
        evidence_region = component_region & evidence
        reference_disparities = disparity[evidence_region]
        reference_depths = 320.0 * 0.3 / reference_disparities
        reference_depth = float(np.median(reference_depths))
        reference_mad = float(
            np.median(np.abs(reference_depths - reference_depth))
        )
        reference_lr_support = float(
            np.count_nonzero(evidence_region & consistent)
            / np.count_nonzero(evidence_region)
        )
        reference_corridor_overlap = float(
            np.count_nonzero(component_region & corridor)
            / np.count_nonzero(component_region)
        )
        reference_quality = float(
            0.40
            + 0.25 * reference_lr_support
            + 0.20 * reference_corridor_overlap
            + 0.15
            * np.exp(-reference_mad / max(1.0, reference_depth * 0.15))
        )

        self.assertEqual(component.area, np.count_nonzero(component_region))
        self.assertEqual(component.depth_m, reference_depth)
        self.assertEqual(
            component.depth_p20_m,
            float(np.percentile(reference_depths, 20.0)),
        )
        self.assertEqual(
            component.depth_p35_m,
            float(np.percentile(reference_depths, 35.0)),
        )
        self.assertEqual(component.depth_mad_m, reference_mad)
        self.assertEqual(component.lr_support, reference_lr_support)
        self.assertEqual(component.corridor_overlap, reference_corridor_overlap)
        self.assertAlmostEqual(component.quality, reference_quality, places=15)


class ObjectDepthTests(unittest.TestCase):
    def test_selects_nearer_significant_inner_roi_mode(self) -> None:
        disparity = np.full((40, 60), 8.0, dtype=np.float32)
        disparity[:, 30:] = 16.0
        evidence = np.ones_like(disparity, dtype=bool)
        consistent = np.ones_like(disparity, dtype=bool)

        estimate = GEOMETRY.estimate_object_depth(
            disparity,
            evidence,
            consistent,
            focal_length_px=320.0,
            baseline_m=0.3,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertEqual(estimate.mode_count, 2)
        self.assertAlmostEqual(estimate.depth_m, 6.0, places=2)
        self.assertGreater(estimate.confidence, 0.6)

    def test_inner_roi_ignores_boundary_disparity(self) -> None:
        disparity = np.full((40, 60), 12.0, dtype=np.float32)
        disparity[:, :8] = 30.0
        disparity[:, -8:] = 30.0
        evidence = np.ones_like(disparity, dtype=bool)
        consistent = np.ones_like(disparity, dtype=bool)

        estimate = GEOMETRY.estimate_object_depth(
            disparity,
            evidence,
            consistent,
            focal_length_px=320.0,
            baseline_m=0.3,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.depth_m, 8.0, places=2)

    def test_sparse_inner_roi_returns_none(self) -> None:
        disparity = np.full((20, 20), np.nan, dtype=np.float32)
        evidence = np.zeros_like(disparity, dtype=bool)
        consistent = np.zeros_like(disparity, dtype=bool)

        self.assertIsNone(
            GEOMETRY.estimate_object_depth(
                disparity,
                evidence,
                consistent,
                focal_length_px=320.0,
                baseline_m=0.3,
            )
        )


if __name__ == "__main__":
    unittest.main()
