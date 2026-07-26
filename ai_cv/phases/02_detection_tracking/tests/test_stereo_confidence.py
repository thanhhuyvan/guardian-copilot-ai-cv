from __future__ import annotations

import importlib.util
import math
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

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


class DeterministicMatcher:
    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        self.barrier = barrier

    def compute(
        self,
        reference_gray: np.ndarray,
        target_gray: np.ndarray,
    ) -> np.ndarray:
        if self.barrier is not None:
            self.barrier.wait(timeout=2.0)
        return (
            reference_gray.astype(np.int16)
            - target_gray.astype(np.int16)
        ) * 16


class StereoExecutionTests(unittest.TestCase):
    def test_concurrent_outputs_are_identical_to_sequential_outputs(self) -> None:
        generator = np.random.default_rng(20260726)
        left = generator.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        right = generator.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        expected_left, expected_right, _, _ = (
            ANALYSIS.compute_disparities_with_timing(
                left,
                right,
                DeterministicMatcher(),
                DeterministicMatcher(),
            )
        )

        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            actual_left, actual_right, left_ms, right_ms = (
                ANALYSIS.compute_disparities_with_timing(
                    left,
                    right,
                    DeterministicMatcher(barrier),
                    DeterministicMatcher(barrier),
                    executor=executor,
                )
            )

        np.testing.assert_array_equal(actual_left, expected_left)
        np.testing.assert_array_equal(actual_right, expected_right)
        self.assertGreaterEqual(left_ms, 0.0)
        self.assertGreaterEqual(right_ms, 0.0)

    def test_top_crop_restores_native_shape_and_marks_rows_invalid(self) -> None:
        generator = np.random.default_rng(20260727)
        left = generator.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        right = generator.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        expected_left, expected_right, _, _ = (
            ANALYSIS.compute_disparities_with_timing(
                left[4:],
                right[4:],
                DeterministicMatcher(),
                DeterministicMatcher(),
            )
        )

        actual_left, actual_right, _, _ = (
            ANALYSIS.compute_cropped_disparities_with_timing(
                left,
                right,
                DeterministicMatcher(),
                DeterministicMatcher(),
                roi_top=4,
            )
        )

        self.assertEqual(actual_left.shape, (12, 16))
        self.assertEqual(actual_right.shape, (12, 16))
        self.assertTrue(np.all(actual_left[:4] == -1.0))
        self.assertTrue(np.all(actual_right[:4] == -97.0))
        np.testing.assert_array_equal(actual_left[4:], expected_left)
        np.testing.assert_array_equal(actual_right[4:], expected_right)
        valid, consistent, _ = ANALYSIS.left_right_consistency(
            actual_left, actual_right
        )
        self.assertFalse(np.any(valid[:4]))
        self.assertFalse(np.any(consistent[:4]))

    def test_zero_top_crop_is_exact_full_frame_path(self) -> None:
        generator = np.random.default_rng(20260728)
        left = generator.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        right = generator.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        expected = ANALYSIS.compute_disparities_with_timing(
            left,
            right,
            DeterministicMatcher(),
            DeterministicMatcher(),
        )
        actual = ANALYSIS.compute_cropped_disparities_with_timing(
            left,
            right,
            DeterministicMatcher(),
            DeterministicMatcher(),
            roi_top=0,
        )

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_opencv_thread_count_is_explicit_and_validated(self) -> None:
        with mock.patch.object(ANALYSIS.cv2, "setNumThreads") as setter:
            ANALYSIS.configure_opencv_threads(3)
            setter.assert_called_once_with(3)

        with self.assertRaisesRegex(ValueError, "positive"):
            ANALYSIS.configure_opencv_threads(0)


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
