from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "audit_dataset.py"
SPEC = importlib.util.spec_from_file_location("audit_dataset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditDatasetUnitTests(unittest.TestCase):
    def test_parse_frame_id(self) -> None:
        self.assertEqual(AUDIT.parse_frame_id(Path("000123.jpg")), 123)
        self.assertEqual(AUDIT.parse_frame_id(Path("frame_000123.jpg"), "frame_"), 123)
        self.assertIsNone(AUDIT.parse_frame_id(Path("notes.txt")))

    def test_contiguous_requires_exact_zero_based_ids(self) -> None:
        self.assertTrue(AUDIT.contiguous([0, 1, 2], 3))
        self.assertFalse(AUDIT.contiguous([0, 2], 2))
        self.assertFalse(AUDIT.contiguous([1, 2, 3], 3))

    def test_collapse_ranges(self) -> None:
        self.assertEqual(AUDIT.collapse_ranges([1, 2, 3, 7, 9, 10]), "1-3;7;9-10")
        self.assertEqual(AUDIT.collapse_ranges([]), "")

    def test_finite_ttc_normalization(self) -> None:
        self.assertEqual(AUDIT.as_finite_float("2.5"), 2.5)
        self.assertIsNone(AUDIT.as_finite_float(float("inf")))
        self.assertIsNone(AUDIT.as_finite_float(float("nan")))
        self.assertIsNone(AUDIT.as_finite_float(None))

    def test_episode_count(self) -> None:
        values = [None, 2.5, 2.0, 3.5, 1.0, 0.5, None]
        self.assertEqual(AUDIT.count_episodes(values, 3.0), 2)
        self.assertEqual(AUDIT.count_episodes(values, 2.0), 1)

    def test_nested_path_presence(self) -> None:
        document = {"frames": [{"targets": [{"ttc_2d": math.inf}]}]}
        self.assertTrue(AUDIT.path_exists(document, "frames[].targets[].ttc_2d"))
        self.assertFalse(AUDIT.path_exists(document, "frames[].targets[].closing_speed"))


if __name__ == "__main__":
    unittest.main()
