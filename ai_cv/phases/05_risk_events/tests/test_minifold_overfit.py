from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "diagnose_minifold_capacity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_minifold_capacity",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAGNOSTIC
SPEC.loader.exec_module(DIAGNOSTIC)


class MiniFoldDiagnosticTests(unittest.TestCase):
    def test_tree_fits_separable_signal(self) -> None:
        features = np.asarray([[0.0], [0.1], [0.8], [1.0]])
        labels = np.asarray([False, False, True, True])
        tree = DIAGNOSTIC.fit_tree(
            features,
            labels,
            maximum_depth=2,
            minimum_leaf=1,
        )
        predictions = DIAGNOSTIC.predict_tree(tree, features)
        np.testing.assert_array_equal(predictions, labels)

    def test_blocked_predictions_cover_every_row_once(self) -> None:
        features = np.arange(40, dtype=float).reshape(-1, 1)
        labels = features[:, 0] >= 20
        predictions, folds, _ = DIAGNOSTIC.blocked_predictions(
            features,
            labels,
            folds=4,
        )
        self.assertEqual(predictions.shape, labels.shape)
        self.assertEqual(len(folds), 4)
        covered = []
        for fold in folds:
            covered.extend(
                range(fold["first_index"], fold["last_index"] + 1)
            )
        self.assertEqual(covered, list(range(40)))

    def test_frame_aggregation_has_fixed_finite_shape(self) -> None:
        vector = DIAGNOSTIC.aggregate_frame(())
        self.assertEqual(vector.shape, (len(DIAGNOSTIC.FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(vector)))


if __name__ == "__main__":
    unittest.main()
