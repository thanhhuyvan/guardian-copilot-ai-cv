from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
PHASE02_SRC = Path(__file__).resolve().parents[2] / "02_detection_tracking" / "src"
for path in (SRC, PHASE02_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cross_validate_confidence_router import (
    BANNED_FEATURE_TOKENS,
    FEATURE_NAMES,
    PairedTrip,
    conservative_union_predictions,
    fit_router,
    route_predictions,
    router_features,
)


def test_feature_contract_excludes_leakage_identifiers() -> None:
    assert not any(
        token in feature.lower()
        for feature in FEATURE_NAMES
        for token in BANNED_FEATURE_TOKENS
    )


def test_router_features_ignore_identifier_columns() -> None:
    classical = {
        "predicted_ttc": "inf",
        "prediction_confidence": "0.6",
        "frame_id": "10",
        "timestamp": "0.5",
    }
    detector = {
        "predicted_ttc": "1.2",
        "selection_confidence": "0.9",
        "frame_id": "10",
        "timestamp": "0.5",
    }
    first = router_features(classical, detector)
    classical["frame_id"] = "999"
    classical["timestamp"] = "99"
    detector["frame_id"] = "999"
    detector["timestamp"] = "99"
    np.testing.assert_array_equal(first, router_features(classical, detector))


def test_router_changes_only_disagreement_frames() -> None:
    negative = np.zeros((20, len(FEATURE_NAMES)), dtype=float)
    positive = np.ones((20, len(FEATURE_NAMES)), dtype=float)
    router = fit_router(
        np.vstack([negative, positive]),
        np.asarray([False] * 20 + [True] * 20),
    )
    features = np.vstack([negative[0], positive[0], positive[1]])
    data = PairedTrip(
        trip_id="held-out",
        frame_ids=np.arange(3),
        truth=np.asarray([5.0, 1.0, 1.0]),
        classical_ttc=np.asarray([5.0, math.inf, 1.5]),
        detector_ttc=np.asarray([5.5, 1.0, 1.0]),
        features=features,
    )

    predictions, selected = route_predictions(data, router)

    assert predictions[0] == 5.0
    assert predictions[1] == 1.0
    assert predictions[2] == 1.5
    assert selected.tolist() == [False, True, False]


def test_conservative_union_adds_only_detector_only_danger() -> None:
    data = PairedTrip(
        trip_id="held-out",
        frame_ids=np.arange(4),
        truth=np.full(4, math.inf),
        classical_ttc=np.asarray([math.inf, 1.2, 1.0, math.inf]),
        detector_ttc=np.asarray([1.5, math.inf, 0.8, math.inf]),
        features=np.zeros((4, len(FEATURE_NAMES))),
    )

    predictions = conservative_union_predictions(data)

    assert predictions[0] == 1.5
    assert predictions[1] == 1.2
    assert predictions[2] == 1.0
    assert math.isinf(predictions[3])
