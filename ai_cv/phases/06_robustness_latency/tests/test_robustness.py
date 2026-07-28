from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "ai_cv" / "phases" / "06_robustness_latency" / "src"
for path in (SRC,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robustness import (
    FAULT_REASONS,
    Perturbation,
    apply_perturbation,
    screening_selector,
    unknown_perception_document,
)


def test_visual_perturbations_are_deterministic_and_non_mutating() -> None:
    left = np.full((40, 80, 3), 140, dtype=np.uint8)
    right = np.full((40, 80, 3), 120, dtype=np.uint8)
    source_left, source_right = left.copy(), right.copy()

    for kind in ("blur", "darkness", "noise", "occlusion"):
        first = apply_perturbation(
            left, right, trip_id="T01-Sample", frame_id=9,
            perturbation=Perturbation(kind, 2),
        )
        second = apply_perturbation(
            left, right, trip_id="T01-Sample", frame_id=9,
            perturbation=Perturbation(kind, 2),
        )
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
    assert np.array_equal(left, source_left)
    assert np.array_equal(right, source_right)


def test_screening_selector_keeps_every_danger_frame_and_safe_stride() -> None:
    class Frame:
        def __init__(self, frame_id: int, min_ttc: float) -> None:
            self.frame_id = frame_id
            self.min_ttc = min_ttc

    assert screening_selector(Frame(3, 1.5), safe_stride=8)
    assert screening_selector(Frame(16, float("inf")), safe_stride=8)
    assert not screening_selector(Frame(17, float("inf")), safe_stride=8)


def test_fault_documents_are_schema_valid_and_fail_closed() -> None:
    schema = (ROOT / "ai_cv" / "shared" / "contracts" / "perception.v1.schema.json")
    validator = Draft202012Validator(__import__("json").loads(schema.read_text()))
    for reason in FAULT_REASONS:
        document = unknown_perception_document(
            trip_id="T02-Sample", frame_id=10, timestamp=0.5,
            latency_ms=1.0, reason=reason,
        )
        assert not list(validator.iter_errors(document))
        assert document["status"] == "unknown"
        assert document["risk_level"] == "UNKNOWN"
        assert document["objects"] == []
