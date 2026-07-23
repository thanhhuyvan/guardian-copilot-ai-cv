from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

from jsonschema import ValidationError


ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "shared" / "contracts"
sys.path.insert(0, str(CONTRACTS))

from validate_contracts import (  # noqa: E402
    ContractSemanticError,
    load_json,
    validate_perception,
    validate_risk_event,
    validate_run_manifest,
)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        examples = CONTRACTS / "examples"
        cls.perception = load_json(examples / "valid.json")
        cls.event = load_json(examples / "risk_event.json")
        cls.manifest = load_json(examples / "run_manifest.json")

    def test_ttc_zero_is_valid(self) -> None:
        payload = copy.deepcopy(self.perception)
        payload["objects"][0]["ttc_sec"] = 0
        payload["min_ttc_sec"] = 0
        payload["risk_level"] = "CRITICAL"
        validate_perception(payload)

    def test_rejects_unknown_field(self) -> None:
        payload = copy.deepcopy(self.perception)
        payload["surprise"] = True
        with self.assertRaises(ValidationError):
            validate_perception(payload)

    def test_rejects_missing_required_field(self) -> None:
        payload = copy.deepcopy(self.perception)
        del payload["trip_id"]
        with self.assertRaises(ValidationError):
            validate_perception(payload)

    def test_rejects_non_finite_perception_numbers(self) -> None:
        cases = (
            ("latency", lambda payload: payload.__setitem__("latency_ms", math.nan)),
            ("timestamp", lambda payload: payload.__setitem__("timestamp", math.inf)),
            ("quality", lambda payload: payload["objects"][0].__setitem__("ttc_quality", math.nan)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                payload = copy.deepcopy(self.perception)
                mutate(payload)
                with self.assertRaises(ContractSemanticError):
                    validate_perception(payload)

    def test_rejects_reversed_bbox(self) -> None:
        payload = copy.deepcopy(self.perception)
        payload["objects"][0]["bbox_xyxy"] = [431, 126, 212, 328]
        with self.assertRaises(ContractSemanticError):
            validate_perception(payload)

    def test_rejects_bbox_outside_image(self) -> None:
        payload = copy.deepcopy(self.perception)
        payload["objects"][0]["bbox_xyxy"] = [212, 126, 641, 328]
        with self.assertRaises(ContractSemanticError):
            validate_perception(payload)

    def test_rejects_min_ttc_mismatch(self) -> None:
        payload = copy.deepcopy(self.perception)
        payload["min_ttc_sec"] = 1.0
        with self.assertRaises(ContractSemanticError):
            validate_perception(payload)

    def test_allows_stateful_risk_during_hysteresis(self) -> None:
        payload = copy.deepcopy(self.perception)
        payload["objects"][0]["ttc_sec"] = 3.1
        payload["min_ttc_sec"] = 3.1
        payload["risk_level"] = "WARNING"
        validate_perception(payload)

    def test_allows_direct_ttc_without_metric_motion(self) -> None:
        payload = copy.deepcopy(self.perception)
        payload["objects"][0]["distance_m"] = None
        payload["objects"][0]["closing_speed_mps"] = None
        validate_perception(payload)

    def test_rejects_unknown_without_reason(self) -> None:
        payload = load_json(CONTRACTS / "examples" / "unknown.json")
        payload["degraded_reasons"] = []
        with self.assertRaises(ValidationError):
            validate_perception(payload)

    def test_rejects_event_frame_order(self) -> None:
        payload = copy.deepcopy(self.event)
        payload["start_frame"] = payload["end_frame"] + 1
        with self.assertRaises(ContractSemanticError):
            validate_risk_event(payload)

    def test_rejects_non_finite_event_quality(self) -> None:
        payload = copy.deepcopy(self.event)
        payload["event_quality"] = math.nan
        with self.assertRaises(ContractSemanticError):
            validate_risk_event(payload)

    def test_rejects_non_causal_online_manifest(self) -> None:
        for field in ("uses_future_frames", "uses_full_event_schedule"):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.manifest)
                payload[field] = True
                with self.assertRaises(ValidationError):
                    validate_run_manifest(payload)

    def test_offline_manifest_may_use_future_context(self) -> None:
        payload = copy.deepcopy(self.manifest)
        payload["processing_mode"] = "offline_post_trip"
        payload["uses_future_frames"] = True
        payload["uses_full_event_schedule"] = True
        validate_run_manifest(payload)

    def test_rejects_future_depth_interpolation_in_causal_mode(self) -> None:
        payload = copy.deepcopy(self.manifest)
        payload["depth_keyframe_policy"] = "interpolation"
        with self.assertRaises(ValidationError):
            validate_run_manifest(payload)

    def test_rejects_bad_config_hash(self) -> None:
        payload = copy.deepcopy(self.manifest)
        payload["config_sha256"] = "not-a-sha256"
        with self.assertRaises(ValidationError):
            validate_run_manifest(payload)

if __name__ == "__main__":
    unittest.main()
