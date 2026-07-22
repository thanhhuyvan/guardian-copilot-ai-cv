"""JSON Schema and cross-field validation for GuardianCoPilot contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


CONTRACTS = Path(__file__).resolve().parent
CONFIGS = CONTRACTS.parent / "configs"


class ContractSemanticError(ValueError):
    """Raised when a payload is structurally valid but semantically inconsistent."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _schema(name: str) -> dict[str, Any]:
    schema = load_json(CONTRACTS / name)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_schema(document: dict[str, Any], schema_name: str) -> None:
    validator = Draft202012Validator(_schema(schema_name), format_checker=FormatChecker())
    validator.validate(document)


def risk_level_from_ttc(ttc_sec: float | None) -> str:
    if ttc_sec is None or not math.isfinite(ttc_sec):
        return "SAFE"
    if ttc_sec < 1.5:
        return "CRITICAL"
    if ttc_sec < 2.0:
        return "DANGER"
    if ttc_sec < 3.0:
        return "WARNING"
    return "SAFE"


def confidence_level_from_quality(quality: float) -> str:
    thresholds = load_json(CONFIGS / "quality_levels.v1.json")
    if quality < thresholds["low_upper_exclusive"]:
        return "LOW"
    if quality < thresholds["high_lower_inclusive"]:
        return "MEDIUM"
    return "HIGH"


def validate_perception(document: dict[str, Any]) -> None:
    validate_schema(document, "perception.v1.schema.json")
    width, height = document["image_width"], document["image_height"]
    finite_corridor_ttc: list[float] = []

    for obj in document["objects"]:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        if not (x1 < x2 <= width and y1 < y2 <= height):
            raise ContractSemanticError(
                f"track {obj['track_id']}: bbox must satisfy 0 <= x1 < x2 <= image_width "
                "and 0 <= y1 < y2 <= image_height"
            )
        ttc = obj["ttc_sec"]
        closing_speed = obj["closing_speed_mps"]
        if ttc is not None and (closing_speed is None or closing_speed <= 0):
            raise ContractSemanticError(
                f"track {obj['track_id']}: finite TTC requires positive closing_speed_mps"
            )
        if obj["in_collision_corridor"] and ttc is not None:
            finite_corridor_ttc.append(ttc)

    expected_min = min(finite_corridor_ttc) if finite_corridor_ttc else None
    actual_min = document["min_ttc_sec"]
    if expected_min is None and actual_min is not None:
        raise ContractSemanticError("min_ttc_sec must be null when no corridor object has finite TTC")
    if expected_min is not None and (
        actual_min is None or not math.isclose(actual_min, expected_min, rel_tol=1e-6, abs_tol=1e-6)
    ):
        raise ContractSemanticError("min_ttc_sec must equal the minimum finite TTC in the collision corridor")

    expected_risk = risk_level_from_ttc(actual_min)
    if document["status"] == "degraded" and actual_min is None:
        expected_risk = "UNKNOWN"
    if document["status"] != "unknown" and document["risk_level"] != expected_risk:
        raise ContractSemanticError(
            f"risk_level {document['risk_level']} is inconsistent with min_ttc_sec; expected {expected_risk}"
        )


def validate_risk_event(document: dict[str, Any]) -> None:
    validate_schema(document, "risk_event.v1.schema.json")
    if document["start_frame"] > document["end_frame"]:
        raise ContractSemanticError("start_frame must not exceed end_frame")
    if document["start_time"] > document["end_time"]:
        raise ContractSemanticError("start_time must not exceed end_time")
    expected_severity = risk_level_from_ttc(document["min_ttc_sec"])
    if expected_severity not in {"WARNING", "DANGER", "CRITICAL"}:
        raise ContractSemanticError("risk event min_ttc_sec must be below the 3.0 s event threshold")
    if document["severity"] != expected_severity:
        raise ContractSemanticError(
            f"severity {document['severity']} is inconsistent with min_ttc_sec; expected {expected_severity}"
        )
    expected_confidence = confidence_level_from_quality(document["event_quality"])
    if document["confidence_level"] != expected_confidence:
        raise ContractSemanticError(
            f"confidence_level {document['confidence_level']} is inconsistent with event_quality; "
            f"expected {expected_confidence}"
        )


def validate_run_manifest(document: dict[str, Any]) -> None:
    validate_schema(document, "run_manifest.v1.schema.json")


def validate_class_mapping(document: dict[str, Any]) -> None:
    validate_schema(document, "class_mapping.v1.schema.json")


def validate_examples() -> None:
    examples = CONTRACTS / "examples"
    for name in ("valid.json", "degraded.json", "unknown.json"):
        validate_perception(load_json(examples / name))
    validate_risk_event(load_json(examples / "risk_event.json"))
    validate_run_manifest(load_json(examples / "run_manifest.json"))
    validate_class_mapping(load_json(CONTRACTS / "class_mapping.v1.json"))


if __name__ == "__main__":
    validate_examples()
    print("Contract schema and semantic validation: OK")
