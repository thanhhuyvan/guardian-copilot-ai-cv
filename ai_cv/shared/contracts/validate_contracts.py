"""JSON Schema and cross-field validation for GuardianCoPilot contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


CONTRACTS = Path(__file__).resolve().parent


class ContractSemanticError(ValueError):
    """Raised when a payload is structurally valid but semantically inconsistent."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: _reject_json_constant(value, path))


def _reject_json_constant(value: str, source: Path) -> None:
    raise ValueError(f"{source}: non-standard JSON number {value} is not allowed")


def _reject_non_finite_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractSemanticError(f"{path}: non-finite numbers are not valid JSON contract values")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_non_finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_non_finite_numbers(child, f"{path}[{index}]")


def _schema(name: str) -> dict[str, Any]:
    schema = load_json(CONTRACTS / name)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_schema(document: dict[str, Any], schema_name: str) -> None:
    _reject_non_finite_numbers(document)
    validator = Draft202012Validator(_schema(schema_name), format_checker=FormatChecker())
    validator.validate(document)


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

def validate_risk_event(document: dict[str, Any]) -> None:
    validate_schema(document, "risk_event.v1.schema.json")
    if document["start_frame"] > document["end_frame"]:
        raise ContractSemanticError("start_frame must not exceed end_frame")
    if document["start_time"] > document["end_time"]:
        raise ContractSemanticError("start_time must not exceed end_time")


def validate_run_manifest(document: dict[str, Any]) -> None:
    validate_schema(document, "run_manifest.v1.schema.json")


def validate_examples() -> None:
    examples = CONTRACTS / "examples"
    for name in ("valid.json", "degraded.json", "unknown.json"):
        validate_perception(load_json(examples / name))
    validate_risk_event(load_json(examples / "risk_event.json"))
    validate_run_manifest(load_json(examples / "run_manifest.json"))


if __name__ == "__main__":
    validate_examples()
    print("Contract schema and semantic validation: OK")
