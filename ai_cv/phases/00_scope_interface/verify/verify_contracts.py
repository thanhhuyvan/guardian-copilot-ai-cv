"""Dependency-free smoke verification for Phase 00 JSON contracts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACTS = ROOT / "shared" / "contracts"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_keys(document: dict, keys: set[str], source: Path) -> None:
    missing = keys - document.keys()
    if missing:
        raise AssertionError(f"{source}: missing keys {sorted(missing)}")


def main() -> int:
    perception_schema = load_json(CONTRACTS / "perception.v1.schema.json")
    event_schema = load_json(CONTRACTS / "risk_event.v1.schema.json")
    require_keys(perception_schema, {"$schema", "$id", "type", "required", "properties"}, CONTRACTS)
    require_keys(event_schema, {"$schema", "$id", "type", "required", "properties"}, CONTRACTS)

    perception_required = set(perception_schema["required"])
    for name in ("valid.json", "degraded.json", "unknown.json"):
        source = CONTRACTS / "examples" / name
        document = load_json(source)
        require_keys(document, perception_required, source)
        assert document["schema_version"] == "perception.v1"
        assert document["status"] in {"valid", "degraded", "unknown"}
        assert 0 <= document["perception_quality"] <= 1

    event_source = CONTRACTS / "examples" / "risk_event.json"
    event = load_json(event_source)
    require_keys(event, set(event_schema["required"]), event_source)
    assert event["schema_version"] == "risk_event.v1"
    assert event["start_frame"] <= event["end_frame"]
    assert event["start_time"] <= event["end_time"]
    assert event["severity"] in {"WARNING", "DANGER", "CRITICAL"}

    print("Phase 00 contract smoke verification: OK")
    print("Perception examples checked: 3")
    print("Risk event examples checked: 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

