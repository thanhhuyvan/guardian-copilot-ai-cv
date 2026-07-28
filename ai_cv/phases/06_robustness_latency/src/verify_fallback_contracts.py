"""Validate the fail-closed contract for every Phase 06 injected fault."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE05_SRC = REPOSITORY_ROOT / "ai_cv" / "phases" / "05_risk_events" / "src"
PHASE06_SRC = Path(__file__).resolve().parent
for path in (PHASE05_SRC, PHASE06_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from risk_events import RiskState, RiskStateMachine  # noqa: E402
from robustness import FAULT_REASONS, unknown_perception_document  # noqa: E402


def verify() -> dict[str, object]:
    schema = json.loads(
        (REPOSITORY_ROOT / "ai_cv" / "shared" / "contracts" / "perception.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema)
    documents = []
    for index, reason in enumerate(FAULT_REASONS):
        document = unknown_perception_document(
            trip_id="T01-Sample",
            frame_id=index,
            timestamp=index / 20.0,
            latency_ms=0.1,
            reason=reason,
        )
        errors = list(validator.iter_errors(document))
        if errors:
            raise AssertionError(f"invalid {reason}: {errors[0].message}")
        documents.append(document)

    machine = RiskStateMachine()
    machine.update(1, 0.05, 1.0)
    unknown = machine.update(2, 0.10, float("inf"), reliable=False)
    recovered = machine.update(3, 0.15, float("inf"), reliable=True)
    if unknown.state != RiskState.UNKNOWN or recovered.state != RiskState.NORMAL:
        raise AssertionError("fault recovery must clear risk state")
    return {
        "fault_reasons": list(FAULT_REASONS),
        "schema_valid_documents": len(documents),
        "fail_closed": True,
        "tracker_state_recovery": "verified",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
