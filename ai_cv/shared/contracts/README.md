# Contracts

Versioned contracts between AI/CV and integration live here. JSON Schema checks
shape and types; `validate_contracts.py` additionally checks cross-field semantics.

These `v1` files are internal drafts. They are not an externally frozen API until
the integration sign-off gate in Phase 05.

## Contracts and policies

- `perception.v1.schema.json`: one frame of perception output.
- `risk_event.v1.schema.json`: one aggregated collision-risk event.
- `run_manifest.v1.schema.json`: model/config/code/data/mode/hardware traceability.
- `CONTRACT_SEMANTICS.md`: TTC, risk, quality, status and bounding-box invariants.
- `DATA_USAGE_POLICY.md`: causal-online versus offline-post-trip data rules.

## Verification

Install `requirements-dev.txt`, then run:

```text
python ai_cv/phases/00_scope_interface/verify/verify_contracts.py
python -m unittest discover -s ai_cv/phases/00_scope_interface/tests -p "test_*.py"
```

CI runs both commands and rejects schema violations and semantic contradictions.
Full schema validation is a boundary/test-time check, not part of the per-frame inference hot path.
