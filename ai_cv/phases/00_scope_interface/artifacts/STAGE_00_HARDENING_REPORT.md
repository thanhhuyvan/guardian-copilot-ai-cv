# Stage 00.1 Hardening Report

Date: 2026-07-23  
Repository version: 0.1.1  
Branch: `fix/phase-00-contract-hardening`
Tracking issue: `#4`

## Outcome

The Stage 00 scope remains out-car Fleet Collision Intelligence with a reusable causal
AI/CV core. The contracts are now executable specifications rather than documentation-only
examples: JSON Schema, semantic invariants, negative tests and CI jointly enforce them.

## Risk closure matrix

| ID | Audited risk | Resolution | Automated evidence |
|---|---|---|---|
| R00-01 | CI checked folders but not contracts | CI installs the pinned development dependency and runs schema, semantic and negative tests | `.github/workflows/ci.yml` |
| R00-02 | Future `events_log` could leak into online predictions | Added explicit processing modes; causal manifests reject future frames and full future event schedules | `run_manifest.v1.schema.json`, negative test |
| R00-03 | Dataset labels and contract taxonomy disagreed | Added versioned `vehicle/walker/bike` to `vehicle/pedestrian/two_wheeler` mapping | `class_mapping.v1.*` |
| R00-04 | TTC at contact (`0`) was invalid | TTC fields now allow zero and semantic tests confirm it maps to `CRITICAL` | schema plus `test_ttc_zero_is_valid_and_critical` |
| R00-05 | Integration approval was overstated | Documentation now says CV-owner freeze; integration sign-off is a gate before Phase 05 | Phase 00 README/checklist |
| R00-06 | Runs lacked reproducibility metadata | Added mandatory run ID, code commit, model/config, dataset, inputs, mode, depth policy and hardware manifest | `run_manifest.v1.schema.json` |
| R00-07 | Cross-field contradictions could pass | Validator checks bbox ordering/bounds, closing motion, min TTC aggregation, risk/severity and event ordering | `validate_contracts.py` and negative tests |
| R00-08 | Confidence and quality were ambiguous | Uncalibrated numeric scores are named `*_quality`; product-facing event confidence is a configured ordinal level | contract semantics and `quality_levels.v1.json` |
| R00-09 | Runtime was deferred entirely to Phase 06 | Every promoted experiment must record wall time, FPS, P50/P95 and hardware; Phase 06 retains the hard gate | `runtime_guardrails.v1.json` |
| R00-10 | Depth keyframe legality was implicit | Every run declares a depth policy; direct/interpolated final use requires organizer confirmation | data-usage policy and run manifest |

## Enforced semantics

- TTC bands: `CRITICAL < 1.5`, `DANGER < 2.0`, `WARNING < 3.0`, otherwise `SAFE`.
- JSON `null` means no reliable finite TTC; competition CSV may serialize this as `inf`.
- Frame minimum TTC equals the minimum finite TTC among collision-corridor objects.
- A finite TTC requires positive closing speed.
- Bounding boxes must be ordered and remain inside the declared image dimensions.
- `unknown` frames carry no objects, no TTC, `UNKNOWN` risk, zero quality and at least one reason.
- Event time/frame ordering, severity and quality-to-confidence mapping are consistent.

## Verification result

Executed locally on 2026-07-23:

```text
Phase 00 contract schema and semantic verification: OK
Perception examples checked: 3
Risk event examples checked: 1
Run manifest examples checked: 1
Class mapping checked: 1

Ran 16 tests in 0.202s
OK

Workspace structure: OK
Phases checked: 8
Dataset/starter roots: SKIPPED
```

## Remaining explicit gates (not silent risks)

- Integration owner sign-off before Phase 05 integration work.
- Organizer confirmation before promoting direct/interpolated depth keyframes to a final submission feature.
- Exact target hardware and production latency SLA in Phase 06. Coarse runtime measurement starts immediately.
- The quality score is not a calibrated probability; calibration must be demonstrated before making probability claims.

## Stage decision

Stage 00.1 passes its local verification gate and is ready for pull-request review. Phase 01
may proceed using the hardened contracts and must produce a run manifest for each promoted
baseline or TTC experiment.
