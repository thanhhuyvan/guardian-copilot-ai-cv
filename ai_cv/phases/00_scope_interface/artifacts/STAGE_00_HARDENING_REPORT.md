# Stage 00 Contract Review

Date: 2026-07-23  
Status: reviewed internal draft; integration sign-off pending
Tracking issue: `#6`

## Review conclusion

Stage 00.1 materially improves payload correctness and auditability, but it does not prove
causal execution, model robustness or latency. The contracts therefore remain internal
drafts. This report uses `Closed`, `Mitigated`, `Open` and `Deferred` instead of claiming
that every risk is closed.

## Risk treatment

| ID | Risk | Status after review | Residual work |
|---|---|---|---|
| R00-01 | CI only checked folders | Mitigated | CI validates schemas, examples and negative cases. Generated pipeline outputs must join this gate when they exist. |
| R00-02 | Future-frame/event leakage | Mitigated | Causal manifests reject declared future use and two-sided depth interpolation. Phase 01 still needs a causal accessor and future-invariance test; a manifest cannot prove runtime behavior. |
| R00-03 | Dataset/contract taxonomy mismatch | Mitigated | A single mapping config covers `vehicle/walker/bike`; Phase 02 must apply and test it in the loader. |
| R00-04 | TTC at contact (`0`) rejected | Closed | Schemas and a regression test accept zero. |
| R00-05 | Integration approval overstated | Open | Contracts are internal drafts until integration signs off before Phase 05 freeze. |
| R00-06 | Missing run traceability | Mitigated | A manifest defines run/model/config/commit/data/mode/hardware fields. Phase 01 must generate it and cross-check run IDs. |
| R00-07 | Contradictory or invalid payload fields | Mitigated | Per-document checks cover non-finite numbers, bbox bounds, minimum TTC and event ordering. Stream ordering/uniqueness remains Phase 01 work. |
| R00-08 | Confidence versus quality ambiguous | Deferred | Scores are explicitly uncalibrated; numeric-to-label mapping is intentionally deferred to Phase 05 rather than invented now. |
| R00-09 | Runtime/latency fully deferred | Open | Promoted experiments must record wall time, FPS, P50/P95 and hardware. Numeric SLA and hard gates remain Phase 06 work. |
| R00-10 | Depth-keyframe usage unclear | Open | Usage is declared per run; direct/interpolated final-submission use still needs organizer confirmation. |

## Simplifications made by this review

- Removed unused quality-threshold and runtime-target JSON files with unsupported numbers.
- Removed the one-document class-mapping schema and kept one mapping config under `shared/configs`.
- Removed the generic requirement that every finite TTC have metric distance/closing speed;
  direct image-space TTC methods remain valid.
- Stopped enforcing stateless TTC-to-risk equality so Phase 05 can add quality gates,
  debounce and hysteresis without violating the shared contract.
- Added explicit NaN/Infinity rejection and blocked future depth interpolation in causal mode.
- Removed duplicate payload examples and duplicate positive-test execution.
- Kept Stage 00.1 changes under `Unreleased`; project version remains tagged `0.1.0`.

## Verification gate

- Contract example verifier covers positive payloads.
- Unit tests cover structural, semantic, non-finite and causal negative cases plus algorithm-flexibility regressions.
- Repository structure verification passes.
- CI enforces these checks for pull requests targeting `main` or `develop`.

## Next owners

- Phase 01: causal data accessor, future-invariance test, stream/run validation, manifest generation and actual baseline runtime measurements.
- Phase 02: apply and test the class mapping in the detector/loader boundary.
- Phase 05: integration sign-off, event state machine and calibrated confidence mapping.
- Phase 06: robustness matrix, target hardware, latency SLA and promotion gate.
