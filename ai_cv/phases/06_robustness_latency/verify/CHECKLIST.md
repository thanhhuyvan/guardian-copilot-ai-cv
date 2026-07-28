# Verification — Phase 06

## Clean input

- [x] Accuracy gates pass on all six trips.
- [x] Compute P95 is below the accepted 75 ms target.
- [x] RAM and VRAM are reported with hardware context.
- [x] Five-repeat prediction determinism passes.
- [x] Perception and risk-event documents pass their schemas.

## Degraded input

- [x] Medium-severity robustness matrix is complete across all six trips.
- [x] A single injected failure uses a fail-closed contract.
- [x] Clean-versus-degraded macro degradation is reported.
- [x] Every injected degraded mode emits a valid unknown contract.
- [x] Fallback behavior is deterministic and documented.
