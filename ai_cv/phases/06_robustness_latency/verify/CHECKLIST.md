# Verification — Phase 06

## Clean input

- [x] Accuracy gates pass on all six trips.
- [x] Compute P95 is below the accepted 75 ms target.
- [x] RAM and VRAM are reported with hardware context.
- [x] Five-repeat prediction determinism passes.
- [x] Perception and risk-event documents pass their schemas.

## Degraded input

- [ ] Robustness matrix is complete.
- [ ] A single injected failure cannot crash an entire trip.
- [ ] Mean and worst-trip degradation are reported.
- [ ] Every degraded mode emits a valid degraded/unknown contract.
- [ ] Fallback behavior is deterministic and documented.
