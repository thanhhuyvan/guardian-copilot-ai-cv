# Tasks — Phase 06

## Completed clean-input deployment gate

- [x] Integrate live stereo, YOLO26, dual TTC, routing, and risk events.
- [x] Validate output schemas.
- [x] Run six trips, five repeats, and 100 warm-up frames.
- [x] Verify deterministic predictions across repeats.
- [x] Measure stage P50/P95/P99, RAM, and VRAM.
- [x] Freeze a live-only Phase 06 entry point.
- [x] Record the clean deployment-readiness decision.

## Remaining robustness gate

- [ ] Create a perturbation generator that never edits the source dataset.
- [ ] Test blur, darkness, noise, occlusion, and frame drops.
- [ ] Test missing/corrupt cameras and calibration failures.
- [ ] Test detector/tracker exceptions and empty detections.
- [ ] Measure accuracy degradation by severity and worst trip.
- [ ] Validate degraded/unknown output contracts.
- [ ] Define and verify the fallback matrix.
- [ ] Publish the final robustness and latency report.
