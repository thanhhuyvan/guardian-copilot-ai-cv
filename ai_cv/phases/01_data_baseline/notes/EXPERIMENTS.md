# Experiment Registry - Phase 01

## EXP-01-001 - Reproduce official fixed-ROI stereo baseline

**Status:** COMPLETE — frozen reference

### Hypothesis

The organizer baseline can be reproduced deterministically on all six practice trips and provides a stable metric floor for later TTC methods.

### Controlled inputs

- Original practice images and calibration.
- Unmodified `baseline_ttc_predictor.py` parameters.
- Official `evaluation.py`.

### Required outputs

- One prediction CSV per practice trip.
- One evaluation JSON/report per trip and one aggregate report.
- Runtime per trip.
- Failure-case frames for critical false positives and false negatives.

### Promotion decision

This experiment is a reference and is never promoted as the final method. Its metrics become regression fixtures.

Result: mean composite `19.7`, worst-trip composite `5.0`, effective throughput
`18.28 FPS`. See `../artifacts/BASELINE_RUN_001.md`.

## EXP-01-002 - Validate depth keyframes as a research signal

**Status:** COMPLETE — validation-only

### Hypothesis

Provided depth keyframes are geometrically consistent with stereo calibration and can be used to validate or calibrate target-depth estimators without accessing redacted TTC labels.

### Required checks

- Shape, dtype, finite ratio and physical range.
- Alignment with `image_2` resolution.
- Availability pattern and timestamp spacing.
- Agreement with stereo depth on representative regions.

### Decision gate

Use keyframes only if the input is allowed, aligned and demonstrably stable. Document whether usage is direct inference, interpolation, calibration or evaluation-only.

Decision: keep provided depth as validation-only. `T01d` contains a 117-keyframe
all-zero dropout and sentinel values near 1000 m; it is not a reliable direct
inference dependency.

## EXP-01-003 - Lightweight robust temporal/ROI policies

**Status:** COMPLETE — not promoted

The best observed variant reaches composite `32.7` and worst-trip `22.1`, but it
was selected on the same six practice trips, still misses all T05 danger true
positives, uses a hard 20 m/s gate and adds temporal history. It is a Stage 2
target, not the frozen project baseline.
