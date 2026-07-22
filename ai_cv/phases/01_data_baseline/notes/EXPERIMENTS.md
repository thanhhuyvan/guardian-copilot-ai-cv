# Experiment Registry - Phase 01

## EXP-01-001 - Reproduce official fixed-ROI stereo baseline

**Status:** PLANNED

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

## EXP-01-002 - Validate depth keyframes as a research signal

**Status:** PLANNED

### Hypothesis

Provided depth keyframes are geometrically consistent with stereo calibration and can be used to validate or calibrate target-depth estimators without accessing redacted TTC labels.

### Required checks

- Shape, dtype, finite ratio and physical range.
- Alignment with `image_2` resolution.
- Availability pattern and timestamp spacing.
- Agreement with stereo depth on representative regions.

### Decision gate

Use keyframes only if the input is allowed, aligned and demonstrably stable. Document whether usage is direct inference, interpolation, calibration or evaluation-only.

