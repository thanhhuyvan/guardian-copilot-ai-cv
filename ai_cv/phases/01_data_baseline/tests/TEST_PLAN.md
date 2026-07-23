# Test Plan - Phase 01

- Loader/count/timestamp tests cho 16 trip.
- Future-invariance test: changing future frames/events cannot change earlier causal predictions.
- Calibration và stereo-pair validation.
- Depth zero/sentinel/saturation and `T01d` degraded-range test.
- Ground-truth availability test giữa practice/redacted.
- Baseline output format và evaluator smoke test.
- Reject missing/duplicate/extra frame, timestamp mismatch, negative/NaN TTC trước evaluator.
- Perfect prediction, all-`inf`, TTC boundaries `3.0/2.0/1.5/0.1` và ignored GT-column tests.
- Predictor reset, corrupt/missing stereo fallback và timestamp gap/reversal tests.
- Stream monotonicity/uniqueness and manifest-to-output run ID validation.
- Baseline metric regression test.
