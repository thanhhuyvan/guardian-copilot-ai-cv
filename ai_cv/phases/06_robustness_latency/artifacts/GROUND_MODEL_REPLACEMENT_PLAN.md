# Ground-model plan — corrected after implementation audit

## Finding

Guardian's current `estimate_ground_model()` is already the proposed
V-disparity replacement:

1. vectorized V-disparity histogram (`v_disparity_histogram`);
2. per-row disparity modes (`row_disparity_modes`);
3. deterministic RANSAC-like weighted ground-line fit (`fit_ground_line`);
4. predicted-ground mask and closer-than-ground obstacle mask.

Therefore adding another V-disparity stage would duplicate production work,
consume latency headroom, and provide no new evidence. It is rejected before
implementation.

## Measured boundary

`ground_ms` measures only this model estimation. Masking and component
extraction are measured separately under `components_ms`; do not attribute the
entire downstream obstacle pipeline to the 18.60 ms ground-model row.

## Correct next experiment

Only a behavior-preserving implementation optimization is justified:

1. profile histogram, row-mode selection, and robust-line fit independently;
2. optimize the dominant substep without changing parameters or output;
3. require bit-identical ground model/masks over all 3,600 frames;
4. require ground P95 ≤ 18.60 ms and full certified P95 ≤ 75 ms;
5. rerun full latency certification only if parity holds.

Changing ground thresholds/masks is an accuracy experiment, not a latency
optimization. The existing T05 ground-leakage screen found false and true
alerts overlap in ground-quality metrics, so it must not be promoted as an F1
fix without object/path labels.
