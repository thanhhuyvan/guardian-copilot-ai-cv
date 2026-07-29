# Path-only shadow study

## Isolated variable

For each current YOLO box-depth measurement, project its centre to camera
forward/lateral coordinates. Use ego speed and lateral acceleration only to
estimate yaw rate and host lateral displacement at the object range. Emit the
object's lateral offset from that geometric host path.

No EKF, temporal velocity, TTC, occupancy threshold, FSM, association, or
prediction is changed. The output is telemetry for the existing review labels
only. A row is unavailable rather than imputed when the linked YOLO-depth
measurement, non-zero ego speed, or valid telemetry is absent.

## Validation

The human reviewer supplies `path_relation` without looking at the numerical
offset: `on_path`, `adjacent`, or `diverging`. The review pack is deliberately
stratified over T01 and T05 known failures plus true-danger anchors from the
other trips.

Pass criteria, fixed before scoring:

1. At least 24 of the 30 sampled rows have direct geometry available.
2. For reviewed rows, the sign/magnitude of direct offset must separate the
   `on_path` group from the combined `adjacent`/`diverging` group in a
   directionally consistent way; report all rows rather than tuning a cutoff.
3. The output must never replace a finite TTC or modify a V1 event.

Only after the review confirms the geometric state is credible may a
path-risk gate and one official F1/MAE run be proposed. A score change without
this review is not evidence that the path model is correct.

## Fixed diagnostic event policy

The first scored diagnostic is intentionally narrow. It starts from the
frozen conservative-union TTC in Phase 17 and considers only a classical
danger candidate (`TTC < 2 s`) that has a current YOLO track with IoU at least
0.30. It projects that YOLO box-depth measurement directly into the
ego-compensated path frame. If the absolute offset exceeds the physical 1.75 m
half-lane corridor, the diagnostic emits `2.0 s` for the *event* policy while
retaining the original finite TTC in a parallel raw column. No missing or
unassociated measurement is suppressed.

`2.0 s`, rather than infinity, is the pre-registered event-to-TTC encoding:
the challenge interface has one TTC field and derives danger from `< 2 s`.
It avoids the evaluator's `99 s` substitution, but does not make its
framewise TTC MAE equivalent to an event-system metric.

This diagnostic is a candidate only when the frozen raw score reproduces the
Phase 17 V1 score, T01 improves, neither T02 nor T05 F1 declines, and critical
TTC MAE does not regress. The current review is AI-provisional, so even a
passing score cannot be promoted before independent visual review.
