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
