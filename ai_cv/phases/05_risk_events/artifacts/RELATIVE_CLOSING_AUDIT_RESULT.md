# T05 relative-closing reliability audit

The audit compares the last five current/past YOLO-box stereo depths for every
hard-associated, on-path classical-danger frame. It uses a robust Theil-Sen
depth slope only for diagnosis; no threshold, TTC, or event was changed.

## Result

All 13 T05 frames that are labelled non-danger but have a hard-associated,
on-path track have a positive robust closing estimate:

| Statistic | Value |
|---|---:|
| Median robust closing speed | 3.168 m/s |
| Median depth-fit residual MAD | 0.0098 m |
| Non-positive closing estimates | 0 / 13 |

There are no hard-associated, on-path T05 true-danger frames with sufficient
history in this selection, so this audit cannot form a matched true/false
separability comparison.

## Decision

Do **not** add a temporal or EKF closing-rate gate for these cases. The
independent YOLO-box depth history agrees with the classical branch that range
is decreasing, and it is stable rather than noisy. Such a gate would preserve
the same alerts and cannot explain the T05 metric disagreement.

The remaining uncertainty is event correspondence: the challenge's framewise
minimum TTC label does not identify which visible object should own the event.
For these particular frames, the data cannot establish whether the alert is a
wrong object, a non-collision closing interaction, or a label/event-policy
mismatch. More thresholding is not justified.
