# V2 event-to-TTC comparison policy

## Purpose

The organizer calculates danger-F1 from `predicted_ttc < 2.0 s`, while V2
produces a path-occupancy risk event. This document fixes one conversion
before its first score. It is an experiment, not the V1 deployment policy.

## Fixed conversion

For each frame, begin with frozen conservative-union TTC `t_raw`.

1. Consider a V2 gate only when `t_raw < 2.0 s` comes from the classical
   branch.
2. Match that classical track to a current YOLO/stereo track only when IoU is
   at least `0.30`. No match means preserve `t_raw`.
3. Read that matched track's EKF Gaussian corridor-occupancy probability.
   Occupancy below `0.50` supplies a finite non-danger input TTC of `2.0 s`.
   Occupancy at or above `0.50`, unavailable occupancy, rejected EKF update,
   or unavailable match preserves `t_raw`.
4. Feed this candidate TTC sequence through the existing `RiskStateMachine`;
   do not add a second debounce or tune frame counts.
5. For the experimental submission sequence, emit `t_raw` only while that
   FSM is `HIGH_RISK`; otherwise emit `max(t_raw, 2.0)`. Thus every suppressed
   value stays finite and no value is invented from V2 geometry.

`0.30` is a fixed geometric association contract and `0.50` is the natural
decision boundary of a calibrated probability. Neither may be changed after
scoring this run.

## Required reports

- Organizer TTC F1, critical TTC MAE, composite, TP/FP/FN, per trip.
- Separate deployment-event F1 from the FSM output.
- Gate counts: raw classical danger, matched tracks, low-occupancy suppression,
  unavailable association/occupancy, and FSM suppression.
- P50/P95 pipeline latency and memory.

## Pass / fail

Pass only when all hold:

1. Organizer F1 is at least V1 `0.654`.
2. Critical TTC MAE is at most V1 `29.993 s`.
3. T02 and T05 F1 do not fall below V1 values (`0.757`, `0.509`).
4. P95 compute latency is at most `75 ms`.
5. V2 associated-track CPA audit remains directionally correct.

If any condition fails, reject this conversion unchanged. Do not retune IoU,
occupancy, floor TTC, or FSM timing on these six trips.
