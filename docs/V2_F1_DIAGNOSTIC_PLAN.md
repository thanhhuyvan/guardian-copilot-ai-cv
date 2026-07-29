# V2 F1 diagnostic plan

## Goal

Determine whether path-aware V2 can improve Guardian's official framewise TTC
F1, or whether it should remain a deployment-only explanatory alert lane.

V1 stays frozen at F1 `0.654`, critical TTC MAE `29.993 s`, composite `42.8`.
No threshold, IoU, floor-TTC, covariance, or FSM timing sweep is allowed.

## Step 1: V2 coverage upper bound

**Question:** Of V1's false-positive and true-positive danger frames, how many
can V2 actually evaluate?

For every V1 danger frame, record:

- V1 outcome: TP or FP from organizer ground truth;
- classical-track-to-YOLO match IoU;
- matched detector track present or absent;
- valid accepted EKF update present or absent;
- finite corridor occupancy present or absent.

Report per trip and overall coverage. The upper bound is the number of V1 FP
frames with valid V2 eligibility. If this is small, V2 cannot materially raise
F1 regardless of occupancy accuracy.

**Pass for continuing:** enough eligible V1 FPs exist to offset any loss of
eligible V1 TPs. Report counts before interpreting a possible F1 gain.

## Step 2: T05 association taxonomy

**Question:** Why did T05 have zero low-occupancy suppressions despite many V1
classical danger frames?

Classify each T05 V1 false-positive danger frame as one of:

1. no YOLO road-user box;
2. YOLO box exists but no classical-track IoU match;
3. match exists but detector stereo depth unavailable;
4. match and depth exist but EKF rejected/unavailable;
5. valid occupancy, but occupancy not low;
6. V1 frame is not actually a false positive.

This identifies whether the bottleneck is detection coverage, association,
stereo measurement, state filtering, or risk logic.

## Step 3: Lost-danger episode trace

**Question:** Does the existing event FSM, not path occupancy, cause recall
loss?

Select 5–8 episodes where V1 has a correct sustained danger interval but V2
event-to-TTC does not. For each frame log:

- V1 raw union TTC;
- classical/detector source;
- IoU match and occupancy;
- low-occupancy floor decision;
- V2 FSM state;
- V2 submitted TTC;
- ground-truth TTC.

The trace must distinguish raw path gating from FSM persistence. V2 does not
switch to an EKF-derived TTC; it preserves V1 TTC or applies the fixed `2.0 s`
floor before FSM processing.

## Step 4: Pre-register one no-FSM framewise ablation

Only after Steps 1–3, write a separate fixed protocol:

- same IoU `0.30`, occupancy `0.50`, and finite floor `2.0 s`;
- no FSM on the experimental TTC submission path;
- no modifications after the result;
- report organizer F1, MAE, TP/FP/FN, per-trip metrics, coverage, and latency.

This isolates framewise scoring compatibility from event-hysteresis behavior.

## Step 5: Decision

| Evidence | Decision |
|---|---|
| Low V2 FP coverage | Stop V2-for-F1; fix T05 semantic/classical association or retain V2 for deployment explanation. |
| Adequate coverage, FSM causes loss, no-FSM passes gates | Keep V1 scorer lane; evaluate V2 separately as an alert policy. |
| Adequate coverage, no-FSM also fails | Stop occupancy-to-TTC conversion; CPA remains explanatory only. |
| No-FSM passes all pre-registered gates | Validate on external/synthetic turn holdout before promotion. |

## Completion criteria

The diagnostic cycle is complete when all three evidence reports and one
pre-registered ablation result are available, followed by one of the above
decisions. It is not complete merely because a threshold produces a better
six-trip score.
