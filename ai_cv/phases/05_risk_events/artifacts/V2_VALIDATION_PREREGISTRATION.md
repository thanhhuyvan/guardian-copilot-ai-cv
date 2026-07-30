# V2 validated-state protocol — pre-registered

## Scope

V1 remains frozen. This protocol validates object state before any V2 TTC or
risk-event output is allowed to change. It supersedes unlabelled threshold
experiments; no IoU, corridor, covariance, or temporal parameter may be swept
against six-trip F1.

## Blind label set

Generate 72 clips: six V1-danger/true-danger and six V1-danger/non-danger
examples per practice trip. The reviewer sees only the target clip and box,
not TTC, prediction source, candidate name, or challenge target.

Required fields are defined in `OBJECT_EVENT_LABEL_SCHEMA.md`, including
`candidate_type`. The latter separates two mechanisms:

1. a real road user with ambiguous event ownership/path relation; and
2. a spurious classical candidate such as static structure, shadow/reflection,
   or stereo artifact.

At least 24 clips must receive independent second-review labels. Only
high/medium-confidence non-provisional labels may enter validation.

## Stage-validation metrics

Before a V2 risk gate is built, report on held-out trip labels:

- candidate-type confusion matrix;
- road-user association coverage and precision;
- on-path versus non-on-path classification;
- CPA error only where a reviewer provides a numeric CPA;
- track continuity, innovation residuals, and missing-measurement rate.

Use leave-one-trip-out rotation for any calibration. The held-out trip cannot
select an association, process-noise, or path-risk parameter.

## Build and selection rule

Only if state validation is acceptable, build one fixed candidate with:

1. multi-cue association that may explicitly return `no_real_object`;
2. ego-compensated per-track planar EKF;
3. covariance-derived path occupancy at CPA; and
4. TTC retained independently of event gating.

Promotion requires all of the following:

- labelled state metrics improve or hold against V1;
- overall F1 is at least 0.654;
- T01 and T05 F1 do not decline;
- critical TTC MAE does not exceed 29.993 s;
- no determinism, missing-frame, or NaN failure; and
- latency is recertified under the chosen deployment protocol.

## External holdout

After internal state validation, test the frozen candidate on untouched CARLA
scenarios with full simulator ground truth: cut-in, lead brake versus cruising,
turn/side traffic, roadside pedestrians, adverse weather, and occlusion.
Redacted challenge trips remain deployment/submission inputs, not accuracy
ground truth.
