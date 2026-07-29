# V2 framewise occupancy ablation — rejected

## Corrected, valid result — July 30

After fixing conversion, unchanged pre-registered policy ran once across six
untouched trips. It starts with finalized conservative-union TTC, uses
classical-source danger, IoU `>=0.30`, accepted EKF update, occupancy `<0.50`,
and finite `2.0 s` suppression. No FSM; no parameter changes.

| Candidate | Danger-F1 | Critical TTC MAE | Composite |
|---|---:|---:|---:|
| Frozen V1 conservative union | **0.654** | **29.993 s** | **42.8** |
| Corrected V2 framewise occupancy | 0.547 | 30.119 s | 39.1 |

Reject fixed hard-IoU plus occupancy policy: it fails frozen F1 and MAE gates.
This does not test multi-cue association, which still needs label validation.

> Historical comparison below remains provenance only. It came from the
> invalid pre-fix run and must not be used for decisions.

## Historical comparison

The pre-registered framewise policy was run once on all six untouched practice
trips and scored by the organizer evaluator. It used exactly the rejected
event-policy geometry contract (classical source, raw TTC below `2.0 s`, IoU
at least `0.30`, accepted EKF update, occupancy below `0.50`) but deliberately
removed the event FSM.

| Candidate | Danger-F1 | Critical TTC MAE | Composite |
|---|---:|---:|---:|
| Frozen V1 conservative union | **0.654** | **29.993 s** | **42.8** |
| V2 framewise occupancy | 0.623 | 38.752 s | 42.5 |

| Trip | V1 F1 | Framewise V2 F1 | Result |
|---|---:|---:|---|
| T01 | 0.292 | 0.312 | improves slightly |
| T02 | 0.757 | 0.462 | substantial false suppression |
| T03 | 0.710 | 0.840 | improves |
| T04 | 0.838 | 0.874 | improves |
| T05 | 0.509 | 0.429 | worsens |
| T06 | 0.821 | 0.821 | unchanged |

## Corrected decision

Corrected run rejects this policy. No occupancy, IoU, EKF-noise, or timing
value may be tuned from the six trips.

## What the experiment established

1. Historical pre-fix results were invalid because conversion did not begin
   from finalized union TTC and framewise mode did not attempt association.
2. Corrected hard-IoU occupancy suppression reduces overall F1 to `0.547`.
3. Coverage audit explains why V2 cannot be a broad V1 repair yet: it covers
   only 34/112 V1 false-positive frames (30%) and 66/163 V1 true-positive
   frames (40%). In T05, 28/47 V1 false positives have no classical-to-YOLO
   association; the remaining 17 are associated but not low occupancy.

The next valid research step is measurement and data-association repair with
track-level verified labels—not another risk-gate threshold experiment. V1
remains the selected deployable baseline.

## Reproducibility

Ignored runtime artifacts: `ai_cv/outputs/phase14_v2_framewise_full/`.
The source contract is `V2_FRAMEWISE_OCCUPANCY_PREREGISTRATION.md` and the
runner flag is `--experimental-v2-event-framewise` (default off).
