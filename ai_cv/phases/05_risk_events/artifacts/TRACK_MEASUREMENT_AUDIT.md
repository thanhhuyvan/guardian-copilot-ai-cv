# Track-level measurement audit — Phase 21

## Motivation

The six-trip V1 score is frame-level TTC/danger F1. It cannot distinguish a
wrong object identity from a correct object with an incorrect depth or closing
rate. Earlier screens rejected hard-IoU association, containment-only
association, a generic detector veto, and a binary longitudinal EKF gate.

This audit tests a narrower claim before another F1 run: does the existing
object depth change disagree with independent YOLO image-scale change more in
false classical alerts than in true classical alerts?

## Fixed diagnostic

For each corrected Phase 17 classical-source danger frame, match the best
available YOLO box only for analysis. For consecutive frames of a track,
compute the absolute log residual between stereo range change and the
inverse-square box-area expectation. No value is used as a gate, score, or
filter parameter.

## Result

| Group | Frames | Detection available | Median residual | P95 residual |
|---|---:|---:|---:|---:|
| Classical false alerts | 89 | 88.8% | 0.0175 | 0.1555 |
| Classical true-danger alerts | 123 | 93.5% | 0.0184 | 0.1377 |

The groups overlap almost completely. In fact, false alerts have a marginally
*lower* median residual. A depth-versus-box-scale consistency gate would not
separate false alerts from true danger and must not be added to V1 or V2.

The complementary planar residual audit confirms that object-depth noise is
large and range-dependent: T05 has only seven usable detector-depth track
histories, with depth-residual sigma P50 `0.67 m` and P95 `4.68 m`. It is
evidence for uncertainty-aware state estimation, but not enough to select an
EKF measurement model or a risk threshold.

## New validation test

`outputs/phase21_track_measurement_audit/track_validation_labels.csv` contains
30 balanced cross-trip track snapshots (five per trip). The reviewer labels
path relation, closest-approach distance, and occlusion. Those labels validate
the *state/path estimate* independently of organizer frame F1; they do not
train a model or tune a gate.

## Decision rule

1. If labels show the selected track/path estimate is wrong, repair
   measurement/data association before risk logic.
2. If labels show it is correct but the event is false, audit TTC policy and
   ego-motion compensation next.
3. Run official F1/MAE only after one of those causal repairs is specified.
