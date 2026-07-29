# GuardianCoPilot V2: CPA State-Estimation Study

## Motivation

V1 uses dual-SGBM depth, YOLO26 road-user detection, tracking, TTC, and a
conservative union rule. Its official six-trip result is F1 `0.654`, critical
TTC MAE `29.993 s`, and composite `42.8/100`.

The two main failure clusters were different from an ordinary detector miss:

- **T01:** turning or side traffic can become closer in camera depth while its
  future path does not enter the host path. Raw depth-rate then creates false
  danger events.
- **T05:** noisy classical stereo tracks can create plausible closing TTC even
  when a semantic road user is present.

V2 tested whether an ego-motion-compensated, object-level planar state could
separate **closing in depth** from **being on a collision path**.

## Research question

Can a detector-associated stereo EKF produce a corridor-occupancy probability
that is higher for on-path objects than for adjacent or diverging objects,
without training a new learned fusion model on only six trips?

## Fixed V2 method

1. YOLO boxes define object identity.
2. SGBM depth is estimated with robust median disparity inside each YOLO box.
3. Each object gets a planar forward/lateral constant-velocity EKF.
4. Ego speed and lateral acceleration produce yaw-rate; propagation uses an
   exact constant-yaw bicycle arc.
5. The EKF covariance gives Gaussian probability that the object occupies the
   `±1.75 m` ego corridor at predicted CPA.
6. Measurement covariance is residual-derived, not tuned for F1:

| Range | Longitudinal sigma | Lateral sigma |
|---|---:|---:|
| `<10 m` | `2.0 m` | `0.5 m` |
| `10–20 m` | `2.0 m` | `1.0 m` |
| `>20 m` | `4.0 m` | `2.0 m` |

V1 stays frozen. V2 runs in shadow mode until validation passes.

## Validation study

Thirty detector-associated tracks were sampled: five per trip. T01/T05 cover
known false-alert conditions; T02–T04/T06 include known-danger anchors. Review
overlays show the selected YOLO/stereo track, not an anonymous stereo component.

The labels are **provisional visual labels**, not simulator target-ID labels.
They are useful for mechanism screening, but do not support a broad robustness
claim.

| Path label | Count | Mean V2 corridor occupancy |
|---|---:|---:|
| On path | 19 | `0.512` |
| Adjacent | 7 | `0.078` |
| Diverging | 4 | `0.156` |

The required direction holds: reviewed on-path tracks receive substantially
higher occupancy than non-path tracks. This supports the **state-estimation
mechanism**, not a production safety claim.

## Event-to-TTC experiment

The organizer scores framewise danger from `predicted_ttc <2.0 s`, whereas V2
is designed as a temporally stable risk-event lane. To compare them once, the
following policy was pre-registered before scoring:

- classical danger only;
- detector/classical match IoU at least `0.30`;
- occupancy below `0.50` supplies finite non-danger TTC `2.0 s`;
- no match or uncertain EKF preserves V1 TTC;
- existing risk FSM provides all persistence; no new debounce.

| Candidate | F1 | Critical TTC MAE | Composite |
|---|---:|---:|---:|
| V1 conservative union | `0.654` | `29.993 s` | `42.8` |
| V2 event-to-TTC | `0.428` | `30.547 s` | `35.8` |

V2 fails every pre-registered deployment gate and is rejected as an official
TTC candidate.

## Why V2 F1 fell

The failure is primarily a **metric/output mismatch**. The existing event FSM
deliberately suppresses short alerts; the organizer expects a correct decision
on every frame. Even without a low-occupancy suppression, danger-frame count
fell sharply:

| Trip | V1 danger frames | V2 event-TTC danger frames |
|---|---:|---:|
| T01 | 38 | 7 |
| T02 | 19 | 4 |
| T03 | 33 | 16 |
| T04 | 53 | 34 |
| T05 | 75 | 56 |
| T06 | 57 | 29 |

Thus the experiment does not justify tuning IoU, probability, floor TTC, or
FSM timing on these six trips. That would overfit the benchmark.

## What can be inferred

1. **Object association is prerequisite.** Earlier anonymous-component labels
   could not identify which object the EKF was measuring. YOLO-box association
   made the CPA question testable.
2. **Range-dependent uncertainty matters.** Far stereo measurements show much
   larger residuals, so one fixed depth covariance is not defensible.
3. **Path geometry is useful but not an organizer-TTC replacement.** V2 can
   support deployment alert explanation and conservative suppression of
   clearly off-path traffic, while V1 remains the scored TTC source.
4. **Framewise and eventwise metrics must remain separate.** A stable alert
   policy should be assessed with event precision/recall, time-to-alert, and
   false-alert duration; it cannot be judged solely by a framewise TTC scorer.
5. **Robustness remains unproven.** The next credible validation is a small
   external or synthetic turn/side-traffic holdout with target/path labels.

## Decision

Keep V1 as the official and deployable TTC baseline. Retain V2 as a documented
research lane for path-aware alerting, not as a replacement TTC submission.

Related artifacts:

- `ai_cv/phases/05_risk_events/artifacts/V2_PREREGISTERED_VALIDATION.md`
- `ai_cv/phases/05_risk_events/artifacts/V2_EVENT_TO_TTC_PREREGISTRATION.md`
- `ai_cv/phases/05_risk_events/artifacts/V2_EVENT_TO_TTC_RESULT.md`
