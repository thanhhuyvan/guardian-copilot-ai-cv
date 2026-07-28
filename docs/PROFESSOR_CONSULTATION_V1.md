# GuardianCoPilot V1: method, ablations, and request for guidance

## One-paragraph summary

GuardianCoPilot V1 is a real-time collision-risk monitor for stereo driving
video. It uses YOLO26n to propose road users, dual SGBM stereo to estimate
metric depth, causal tracking to estimate relative closing speed and TTC, and
a deterministic risk-event state machine. The selected conservative fusion
passes the project's latency and reproducibility gates, but its macro danger
F1 is only about `0.65`. The unresolved failures are not detector absence:
they are long, plausible-but-wrong TTC estimates from stereo/tracking, mostly
in T01 (turn/side traffic) and T05 (classical depth-track false alerts). We
need guidance on the next *generalizable* method, rather than more threshold
tuning on six short trips.

## V1 method

Input is a synchronized `640x360` stereo pair plus current ego telemetry
(speed, longitudinal acceleration, lateral acceleration). No ground-truth TTC,
event label, trip identity, frame index, or future frame is used at runtime.

```text
left image ── YOLO26n ── road-user boxes ── box stereo depth ── causal tracks ┐
                                                                                ├─ TTC/risk event
stereo pair ─ dual SGBM ─ ground removal ─ obstacle components ─ causal tracks ┘
```

1. **YOLO26n proposal branch.** Keep road-user classes (car, truck, bus,
   motorcycle, bicycle, person). Estimate robust near-object depth within each
   box from stereo disparity.
2. **Classical branch.** Compute left/right SGBM disparity, reject ground,
   extract obstacle components, and track components in a collision corridor.
3. **TTC.** For each causal track, fit robust depth-vs-time motion and derive
   TTC from current depth / positive closing speed. Apply fixed depth,
   confidence, residual, and physical-speed guards.
4. **Conservative union.** Retain the classical TTC, except where only the
   detector branch independently reports a danger TTC (`<2 s`).
5. **Risk events.** Deterministic hysteresis converts frame TTC into
   SAFE/WARNING/DANGER events; it does not change the benchmark labels.

## Evaluation protocol

- Six practice trips, `600` frames/trip (`3,600` frames total).
- Danger label: ground-truth TTC `<2 s`; reported F1 is macro-average by trip.
- Critical TTC error is evaluated separately for ground-truth TTC `<3 s`.
- Dataset is read-only. All prediction files are generated from the same
  globally fixed configuration.
- Deployment benchmark: `100` warm-up frames, five complete repeats, GPU YOLO
  and CPU stereo in parallel, disk I/O excluded from compute latency.

## Selected V1 result

The frozen integrated deployment certification produced:

| Metric | Result |
|---|---:|
| Macro danger-F1 | `0.6579` |
| Macro composite | `42.8817 / 100` |
| Critical TTC MAE | `29.9929 s` |
| Compute P95 | `65.91 ms` (target `<=75 ms`) |
| Peak VRAM | `460.68 MB` (6 GB RTX 3060) |
| Determinism | `0 / 43,200` repeat mismatches |

Per-trip F1: T01 `.292`, T02 `.757`, T03 `.710`, T04 `.860`, T05 `.509`,
T06 `.821`. Thus, the mean is acceptable for a latency-first V1 but is not
strong enough to claim robust collision prediction.

## Main ablations

| Candidate | Macro F1 | Key outcome |
|---|---:|---|
| Guarded classical SGBM TTC | `.563` | Baseline; weak proposals in difficult scenes. |
| Detector-owned TTC | `.632` | YOLO boxes recover many missed road users. |
| Fixed conservative union (V1) | `.658` | Best global F1/composite/critical MAE. |
| Learned confidence router, leave-one-trip-out | `.488` | Overfits reliability pattern; reject. |
| Heavier YOLO26s | `.655` | No meaningful gain; reject. |
| Classical minimum-closing-speed gates | `.624–.670` | Local F1 changes but MAE/generalization worsens; reject. |
| YOLO association late-fusion gate | about `.655` projected | Removes only one T01 FP; reject. |
| Path-intersection gate from box motion + ego curvature | T01 `.292→.279`, T05 `.509→.491` | Removes true alerts too; reject. |
| LightStereo-S zero-shot / calibrated scale | F1 `.172–.201` | Not competitive with SGBM on this data; reject. |

### Generalization test

We used leave-one-trip-out validation: tune/train on five trips and evaluate
on the sixth. A mini-fold overfit test showed that training folds can be
memorized, but the leave-one-trip-out router failed on all folds. Therefore
capacity is not the main limit; the six-trip dataset has too few diverse
disagreement/critical examples for a learned selector to generalize.

## Failure diagnosis

### T01: turns and side traffic

The system creates sustained false danger alerts around turns. A side vehicle
can move closer in camera depth while its future path does not intersect the
ego path. The current tracker estimates longitudinal depth change, but it has
no reliable ego-motion compensation or object trajectory/path-intersection
model. Simple path projection from YOLO box centres, stereo depth, and lateral
acceleration was tested and harmed recall because the estimated lateral motion
is too noisy.

### T05: stable classical false alerts

T05 has `47` false-positive danger frames in the selected V1 run. Evidence
shows `45/47` originate from the classical branch. These false components
already overlap YOLO road-user boxes, so requiring a YOLO match does not solve
them. The likely issue is depth/association/motion estimation, not merely a
background stereo blob or a missing detector.

### Why TTC MAE looks high

For a critical ground-truth frame where the system outputs no finite TTC, the
organizer evaluator substitutes `99 s`. Thus a limited number of missed
critical frames dominate MAE even though most real TTC values are 2–3 seconds.
F1 and critical-TTC MAE must be considered together.

## What we deliberately did not do

- No trip-specific rules, target-label lookups, or future-frame smoothing.
- No model selection by weather/trip identity.
- No domain-adaptation model promoted from the tiny practice set.
- No heavier detector promoted without an end-to-end gain.

## Questions for the professor

1. With only stereo, ego IMU-like telemetry, and detector boxes, what is the
   most defensible next method for **collision-path prediction**: calibrated
   ego-motion/visual odometry, object-level 3D Kalman filtering, optical-flow
   scene flow, or a different formulation?
2. Is it better to label a modest number of object tracks with object identity,
   lane/path relation, and relative velocity, or to collect a broader external
   driving dataset before training a learned fusion/trajectory module?
3. Should the project treat this as a detection-and-TTC task, or reframe it as
   an ego-path occupancy / future-collision classification task with TTC as a
   secondary output?
4. Is macro danger-F1 around `.65` with P95 `<75 ms` a reasonable V1 stopping
   point, given the data size, or should deployment be postponed until a
   specific object-level validation set exists?
5. What validation protocol would you require before claiming robustness:
   leave-one-trip-out only, or train/validation/test split by location/weather
   plus a fully external dataset?

## Exact V1 reproduction

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_predeploy_certification.ps1
```

The command runs the frozen candidate and the organizer scorer. Generated
reports remain local under `ai_cv/outputs/phase08_predeploy_final`.
