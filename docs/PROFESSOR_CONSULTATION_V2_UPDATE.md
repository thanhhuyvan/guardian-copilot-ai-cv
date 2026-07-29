# GuardianCoPilot V2 update: filtered TTC experiment and decision request

## Decision summary

GuardianCoPilot V1 remains the selected deployment baseline. It uses YOLO26n
road-user proposals, dual SGBM stereo depth, causal tracking, conservative TTC
union, and deterministic event hysteresis. Its certified organizer result is
about danger-F1 `0.654–0.658`, critical-TTC MAE `29.993 s`, compute P95
`65.91 ms`, and peak VRAM `461 MB`.

We tested the first V2 hypothesis: replace raw classical longitudinal TTC with
an innovation-gated planar Kalman estimate. The global result was worse, so it
is rejected and remains experimental/default-off.

## Why V2 was proposed

The two dominant V1 weaknesses are:

- **T01:** turn/side-traffic false alerts. Camera depth can decrease while the
  target does not enter the ego path.
- **T05:** sustained classical stereo-track false alerts. `45/47` false danger
  frames are classical-branch outputs and already overlap YOLO road-user boxes.

The intended V2 architecture is an ego-motion-compensated planar/Frenet track
state, followed by closest-point-of-approach (CPA) and lateral path-overlap
risk. It is model-based, causal, and not trained on the six trips.

## V2 work completed

1. Camera box-centre plus stereo depth -> forward/lateral measurement.
2. Telemetry yaw-rate estimate: `yaw_rate = lateral_accel / speed`, disabled
   near standstill.
3. Four-state per-track planar Kalman core: forward, lateral, and velocities.
4. Mahalanobis innovation logging in shadow mode; no V1 output changed.
5. CPA/path-risk and independent looming/tau cue scaffolds.
6. V-disparity ground-model reliability function. V1 already used a
   V-disparity ground line; this is an added reliability gate, not a replacement.
7. Turn/straight scenario manifest for all 3,600 frames.

## Shadow evidence

The T05 all-classical-track shadow replay captured 90 tracks, with 25 usable
residual estimates:

| Residual | P50 | P90 | P95 |
|---|---:|---:|---:|
| Depth (m) | .050 | 1.563 | 2.560 |
| Lateral (m) | .039 | .509 | .560 |

An EKF shadow gate rejected 26.1% of T01 track updates and 55.7% of T05
updates. This confirms substantial classical-track inconsistency, but does not
prove that rejecting all such updates is safe.

## Rejected EKF-TTC experiment

Experimental policy: use accepted EKF longitudinal TTC for a classical danger
track; otherwise fall back to detector TTC. This was intentionally run on all
six trips and scored by the organizer evaluator.

| Metric | V1 selected | EKF longitudinal gate |
|---|---:|---:|
| Overall danger-F1 | `.654` | `.642` |
| Critical-TTC MAE | `29.993 s` | `36.705 s` |
| Composite | `42.8` | `42.6` |

| Trip | V1 F1 | EKF gate F1 |
|---|---:|---:|
| T01 | `.292` | `.350` |
| T02 | `.757` | `.500` |
| T03 | `.710` | `.840` |
| T04 | `.860` | `.885` |
| T05 | `.509` | `.447` |
| T06 | `.821` | `.832` |

### Interpretation

Longitudinal-only EKF gating improves the known T01 false positives, but
removes true alerts in T02 and T05. It is not an ego-motion-compensated Frenet
or CPA method yet. The correct conclusion is **not** to tune its threshold;
that would repeat the small-data overfitting problem.

## Recommended next decision

Freeze V1 for deployment within a restricted operational claim. Do not add
another scalar TTC gate. Before promoting V2, choose one of these paths:

1. Implement the full ego-motion prediction inside the planar filter, then
   evaluate filtered CPA/lateral path occupancy as the risk condition.
2. First label a modest stratified set of tracks with path relation and CPA;
   use it to validate the state estimator before changing danger outputs.
3. Use a small external/synthetic turn-and-side-traffic holdout before making a
   general robustness claim.

## Questions for the professor

1. Is filtered CPA/path occupancy with a bicycle-model ego transform the right
   next model-based step, or should track labels be obtained first?
2. What minimum track-level annotation protocol would be sufficient: identity,
   lateral path relation, CPA, and occlusion state?
3. Should V1 be presented as a latency-certified baseline restricted to its
   observed operational domain, while V2 remains research work?
4. Is a small external/synthetic turn dataset acceptable as a robustness
   holdout when no object-level labels exist in the competition data?

## Reproduction

```powershell
# V1 certification
powershell -ExecutionPolicy Bypass -File .\scripts\run_predeploy_certification.ps1

# V2 EKF experiment (default-off in normal V1 execution)
.\.venv_yolo26\Scripts\python.exe `
  ai_cv\phases\05_risk_events\src\evaluate_detector_owned_ttc.py `
  --detector-backend yolo26-pytorch --model-path yolo26n.pt `
  --integrated-union-events --experimental-v2-ekf-ttc-gate
```
