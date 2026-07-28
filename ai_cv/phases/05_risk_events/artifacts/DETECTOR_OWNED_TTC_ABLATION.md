# Phase 05A detector-owned TTC ablation

Date: 2026-07-28  
Branch: `research/phase-05-risk-events`

## Question

Can an object detector own the proposal and identity stages while SGBM supplies
metric depth inside each detected road-user box? This directly tests the
failure found on T03, where most missed danger frames had no stereo obstacle
component for the tracker to follow.

No target labels, frame identity, or trip-specific thresholds are used during
inference. The first ablation used frozen YOLO26 reference CSVs. The final
deployment benchmark runs the same YOLO26 model live on the RTX 3060. The same
configuration is applied to every trip.

## Fixed pipeline

1. Load a frozen YOLO road-user detection.
2. Estimate depth from the nearer supported disparity mode inside its box.
3. Track the detection boxes causally with Guardian's existing component
   tracker.
4. Estimate closing speed and TTC with the existing robust Theil-Sen fit.
5. Apply the guarded corridor, confidence, depth, residual, and closing-speed
   limits.
6. For the physical-cap variant, also require closing speed to be no greater
   than `min(20 m/s, ego speed + 3 m/s)`.

The ego-speed signal is an allowed runtime input in the starter kit. The cap is
a forward-lane plausibility constraint, not a fitted trip rule. It assumes the
selected corridor object is moving in the same direction; an oncoming-traffic
deployment must replace it with an explicit direction classifier.

## Reproduction

```powershell
.\.venv\Scripts\python.exe `
  ai_cv\phases\05_risk_events\src\evaluate_detector_owned_ttc.py `
  --trips T01-Sample T02-Sample T03-Sample T04-Sample T05-Sample T06-Sample `
  --output-dir ai_cv\outputs\phase05_detector_owned_all `
  --progress-every 200
```

The generated prediction CSVs and official evaluator JSON are ignored runtime
artifacts. The source practice-dataset fingerprint remains
`8310e4eeadcb8518970624913861245fc38072321821809f8674f6a68871d3ad`.

## Official cached-detection result

| Metric | Guarded SGBM baseline | Detector-owned + ego cap | Change |
|---|---:|---:|---:|
| Macro danger-F1 | 0.564 | **0.618** | **+0.054** |
| Composite | 39.7 | **42.4** | **+2.7** |
| Critical-TTC MAE | 44.806 s | **38.119 s** | **-6.687 s** |
| Inverse-TTC MAE | 0.1198 | **0.1020** | **-0.0178** |

| Trip | Baseline F1 | Detector-owned F1 | Delta | TP / FP / FN |
|---|---:|---:|---:|---:|
| T01 | 0.452 | 0.385 | -0.067 | 5 / 11 / 5 |
| T02 | 0.765 | 0.375 | -0.390 | 6 / 8 / 12 |
| T03 | 0.333 | **0.863** | **+0.530** | 22 / 0 / 7 |
| T04 | 0.763 | **0.874** | **+0.111** | 45 / 6 / 7 |
| T05 | 0.261 | **0.396** | **+0.135** | 19 / 42 / 16 |
| T06 | 0.807 | 0.818 | +0.011 | 45 / 5 / 15 |

The uncapped detector-owned policy reached macro F1 `0.583`, composite
`40.86`, and critical-TTC MAE `36.783 s`. The physical cap reduced false
positives from 104 to 72, at the cost of one additional false negative.

## Live end-to-end result

The deployable configuration runs GPU YOLO and CPU SGBM concurrently:

- YOLO26n PyTorch FP32 on `cuda:0`;
- model SHA-256
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`;
- two OpenCV threads and two concurrent left/right SGBM matchers;
- 100 warm-up frames;
- all 3,600 frames, five measured repeats, 18,000 timing rows;
- decoded stereo pair to TTC/risk output; disk loading excluded.

```powershell
.\.venv_yolo26\Scripts\python.exe `
  ai_cv\phases\05_risk_events\src\evaluate_detector_owned_ttc.py `
  --detector-backend yolo26-pytorch `
  --model-path yolo26n.pt `
  --trips T01-Sample T02-Sample T03-Sample T04-Sample T05-Sample T06-Sample `
  --repeats 5 `
  --warmup-frames 100 `
  --opencv-threads 2 `
  --stereo-workers 2 `
  --output-dir ai_cv\outputs\phase05_live_official `
  --progress-every 300
```

| Accuracy metric | Guarded SGBM baseline | Live detector-owned | Change |
|---|---:|---:|---:|
| Macro danger-F1 | 0.564 | **0.632** | **+0.068** |
| Composite | 39.7 | **42.8** | **+3.1** |
| Critical-TTC MAE | 44.806 s | **37.891 s** | **-6.915 s** |
| Inverse-TTC MAE | 0.1198 | **0.1025** | **-0.0173** |

| Trip | Baseline F1 | Live F1 | Delta | TP / FP / FN |
|---|---:|---:|---:|---:|
| T01 | 0.452 | 0.312 | -0.140 | 5 / 17 / 5 |
| T02 | 0.765 | 0.519 | -0.246 | 7 / 2 / 11 |
| T03 | 0.333 | **0.840** | **+0.507** | 21 / 0 / 8 |
| T04 | 0.763 | **0.874** | **+0.111** | 45 / 6 / 7 |
| T05 | 0.261 | **0.429** | **+0.168** | 21 / 42 / 14 |
| T06 | 0.807 | 0.821 | +0.014 | 46 / 6 / 14 |

| Live timing | P50 | P95 | P99 |
|---|---:|---:|---:|
| Full compute pipeline | 53.36 ms | **63.22 ms** | 70.30 ms |
| Concurrent inference wall | 39.29 ms | 48.01 ms | 54.29 ms |
| SGBM stereo | 38.73 ms | 47.40 ms | 53.64 ms |
| YOLO inference | 23.73 ms | 30.36 ms | 33.98 ms |
| Depth/tracking/TTC postprocess | 14.03 ms | 17.24 ms | 19.29 ms |

The latency target is `P95 <= 75 ms`; the measured margin is `11.78 ms`.
A separate six-trip, two-repeat audit made 7,200 exact prediction comparisons
with zero mismatches. A conservative NVML sample measured peak device memory
at `412.68 MB`; PyTorch reported `87.34 MB` peak allocated and `120 MB` peak
reserved. These values pass the 5 GB gate.

The initial `6 OpenCV threads / 1 matcher` live smoke test reached only
`101.65 ms` P95 because concurrent YOLO preprocessing contended with SGBM.
The frozen `2 threads / 2 matchers` configuration reduced the official P95 to
`63.22 ms`; no accuracy threshold was changed during this latency correction.

## Earlier stereo-owned ablations

The preceding T03/T05 gate rejected all three stereo-component-owned variants:

| Candidate | T03 F1 | T05 F1 | Decision |
|---|---:|---:|---|
| Object-depth only | 0.256 | 0.147 | Reject |
| Filtered-motion only | 0.000 | 0.105 | Reject |
| Combined object-centric | 0.263 | 0.026 | Reject |

Changing depth or motion after component extraction cannot recover frames where
component extraction produced no proposal. The detector-owned result validates
the architecture change rather than additional guard tuning.

## Decision

The live comparator passes the macro accuracy, latency, memory, stability, and
determinism gates. Macro F1 exceeds the `0.50` goal and the frozen `0.564`
baseline. It is not yet the sole production default because T01 and T02
regress.
Selecting a method by trip would leak environment identity and is prohibited.

Phase 05 should now implement confidence and event logic with both candidates
available:

- retain guarded SGBM as the safe fallback;
- expose detector depth/track confidence in a backend-neutral frame record;
- choose or fuse candidates only from causal observable confidence;
- evaluate frame metrics and event metrics separately;
- validate confidence/fusion with leave-one-trip-out selection.

This is the stopping point for TTC threshold tuning. The next work is the
deterministic risk-state/event layer, not a larger classifier.
