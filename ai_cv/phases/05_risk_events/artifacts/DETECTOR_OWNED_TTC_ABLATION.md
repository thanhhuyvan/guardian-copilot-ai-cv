# Phase 05A detector-owned TTC ablation

Date: 2026-07-28  
Branch: `research/phase-05-risk-events`

## Question

Can an object detector own the proposal and identity stages while SGBM supplies
metric depth inside each detected road-user box? This directly tests the
failure found on T03, where most missed danger frames had no stereo obstacle
component for the tracker to follow.

No target labels, frame identity, or trip-specific thresholds are used during
inference. The input detections are the frozen YOLO26 reference CSVs. The same
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

## Official six-trip result

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

The comparator passes the **macro** accuracy safeguards and demonstrates real
capacity: macro F1 exceeds the `0.50` goal and the frozen `0.564` baseline.
It is not yet the production default because T01 and especially T02 regress.
Selecting a method by trip would leak environment identity and is prohibited.

Phase 05 should now implement confidence and event logic with both candidates
available:

- retain guarded SGBM as the safe fallback;
- expose detector depth/track confidence in a backend-neutral frame record;
- choose or fuse candidates only from causal observable confidence;
- evaluate frame metrics and event metrics separately;
- measure live YOLO latency before deployment promotion.

This is the stopping point for TTC threshold tuning. The next work is the
deterministic risk-state/event layer, not a larger classifier.
