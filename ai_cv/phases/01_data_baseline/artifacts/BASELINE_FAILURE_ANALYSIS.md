# Baseline Failure Analysis

## What the baseline actually computes

For every frame, the organizer baseline:

1. Runs SGBM stereo disparity over the complete left/right pair.
2. Crops one fixed rectangle: x=35-65% and y=50-85% of the image.
3. Converts all valid ROI disparities to depth and takes one median value.
4. Fits a depth-versus-time line over the latest five valid median depths.
5. Uses the negative slope as closing speed.
6. Returns `depth / closing_speed`, or `inf` when closing speed is at most
   0.3 m/s.

It never detects, classifies, tracks, or associates an individual object with
the ego collision path.

## Direct evidence from selected failures

| Outcome | Trip/frame | GT | Prediction | Five ROI median depths (m) | Estimated closing speed |
|---|---|---:|---:|---|---:|
| FN | T01 #324 | 1.06 s | inf | 7.4927, 7.6418, 7.7576, 7.4563, 7.5294 | 0.2240 m/s |
| FN | T05 #469 | 1.44 s | inf | 7.4563, 7.4203, 7.3846, 7.3846, 8.0000 | -2.1034 m/s |
| FN | T06 #146 | 0.84 s | inf | 2.5946, 2.4935, 2.6528, 3.7327, 3.4058 | -5.7231 m/s |
| FP | T03 #293 | inf | 0.17 s | 11.9070, 11.7252, 7.4563, 6.0000, 6.0235 | 34.9842 m/s |
| FP | T05 #314 | inf | 0.29 s | 11.3778, 7.4563, 11.5489, 7.4203, 6.0711 | 21.2986 m/s |
| TP | T04 #265 | 1.75 s | 1.71 s | 7.3846, 7.0459, 6.8571, 6.8267, 6.5362 | 3.8322 m/s |

## Why it fails

### 1. One median erases the dangerous object

The ROI mixes road, background, guardrail, vehicle, pedestrian, and motorcycle
pixels. A small but dangerous object can be visible inside the rectangle without
controlling the median. In T01 #324, a crossing pedestrian has GT TTC 1.06 s,
but the ROI median remains around 7.5 m and barely changes, so the baseline emits
`inf`.

### 2. Differentiating noisy depth creates impossible closing speeds

The five-frame history covers roughly 0.2 seconds at 20 FPS. Small stereo or
median changes are divided by a very short time interval. On empty night road in
T03 #293, the median sequence drops from 11.9 m to about 6.0 m and becomes a
physically implausible 34.98 m/s closing speed. The result is a false alarm of
0.17 s TTC while ground truth is `inf`.

### 3. Correct visible threats become `inf` when depth moves the wrong way

SGBM/ROI noise can make median depth increase even while the real target is
closing. T06 #146 visibly contains a motorcycle immediately ahead and has GT TTC
0.84 s, but the estimated closing speed is -5.72 m/s, so the thresholding rule
forces the output to `inf`.

### 4. No object identity or collision-path reasoning

The baseline cannot distinguish:

- A lead vehicle from road texture.
- A pedestrian crossing the ego path from a pedestrian safely off-path.
- A motorcycle from background disparity.
- Ego motion from target-relative motion.

This explains both sides of the error distribution: 106/283 critical frames are
predicted as `inf`, while T03 alone produces 169 false-positive danger frames.

### 5. SGBM confidence is too weak

The only effective acceptance rule is at least 100 valid ROI depth pixels. There
is no left-right consistency confidence, object-level depth dispersion, temporal
innovation gate, or physical acceleration bound. A map can therefore contain
many valid-looking pixels and still produce the wrong median motion.

### 6. The evaluator heavily penalizes `inf` misses

For MAE, the official evaluator replaces `inf` with 99 s. A critical frame near
1-2 s predicted as `inf` contributes roughly 97-98 s absolute error. Repeated
misses drive overall critical MAE to 38.046 s. The MAE component of the composite
score is already zero once MAE reaches 5 s.

## Visual evidence

Generated locally under:

`ai_cv/outputs/reports/baseline_official/visualizations/`

- `ttc_timelines.png`: red prediction is extremely spiky and poorly follows GT.
- `danger_confusion_counts.png`: TP/FP/FN counts per trip.
- `ttc_scatter.png`: predictions are poorly calibrated against the ideal line.
- `failure_frame_montage.png`: representative FN/FP/TP images with fixed ROI.
- `stereo_roi_diagnostics.png`: disparity and ROI depth for FN/FP/TP examples.
- `selected_temporal_diagnostics.csv`: exact five-depth regression state.

## Design requirements for the next baseline

1. Detect collision-relevant objects instead of using one fixed ROI.
2. Track object identity across time and estimate per-object relative motion.
3. Use robust object depth (trimmed median/percentile plus uncertainty), not a
   mixed-scene median.
4. Fuse stereo motion with ego telemetry and impose physical motion gates.
5. Emit calibrated confidence and degraded/unknown states when stereo is weak.
6. Evaluate per object/event as well as per frame to prevent alert flicker.
