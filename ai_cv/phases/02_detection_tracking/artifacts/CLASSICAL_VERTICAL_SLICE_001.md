# Stage 2A Classical Object-Centric TTC Vertical Slice

**Status:** COMPLETE  
**Decision:** keep the geometry and causal TTC building blocks; do not promote
connected components as the final target identity source  
**Next:** compare an instance-aware extractor against the strongest classical
and fixed-ROI references

## 1. Research question

Can a causal classical pipeline replace the organizer's fixed road ROI with
object-level observations and recover threats that the ROI statistic misses?

The evaluated vertical slice is:

```text
stereo SGBM
  -> V-disparity ground model
  -> above-ground vertical-support components
  -> causal component association
  -> robust per-track distance and closing speed
  -> collision corridor
  -> TTC
```

No ground-truth depth or TTC is used by the predictor. Provided labels are used
only after inference by the official evaluator.

## 2. Experiment design

- Six practice trips, 3,600 frames in total.
- Four distance summaries were tested end to end:
  - scene-level component 20th percentile;
  - tracked component 20th percentile;
  - tracked component 35th percentile;
  - tracked component median.
- All association and motion estimation is causal.
- The same predictions are scored with the organizer evaluator.
- Six known Stage 1 failure cases are rendered for visual falsification.

Generated predictions, diagnostics, charts and case renders are git-ignored
under:

```text
ai_cv/outputs/benchmarks/phase02a_vertical_slice/
ai_cv/outputs/reports/phase02a/ground_obstacles/
```

## 3. Geometry findings

The V-disparity model removes coherent road support and preserves small
above-ground structures better than the fixed ROI. A vertical-support gate also
removes the close horizontal road band responsible for the T03 #293 false
positive.

Representative component depths:

| Case | Observation |
|---|---|
| T01 #324 pedestrian FN | Retained at 8.44 m median, 8.13 m p20 |
| T03 #293 empty-road FP | False 4 m road band removed; remaining background is 22.26 m median |
| T04 #265 lead-car TP | Stable component at 6.98 m median |
| T05 #314 off-path pedestrian | Still forms components near the corridor boundary |
| T05 #469 lead car | Merged with background: 35.72 m median, 10.74 m p20 |
| T06 #146 motorcycle | Retained at 2.52 m median, 1.78 m p20 |

This is a meaningful gain over the fixed ROI, but it also exposes the next
bottleneck: one physical object can fragment into multiple components, while a
vehicle can also merge with background geometry.

## 4. Full TTC results

| Variant | MAE critical | inv-TTC MAE | F1 | Mean composite | Worst trip |
|---|---:|---:|---:|---:|---:|
| Official baseline | 38.046 | 0.2982 | 0.220 | 19.7 | 5.0 |
| Stage 1 robust fixed ROI | 43.114 | 0.1896 | 0.258 | 32.2 | 16.9 |
| Scene p20 | 32.878 | 0.2706 | 0.376 | 26.5 | 3.8 |
| Track p20 | 27.505 | 0.2627 | 0.392 | 26.4 | 3.2 |
| Track p35 | 22.125 | 0.2455 | **0.402** | 28.7 | 4.6 |
| Track median | 23.119 | 0.2331 | 0.384 | 29.2 | 9.6 |

The object-centric slice materially improves danger recall and critical-frame
MAE, but it does not beat the robust fixed-ROI reference on composite or
worst-trip score.

Danger confusion totals:

| Variant | TP | FP | FN |
|---|---:|---:|---:|
| Official baseline | 80 | 436 | 124 |
| Track p35 | **135** | 340 | **69** |

Track p35 recovers 55 additional danger true positives and removes 55 false
negatives. Its F1 rises from 0.220 to 0.402. This validates object-level
geometry as a useful direction even though the current identity mechanism is
not stable enough.

## 5. Where it fails

T03 remains the worst trip. Track p35 produces 139 danger false positives on
that trip. Selected-track identity switches per 100 finite predictions are also
high on T02 (roughly 40-50), T03 (roughly 14-16) and T06 (roughly 16-19).

The TTC timelines show:

- T04 follows the lead-car ground truth comparatively well;
- T05 and T06 recover threats missed by the fixed-ROI reference;
- T03 is highly discontinuous because road/background fragments repeatedly
  become new targets;
- component count and ground-model confidence alone cannot cleanly distinguish
  the false-positive trip from difficult true-positive trips.

A simple confidence threshold is therefore not promoted. It would suppress
some T03 errors but also remove valid low-confidence observations, especially
on T02.

## 6. Runtime

Full-run local CPU timings:

| Stage | P50 | P95 | P99 |
|---|---:|---:|---:|
| Left SGBM | 15.31 ms | 19.40 ms | 22.74 ms |
| Right SGBM | 15.34 ms | 18.99 ms | 21.81 ms |
| Ground model | 14.77 ms | 18.98 ms | 28.17 ms |
| Components | 3.06 ms | 4.78 ms | 5.90 ms |
| Tracking/TTC | 1.23 ms | 2.35 ms | 3.11 ms |
| Total measured compute | 54.87 ms | 69.32 ms | 87.39 ms |

The total timer includes image loading, so it is a conservative end-to-end
research measurement rather than a pure kernel benchmark. P50 throughput is
about 18.2 FPS, below the 20 FPS input rate. The right matcher and ground fit
are the largest new costs.

## 7. Keep, reject and next gate

### Keep

- V-disparity ground estimation and explicit degraded confidence.
- Vertical-support obstacle proposals.
- LR residual/coverage as soft confidence, not a hard mask.
- Causal per-track robust distance/closing-speed estimation.
- Path-aware collision corridor.
- The full official-evaluator and latency harness.

### Do not promote

- Connected components as final object identity.
- More morphology-only threshold tuning.
- A global ground-confidence cutoff.
- Any claim that the current classical vertical slice is the new score
  baseline; the Stage 1 robust fixed ROI still has the stronger composite and
  worst-trip result.

### Next high-impact experiment

Use an instance-aware extractor to provide persistent object boundaries and
class labels, then reuse the validated stereo depth, causal tracking, corridor
and evaluator. Compare it against both:

1. Stage 1 robust fixed ROI: score/robustness reference.
2. Stage 2A track p35: object-recall reference.

Semantic segmentation may be tested as a secondary road/free-space mask, but it
does not directly solve the measured identity fragmentation. The primary
comparison should therefore be lightweight object detection or instance
segmentation.

