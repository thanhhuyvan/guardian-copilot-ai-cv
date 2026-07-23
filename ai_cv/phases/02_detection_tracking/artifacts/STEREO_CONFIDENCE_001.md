# Stage 2A Component 1 — SGBM Confidence and Left-Right Consistency

**Status:** COMPLETE  
**Decision:** retain LR residual as a confidence signal; reject hard masking as
the TTC depth feature  
**Next component:** ground-plane/V-disparity removal

## 1. Question

Can explicit left-right (LR) consistency remove bad SGBM pixels and improve the
organizer fixed-ROI TTC baseline before obstacle extraction?

For a valid left disparity:

```text
d_left(x_left) = x_left - x_right
x_right = x_left - d_left(x_left)
residual = |d_left(x_left) + d_right(x_right)|
```

A pixel passes the experiment when the sampled right disparity is valid and the
residual is at most 1.0 px. The check is causal and uses only the current stereo
pair.

## 2. Evidence set

- 72 frames: every 50th frame across all six practice trips.
- All 72 sampled frames coincide with provided depth keyframes.
- Six exact Stage 1 FN/FP/TP cases were rendered in detail.
- Full 3,600-frame TTC ablation was evaluated with the official evaluator.
- Ground truth TTC and provided depth are evaluation-only.

Generated visual evidence is local and git-ignored under:

`ai_cv/outputs/reports/phase02a/stereo_confidence/`

## 3. Pixel/depth findings

Across the 72 sampled frames:

| Measure | Result |
|---|---:|
| Median image pixels with valid left disparity | 60.16% |
| Median LR-pass share among valid image pixels | 78.31% |
| Median LR-pass share among valid organizer-ROI pixels | 95.08% |
| Frames with lower depth abs-relative error after LR filtering | 55/72 |
| Mean raw depth abs-relative error | 5.09% |
| Mean LR-filtered depth abs-relative error | 4.64% |
| Frames where ROI median moves by more than 0.5 m | 9/72 |

LR consistency lowers the mean depth error by about 8.8%, mainly by improving
high-error frames. It does not uniformly improve the median frame: median error
is 4.29% raw versus 4.33% filtered. T03 improves substantially, while T06
slightly regresses.

The organizer ROI has much higher consistency than the full image because road
pixels are usually geometrically coherent. This is the key limitation: LR
consistency answers “is this stereo match plausible?”, not “is this pixel a
collision-relevant obstacle?”.

## 4. Failure-case findings

| Case | ROI LR-pass/valid | Raw median | Filtered median | Interpretation |
|---|---:|---:|---:|---|
| T01 #324 FN pedestrian | 98.2% | 7.53 m | 7.49 m | Nearly all road pixels pass; pedestrian is still erased |
| T03 #293 FP empty night road | 93.2% | 6.02 m | 6.02 m | Filtering does not remove the false median transition |
| T04 #265 TP lead car | 93.9% | 6.54 m | 6.51 m | Stable good case is preserved |
| T05 #314 FP pedestrian off-path | 92.8% | 6.07 m | 8.13 m | Mask materially changes the mixed-scene statistic |
| T05 #469 FN lead car | 92.6% | 8.00 m | 8.00 m | No useful change |
| T06 #146 FN motorcycle | 77.0% | 3.41 m | 4.50 m | Hard mask removes close motorcycle/edge support |

T06 demonstrates the safety risk of treating consistency as a binary truth:
occlusion boundaries and thin object structure are precisely where a relevant
target can lose stereo support.

## 5. Full TTC ablation

| Variant | MAE critical | inv-TTC MAE | F1 | Mean composite | Worst trip |
|---|---:|---:|---:|---:|---:|
| Official replay | 38.046 | 0.2982 | 0.220 | 19.7 | 5.0 |
| Hard LR mask + official temporal | 37.767 | 0.3188 | 0.189 | 17.8 | 4.4 |
| Hard LR mask + robust temporal | 43.583 | 0.1937 | 0.249 | 31.8 | 15.7 |
| Robust temporal without LR mask (Stage 1) | 43.114 | 0.1896 | 0.258 | 32.2 | 16.9 |

Danger confusion totals:

| Variant | TP | FP | FN |
|---|---:|---:|---:|
| Official replay | 80 | 436 | 124 |
| Hard LR mask + official temporal | 78 | 474 | 126 |
| Hard LR mask + robust temporal | 78 | 253 | 126 |
| Robust temporal without LR mask | 81 | 240 | 123 |

Hard filtering alone reduces mean composite by 1.9 points and increases both FP
and FN. With the same robust temporal idea it remains 0.4 mean points and 1.2
worst-trip points below the no-mask version. It does not recover a T05 danger
true positive under the robust policy.

## 6. Compute cost

Matcher-only timing on the 72-frame sample, same local CPU:

| Compute | P50 | P95 | P99 |
|---|---:|---:|---:|
| Left SGBM | 14.76 ms | 16.42 ms | 16.92 ms |
| Right SGBM | 14.46 ms | 16.98 ms | 17.88 ms |
| Stereo pair | 29.32 ms | 33.63 ms | 33.95 ms |

Explicit right matching adds roughly one more SGBM computation. These are
matcher-only timings and must not be presented as end-to-end latency.

## 7. Decision

Do not use `consistent == true` as a hard input mask for the final ROI median.
Retain these outputs for later components:

- LR residual per pixel;
- consistent/valid coverage per obstacle;
- depth dispersion after confidence weighting;
- degraded state when support is weak;
- soft weights or innovation gates rather than unconditional deletion.

The next experiment is ground-plane/V-disparity removal. Visual evidence shows
that the main remaining problem is not invalid road disparity; it is valid,
consistent road disparity dominating the fixed ROI.
