# Lightweight Baseline Improvements 001

## Goal

Test small causal changes around the organizer SGBM baseline before introducing
object detection or learned depth. Ground truth is used only by the evaluator.

## Variants

| Variant | Depth feature | History | Slope | Physical gate |
|---|---|---:|---|---|
| official_replay | Median in original ROI | 5 | OLS | closing >0.3 m/s |
| robust_median | Median in original ROI | 11 | Theil-Sen | 0.3-20 m/s |
| robust_near | 35th percentile in original ROI | 11 | Theil-Sen | 0.3-20 m/s |
| robust_corridor | 35th percentile in wider/upper ROI | 11 | Theil-Sen | 0.3-20 m/s |

Original ROI: x=35-65%, y=50-85%. The corridor variant uses x=30-70%,
y=42-75%, which includes more cut-in area and excludes the closest road strip.

`official_replay` reproduces the original finite/inf mask exactly and finite TTC
values to 0.001 s, validating the shared-disparity experiment runner.

## Official evaluator results

| Variant | MAE critical | inv-TTC MAE | F1 | Mean composite | Worst trip |
|---|---:|---:|---:|---:|---:|
| official_replay | 38.046 | 0.2982 | 0.220 | 19.7 | 5.0 |
| robust_median | 43.114 | 0.1896 | 0.258 | 32.2 | 16.9 |
| robust_near | 63.515 | 0.1587 | 0.270 | 28.5 | 14.8 |
| **robust_corridor** | **34.644** | **0.1922** | **0.280** | **32.7** | **22.1** |

Relative to the official baseline, `robust_corridor`:

- Improves mean composite by 13.0 points (19.7 -> 32.7).
- Improves worst-trip composite by 17.1 points (5.0 -> 22.1).
- Reduces inv-TTC MAE by 35.5%.
- Improves F1 from 0.220 to 0.280.
- Reduces critical MAE by 8.9%, though it remains unacceptably high.

## Per-trip composite

| Trip | Official | Robust corridor | Delta |
|---|---:|---:|---:|
| T01-Sample | 30.6 | 33.8 | +3.2 |
| T02-Sample | 12.2 | 57.2 | +45.0 |
| T03-Sample | 5.0 | 22.1 | +17.1 |
| T04-Sample | 38.2 | 36.5 | -1.7 |
| T05-Sample | 16.0 | 22.2 | +6.2 |
| T06-Sample | 16.2 | 24.5 | +8.3 |

The improvement is not uniform. T04 regresses slightly and the large T02 gain
must not be treated as evidence of generalization.

## Danger confusion counts

| Variant | TP | FP | FN |
|---|---:|---:|---:|
| official_replay | 80 | 436 | 124 |
| robust_median | 81 | 240 | 123 |
| robust_near | 62 | 130 | 142 |
| robust_corridor | 83 | 254 | 121 |

The robust temporal estimator provides most of the gain: `robust_median` already
reaches 32.2. Changing the ROI/statistic adds only 0.5 mean points, but improves
the worst trip from 16.9 to 22.1. The 35th percentile alone is not safe: it cuts
false positives but increases misses.

## Review recommendation

Review the changes as two separable hypotheses:

1. **Candidate core change:** 11-frame Theil-Sen slope plus a 20 m/s physical
   closing-speed gate. It leaves the original spatial feature unchanged and
   yields most of the score improvement.
2. **Experimental spatial change:** wider/upper corridor with a 35th percentile.
   Keep behind configuration until cross-validation/failure review confirms it.

Do not replace the project baseline regression yet. The variants were selected
and evaluated on the same six practice trips, T05 still has zero true-positive
danger frames, and no scored-trip labels exist to validate generalization.

## Reproduction

```powershell
python ai_cv/phases/01_data_baseline/src/experiment_lightweight_improvements.py
```

To regenerate reports/charts without recomputing SGBM:

```powershell
python ai_cv/phases/01_data_baseline/src/experiment_lightweight_improvements.py `
  --reuse-predictions
```

Generated predictions, evaluator JSON, and charts are under
`ai_cv/outputs/benchmarks/lightweight_baseline/` and are git-ignored.
