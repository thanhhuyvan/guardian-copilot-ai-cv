# Original-dataset classical baseline rerun

Date: 2026-07-28  
Branch: `research/phase-05-risk-events`  
Git revision used for inference: `3dd82f0347f35b29a4a3493cae8b93a29699a385`

## Purpose

Verify from the original practice JPEGs and trip metadata—not cached candidate
traces—whether the classical guarded baseline achieves macro danger-F1 above
`0.50`.

## Protocol

- Dataset: `Practice_Dataset/T01-Sample` through `T06-Sample`
- Frames: 600 per trip, 3,600 total
- Dataset fingerprint:
  `8310e4eeadcb8518970624913861245fc38072321821809f8674f6a68871d3ad`
- Backend: SGBM FP32, full native `640x360`
- TTC policy: `guarded`
- Warm-up: 100 frames
- Measured repeats: one
- Evaluation: original starter-kit evaluator

```powershell
.\.venv\Scripts\python.exe `
  ai_cv\phases\02_detection_tracking\src\benchmark_stereo_latency.py `
  --backend sgbm `
  --precision fp32 `
  --repeats 1 `
  --warmup-frames 100 `
  --practice-root Practice_Dataset `
  --starter-root Package_starterkit\package_starterkit `
  --output-root ai_cv\outputs\benchmarks\phase05_original_baseline_rerun `
  --ttc-policy guarded `
  --latency-target-ms 75 `
  --skip-evaluation `
  --progress-every 600
```

The runner's `--skip-evaluation` flag permits a one-repeat metric check. The
newly written six prediction CSVs were subsequently validated and passed to
`team_kit.evaluation.evaluate` against `Practice_Dataset`.

## Result

| Metric | Fresh result | Check |
|---|---:|---|
| Macro danger-F1 | **0.564** | **PASS > 0.50** |
| Composite | 39.7 | Matches frozen result |
| Critical-TTC MAE | 44.806 s | Matches frozen result |
| Inverse-TTC MAE | 0.1198 | Matches frozen result |
| Compute P50 | 60.46 ms | Informational |
| Compute P95 | 73.13 ms | PASS `<75 ms` in this one run |
| Compute P99 | 82.03 ms | Informational |

Per-trip evaluator results:

| Trip | F1 | Precision | Recall | Composite |
|---|---:|---:|---:|---:|
| T01 | 0.452 | 0.333 | 0.700 | 41.2 |
| T02 | 0.765 | 0.812 | 0.722 | 46.3 |
| T03 | 0.333 | 0.421 | 0.276 | 30.7 |
| T04 | 0.763 | 0.822 | 0.712 | 50.1 |
| T05 | 0.261 | 0.211 | 0.343 | 31.0 |
| T06 | 0.807 | 0.852 | 0.767 | 39.0 |

All six prediction files passed the source-frame validator with exactly 600
rows. Compared with the frozen guarded predictions, finite/`inf` decisions and
danger classifications were identical. Numeric finite TTC differences were
only output precision, with maximum absolute difference at most `0.0005 s`.

## Interpretation

The baseline independently reproduces macro F1 above `0.50`; the answer is
**yes, 0.564**. This does not mean every fold exceeds `0.50`: T01, T03, and T05
are below it. The macro score is carried by strong T02, T04, and T06 results,
so Phase 05 must continue to report per-trip and LOTO behavior.

The latency values are a one-repeat check, not the official five-repeat
deployment certification.
