# Baseline Run 001

## Run identity

- Date: 2026-07-23
- Predictor: unmodified organizer `baseline_ttc_predictor.py`
- Evaluator: unmodified organizer `evaluation.py`
- Dataset: `T01-Sample` through `T06-Sample`
- Python: 3.12.13
- OpenCV: 5.0.0
- NumPy: 2.5.1
- Pandas: 3.0.5
- Execution: sequential, CPU path
- Predictions: `ai_cv/outputs/predictions/baseline_official/` (git-ignored)
- Evaluator JSON: `ai_cv/outputs/reports/baseline_official/evaluation.json` (git-ignored)

The CPU model and memory configuration were not captured because hardware-query
access was unavailable in the current sandbox. Runtime numbers must therefore be
treated as a local observation, not a portable SLA.

## Reproduction

Run once per practice trip:

```powershell
& '.\.venv\Scripts\python.exe' `
  'Package_starterkit\package_starterkit\team_kit\baseline_ttc_predictor.py' `
  --trip-dir 'Practice_Dataset\T01-Sample' `
  --output 'ai_cv\outputs\predictions\baseline_official\T01-Sample.csv'
```

Then evaluate the directory:

```powershell
& '.\.venv\Scripts\python.exe' `
  'Package_starterkit\package_starterkit\team_kit\evaluation.py' `
  --predictions 'ai_cv\outputs\predictions\baseline_official' `
  --data-dir 'Practice_Dataset' `
  --output 'ai_cv\outputs\reports\baseline_official\evaluation.json'
```

## Output validation

All six files passed these checks:

- Exactly 600 rows.
- Frame IDs exactly `0..599`.
- Timestamps exactly `frame_id * 0.05` seconds.
- No NaN or negative TTC values.
- `inf` is retained as the baseline's valid no-collision prediction.

| Trip | Finite predictions | Infinite predictions | Minimum TTC (s) |
|---|---:|---:|---:|
| T01-Sample | 152 | 448 | 0.458 |
| T02-Sample | 150 | 450 | 0.170 |
| T03-Sample | 280 | 320 | 0.089 |
| T04-Sample | 130 | 470 | 0.853 |
| T05-Sample | 251 | 349 | 0.285 |
| T06-Sample | 222 | 378 | 0.287 |

## Official evaluator results

| Trip | Critical frames | MAE critical (s) | inv-TTC MAE | F1 | FPR | Composite |
|---|---:|---:|---:|---:|---:|---:|
| T01-Sample | 12 | 58.595 | 0.0520 | 0.125 | 0.008 | 30.6 |
| T02-Sample | 19 | 6.617 | 0.4031 | 0.211 | 0.158 | 12.2 |
| T03-Sample | 32 | 34.066 | 0.5999 | 0.167 | 0.296 | 5.0 |
| T04-Sample | 76 | 45.670 | 0.0887 | 0.450 | 0.018 | 38.2 |
| T05-Sample | 47 | 53.877 | 0.2329 | 0.000 | 0.150 | 16.0 |
| T06-Sample | 97 | 29.450 | 0.4125 | 0.364 | 0.139 | 16.2 |
| **Overall** | **283** | **38.046** | **0.2982** | **0.220** | — | **19.7** |

The evaluator computes MAE in the `GT TTC < 3 s` zone and classifies danger at
`TTC < 2 s`.

## Starter-kit README discrepancy

The starter-kit README shows an illustrative single-trip result for T01-Sample:
62 critical frames, 1.420 s critical MAE, F1 0.480, and composite 52.3. That
example does not match the supplied dataset/evaluator combination used in this
run. The current T01-Sample has only 12 frames below the evaluator's 3-second
critical threshold and produces composite 30.6.

This is provably more than an OpenCV-version difference because `n_critical` is
computed only from trusted ground truth; the predictor and OpenCV cannot change
it. Treat the README block as stale/illustrative documentation, not a regression
target. Also, its 52.3 is a one-trip score, while 19.7 is the mean composite over
all six current practice trips.

## Wall-clock runtime

| Trip | Seconds | Effective FPS |
|---|---:|---:|
| T01-Sample | 33.480 | 17.92 |
| T02-Sample | 31.481 | 19.06 |
| T03-Sample | 32.118 | 18.68 |
| T04-Sample | 33.476 | 17.92 |
| T05-Sample | 33.792 | 17.76 |
| T06-Sample | 32.644 | 18.38 |
| **Total / aggregate** | **196.991** | **18.28** |

The local run averages approximately 54.72 ms per frame versus a 50 ms frame
budget at 20 FPS. This is an end-to-end wall-time estimate, not a P50/P95 latency
measurement.

## Failure pattern

- T05 has zero danger-detection recall: 0 true positives, 35 false negatives,
  and 85 false positives.
- T03 over-alerts most severely: FPR 0.296 and 169 false positives.
- The baseline returns `inf` on many genuinely critical frames: 106 of 283
  critical frames across the six trips.
- T01 has the worst critical-zone MAE (58.595 s).
- T03 has the worst composite score (5.0/100).

These failures match the algorithm design: a fixed road ROI has no object
identity, mixes background and obstacles, and converts noisy frame-to-frame
stereo depth changes directly into closing speed. The next model must isolate
collision-relevant objects and stabilize motion/depth temporally before TTC.
