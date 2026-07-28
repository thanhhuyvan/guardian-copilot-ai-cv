# Integrated Deployment Readiness

## Decision

**PASS — start Phase 06 deployment and robustness work.**

The live integrated pipeline preserves the selected accuracy result and passes
the relaxed `75 ms` compute-latency gate. Further unconstrained F1 tuning is
not required before deployment engineering. The candidate is not yet ready
for a final release because degraded-input and failure-mode tests remain.

## Frozen candidate

- stereo: native `640×360` dual SGBM
- detector: YOLO26n PyTorch/CUDA, confidence `0.25`
- execution: stereo and detector concurrent
- TTC routing: classical fallback; detector output overrides only when the
  detector independently reports TTC below `2 s`
- risk output: recommended deterministic event hysteresis
- OpenCV threads: `2`
- stereo workers: `2`
- model SHA-256:
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

## Protocol

- all six practice trips (`3,600` unique frames)
- `100` warm-up frames
- five complete repeats (`18,000` executions)
- disk image loading excluded from the compute gate
- repeat one used for accuracy and emitted contracts
- repeats two through five compared against repeat one for determinism
- original dataset and labels were read-only

## Results

| Gate | Result | Status |
|---|---:|:---:|
| Macro danger-F1 ≥ `0.60` | `0.6579` | PASS |
| Macro composite ≥ `38.4` | `42.8817` | PASS |
| Macro critical-TTC MAE ≤ `46.638 s` | `29.9929 s` | PASS |
| Compute P95 ≤ `75 ms` | `65.9098 ms` | PASS |
| Repeat mismatches | `0 / 43,200` | PASS |
| Perception schema | `3,600 / 3,600` | PASS |
| Risk-event schema | `16 / 16` | PASS |
| Peak process VRAM ≤ `5 GB` | `460.68 MB` | PASS |
| Peak process RAM | `1,420.67 MB` | REPORTED |

Latency distribution:

| Stage | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| Integrated compute | `55.913` | `65.910` | `71.915` |
| Concurrent inference wall | `39.238` | `47.704` | `52.958` |
| Stereo | `38.650` | `47.030` | `52.223` |
| Detector inference | `24.292` | `30.179` | `33.826` |
| TTC/event postprocess | `16.537` | `20.110` | `22.118` |

Accuracy by trip:

| Trip | F1 | Composite | Critical MAE (s) | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| T01 | `0.292` | `35.43` | `40.55` | 7 | 31 | 3 |
| T02 | `0.757` | `46.08` | `21.32` | 14 | 5 | 4 |
| T03 | `0.710` | `43.52` | `27.87` | 22 | 11 | 7 |
| T04 | `0.860` | `53.46` | `26.93` | 46 | 9 | 6 |
| T05 | `0.509` | `39.32` | `31.22` | 28 | 47 | 7 |
| T06 | `0.821` | `39.49` | `32.08` | 48 | 9 | 12 |

## Known limitation

T01 and T05 still contain sustained false-positive alerts. Event hysteresis
removes fragmentation and short flicker but cannot safely remove a long,
consistently wrong TTC estimate. This is explicitly carried into robustness
testing; it is not hidden by per-trip tuning.

## Exact command

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
  --integrated-union-events `
  --output-dir ai_cv\outputs\phase06_integrated_official `
  --progress-every 300
```

The detailed generated report is
`ai_cv/outputs/phase06_integrated_official/detector_owned_report.json` and is
ignored by Git because it is reproducible and accompanied by large frame-level
outputs.
