# Phase 06 — Robustness and Latency Report

## Decision

**PASS WITH OPERATING LIMITS — Phase 06 is complete.**

The integrated deployment candidate remains suitable for the clean-input
deployment target and fails closed for every injected infrastructure fault.
Medium image noise is the main visual robustness limit and must be surfaced as
reduced-confidence/degraded operation rather than treated as equivalent to a
clean camera feed.

## Clean deployment certification

The separate full certification used all `3,600` frames, `100` warm-up frames,
and five repeats. It remains the latency claim:

| Metric | Result |
|---|---:|
| Macro danger-F1 | `0.6579` |
| Macro composite | `42.8817` |
| Critical-TTC MAE | `29.9929 s` |
| End-to-end compute P95 | `65.9098 ms` |
| Repeat mismatches | `0 / 43,200` |
| Peak process RAM | `1,420.67 MB` |
| Peak process VRAM | `460.68 MB` |

See `DEPLOYMENT_READINESS.md` for the exact certification command.

## Visual robustness screen

This is a **degradation screen**, not a replacement for the full benchmark.
It uses every ground-truth danger frame plus every 32nd safe frame, producing
`387` fixed frames across all six trips. Each condition uses one warm-up pass,
one measured pass, the frozen live YOLO26+SGBM+TTC+event pipeline, and
in-memory changes only. Source images and labels were not modified.

| Condition | Severity | Macro F1 | Δ F1 vs screen clean | Composite | P95 (ms) | VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Clean screen | — | `0.8186` | `0.0000` | `33.13` | `79.57` | `456.68` |
| Gaussian blur | 2 | `0.7816` | `-0.0370` | `31.22` | `75.07` | `468.68` |
| Darkness | 2 | `0.8179` | `-0.0007` | `32.95` | `76.15` | `470.68` |
| Sensor noise | 2 | `0.7053` | `-0.1133` | `29.91` | `78.32` | `470.68` |
| Central occlusion | 2 | `0.8158` | `-0.0028` | `32.99` | `76.41` | `472.68` |

The screen-clean F1 is intentionally higher than the full-set F1 because it
is danger-enriched. Only the deltas within this table are comparable. The
screen P95 values are one-repeat robustness observations, not replacement
latency certification; the five-repeat clean result above remains authoritative.

## Fault and fallback matrix

| Fault | Required output | Verified behavior |
|---|---|---|
| Missing left or right camera | `unknown` | Empty objects, null TTC, `UNKNOWN` risk |
| Corrupt stereo pair or invalid calibration | `unknown` | Same fail-closed contract |
| Detector or tracker exception | `unknown` | Same fail-closed contract; risk state cleared |
| Frame drop | `unknown` | Same fail-closed contract; no stale alert carried forward |
| Irregular timestamp | `unknown` | Same fail-closed contract; state reset |

All seven injected fault documents validate against `perception.v1`. The risk
state-machine recovery check verifies that an unreliable frame enters
`UNKNOWN`, and the next reliable safe frame restarts in `NORMAL` rather than
carrying an earlier alert.

## Operational limits

- Do not claim robustness to high sensor noise: it reduced recall and macro
  F1 materially at medium severity.
- Continue to expose `UNKNOWN` rather than an old TTC whenever a camera,
  detector, tracker, timestamp, or stereo pair is unreliable.
- T01/T05 clean-input false-alert limitations remain accuracy issues from
  Phase 05, not solved by this robustness phase.
- The reproducible full stress command supports severities `1 2 3`; the
  completed release gate uses the defined medium (`2`) operational condition.

## Reproduce

```powershell
.\.venv_yolo26\Scripts\python.exe `
  ai_cv\phases\06_robustness_latency\src\run_robustness_matrix.py `
  --safe-stride 32 --severities 2 `
  --output-dir ai_cv\outputs\phase06_robustness_medium

.\.venv_yolo26\Scripts\python.exe `
  ai_cv\phases\06_robustness_latency\src\verify_fallback_contracts.py
```
