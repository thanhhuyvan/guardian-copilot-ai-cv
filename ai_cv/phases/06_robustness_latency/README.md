# Phase 06 — Deployment Readiness, Robustness, and Latency

**Status:** IN_PROGRESS  
**Depends on:** Phase 05 (complete)

## Current decision

Deployment work has started. The frozen integrated candidate passed its clean
six-trip readiness gate on the RTX 3060:

- macro danger-F1 `0.658`
- macro composite `42.88`
- macro critical-TTC MAE `29.993 s`
- compute P95 `65.91 ms` against the accepted `75 ms` target
- `0` mismatches across `43,200` repeat comparisons
- all `3,600` perception documents and `16` risk events schema-valid
- peak process RAM `1.42 GB`; peak process VRAM `461 MB`

The stable entry point is `src/run_integrated_deployment.py`. It always uses
live detections, the fixed conservative TTC union, and the recommended event
hysteresis. Cached detections and runtime policy switching are not available.

## Remaining Phase 06 work

The clean-input deployment gate does not prove adverse-condition robustness.
Before Phase 07 submission, run a non-destructive perturbation and failure
matrix covering:

- blur, darkness, noise, and partial occlusion
- missing/corrupt camera frames and calibration failure
- dropped frames and irregular timestamps
- detector/tracker exceptions and empty detections
- output quality degradation and fallback contract behavior

## Reproduce the clean certification

```powershell
.\.venv_yolo26\Scripts\python.exe `
  ai_cv\phases\06_robustness_latency\src\run_integrated_deployment.py
```

See `artifacts/DEPLOYMENT_READINESS.md` for the protocol, results, decision,
known limitations, and exact command.
