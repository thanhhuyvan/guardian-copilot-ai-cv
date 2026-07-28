# Phase 06 — Deployment Readiness, Robustness, and Latency

**Status:** COMPLETE  
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

## Completed robustness gate

Phase 06 now includes a non-destructive medium-severity perturbation and
failure matrix covering:

- blur, darkness, sensor noise, and partial occlusion
- missing/corrupt camera frames and injected model/tracker faults
- dropped frames and irregular timestamps
- output degradation and fail-closed contract behavior

Noise is the primary measured robustness limitation. See
`artifacts/ROBUSTNESS_LATENCY_REPORT.md` for the precise operating limits.

## Reproduce the clean certification

```powershell
.\.venv_yolo26\Scripts\python.exe `
  ai_cv\phases\06_robustness_latency\src\run_integrated_deployment.py
```

See the deployment-readiness and robustness reports for the full protocol,
results, operating limits, and exact commands.
