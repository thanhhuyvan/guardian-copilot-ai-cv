# Phase 08: Confidence-Aware Temporal Stereo Evaluation

## Decision

Do not promote either Phase 08 candidate.  The confidence-only gate is
behavior-preserving in the matched robustness screen, but it makes no measured
improvement.  The temporal-motion variant loses true danger events in the T03
smoke window.

The frozen Phase 06 baseline remains the deployment candidate.  The next
accuracy experiment is the separately trained LightStereo domain-adaptation
lane after the WSL CUDA environment is ready.

## Scope and reproducibility

The implementation exposes two opt-in policies; both default to disabled:

- `--depth-confidence-gate --minimum-depth-confidence 0.45`: reject a TTC
  measurement when its left/right stereo-consistency support is insufficient.
- `--confidence-temporal`: additionally enable the existing uncertainty-aware
  temporal motion filter.  Predicted tracks remain excluded from braking.

Stereo left/right support is incorporated into per-object depth confidence.
The TTC selector can then reject low-confidence depth measurements without
changing the conservative-union fallback.  Tracking state is reset per trip by
the existing evaluator contract.

## Results

### T03 targeted smoke window

Protocol: 120 frames beginning at index 280 of `T03-Sample`, two repeats,
20 warm-up frames.  Metrics below use detector-owned danger decisions.

| Candidate | Danger F1 | FN | P95 compute (ms) | Decision |
|---|---:|---:|---:|---|
| Baseline | 0.8400 | 8 | 71.91 | Reference |
| Confidence + temporal, threshold 0.45 | 0.7660 | 11 | 69.05 | Reject: loses 3 true positives |
| Confidence gate only, threshold 0.45 | 0.8400 | 8 | 0.79* | Preserve only |

`*` This short CPU smoke timing is not a latency gate and is affected by host
load; it must not be compared to the Phase 06 full latency benchmark.

### Matched six-trip robustness screen

Protocol: all danger frames plus every 32nd safe frame, six trips, one repeat,
severity-2 synthetic noise.  This is the same screen used for the Phase 06
noise comparison.  Candidate: confidence gate only, threshold 0.45.

| Condition | Composite | Danger F1 | Precision | Recall | Critical TTC MAE |
|---|---:|---:|---:|---:|---:|
| Clean | 33.1318 | 0.8186 | 0.8980 | 0.7626 | 31.3525 |
| Noise S2 | 29.9100 | 0.7053 | 0.8945 | 0.6088 | 40.1763 |
| Noise delta | -3.2219 | -0.1133 | -0.0034 | -0.1539 | +8.8238 |

These values exactly match the prior Phase 06 screen to displayed precision.
Therefore the confidence-only gate is safe in this benchmark, but does not
repair the observed noise sensitivity.  It does not justify a full LOTO or
latency promotion run.

Generated evidence is intentionally ignored and can be regenerated with:

```powershell
.\.venv_yolo26\Scripts\python.exe ai_cv\phases\06_robustness_latency\src\run_robustness_matrix.py `
  --safe-stride 32 --severities 2 --perturbations noise `
  --depth-confidence-gate --minimum-depth-confidence 0.45 `
  --output-dir ai_cv\outputs\phase08_confidence_gate_noise
```

The summary is written to
`ai_cv/outputs/phase08_confidence_gate_noise/robustness_screening.json`.
