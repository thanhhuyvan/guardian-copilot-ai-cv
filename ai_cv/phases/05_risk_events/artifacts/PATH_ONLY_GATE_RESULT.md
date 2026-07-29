# Path-only event-gate diagnostic

## Purpose

Test the direct ego-path geometry alone over the frozen Phase 17
conservative-union predictions. The raw TTC stream is retained unchanged; the
diagnostic only emits a finite `2.0 s` event encoding when an associated YOLO
box-depth measurement is outside a fixed 1.75 m physical corridor.

## Reproduction

```powershell
D:\Python\.venv\Scripts\python.exe ai_cv\phases\05_risk_events\src\evaluate_path_only_gate.py `
  --evidence-root ai_cv\outputs\phase17_v2_framewise_corrected\evidence `
  --practice-root Practice_Dataset `
  --output-dir ai_cv\outputs\phase24_path_only_gate
```

## Six-trip result

| Policy | F1 | Critical TTC MAE | Composite |
|---|---:|---:|---:|
| Frozen Phase 17 raw reproduction | 0.6543 | 29.993 s | 42.773 |
| Direct path-only event diagnostic | 0.6564 | 29.993 s | 42.846 |

The raw reproduction agrees with the frozen V1 report after the challenge's
mean-per-trip aggregation. The gate suppresses two false-danger frames, both
in T01: T01 F1 rises from 0.2917 to 0.3043. T02 and T05 do not change. No
critical TTC MAE changes because neither suppressed frame is critical.

## Interpretation

This passes the pre-registered *diagnostic* criteria, but it is deliberately
small: only two of 86 geometry-eligible frames were outside the corridor and
eligible for suppression. It confirms that direct path relation is a safe,
targeted cue for this frozen run; it does **not** solve T05 or justify a V2
promotion. The visual labels were AI-provisional, not independent, so this
result remains research evidence only.
