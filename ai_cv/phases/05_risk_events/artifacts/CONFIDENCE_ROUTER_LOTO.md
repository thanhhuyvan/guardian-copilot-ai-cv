# Phase 05 confidence router — leave-one-trip-out result

Date: 2026-07-28  
Branch: `research/phase-05-risk-events`

## Purpose

Test whether causal confidence signals can select between the frozen guarded
classical TTC and live detector-owned TTC without using environment identity.

The detector evidence was exported from one live six-trip replay using the
validated YOLO26/SGBM configuration. The classical evidence comes from the
frozen guarded diagnostics. The two sources cover the same 3,600 frames.

## Leakage contract

The router cannot use:

- trip ID;
- frame ID;
- timestamp or position within a trip;
- target/ground-truth values.

Its 12 inputs are detector/classical confidence differences, motion residual,
observation quality, left-right support, track history, detector depth
confidence, proposal counts, ground confidence, ego speed, disagreement size,
and which source currently reports danger.

The router changes a prediction only when the two sources disagree on the
`TTC < 2 s` danger decision. On agreement it retains the classical prediction.

## Protocol

For each of six folds:

1. Fit a class-balanced L2 logistic router on disagreement frames from five
   trips.
2. Apply the fixed model to the untouched sixth trip.
3. Score all 600 held-out frames.

Only 124 disagreement frames exist across the complete dataset. Individual
training folds contain 91–116 examples and validation folds contain 8–33.

```powershell
.\.venv\Scripts\python.exe `
  ai_cv\phases\05_risk_events\src\cross_validate_confidence_router.py
```

## Result

| Policy | Macro F1 | Composite | Critical MAE |
|---|---:|---:|---:|
| Guarded classical | 0.563 | 39.71 | 44.806 s |
| Live detector-owned | 0.632 | 42.83 | 37.891 s |
| Learned LOTO router | **0.488** | 37.32 | 51.820 s |
| Fixed conservative union | **0.658** | **42.88** | **29.993 s** |

Per-trip F1:

| Held-out trip | Classical | Detector | Learned router | Conservative union |
|---|---:|---:|---:|---:|
| T01 | 0.452 | 0.312 | 0.298 | 0.292 |
| T02 | 0.765 | 0.519 | 0.462 | 0.757 |
| T03 | 0.333 | 0.840 | 0.465 | 0.710 |
| T04 | 0.763 | 0.874 | 0.776 | 0.860 |
| T05 | 0.261 | 0.429 | 0.122 | 0.509 |
| T06 | 0.807 | 0.821 | 0.804 | 0.821 |

## Interpretation

The learned router fails the generalization gate. The relationship between
confidence features and which backend is correct changes by environment. With
only 8–33 held-out disagreements per trip, a dynamic CNN or larger selector
would have even more capacity to recognize the practice environment rather
than learn transferable reliability.

The conservative union has no fitted parameters: retain classical TTC, except
when only the detector reports danger. It improves macro recall, F1, composite,
and critical MAE, but T01 false positives rise to 31 and its F1 falls to
`0.292`. It is therefore a safety-oriented candidate for the event state
machine, not a promoted frame-level default.

## Decision

- Reject the learned confidence router.
- Do not introduce a dynamic CNN with the current six-trip dataset.
- Keep guarded classical and detector-owned TTC as separate observable inputs.
- Test conservative union through hysteresis/debounce and report false-event
  duration, not only frame false positives.
- Require external driving environments before training another selector.

The machine-readable fold report is
`artifacts/confidence_router_loto.json`.
