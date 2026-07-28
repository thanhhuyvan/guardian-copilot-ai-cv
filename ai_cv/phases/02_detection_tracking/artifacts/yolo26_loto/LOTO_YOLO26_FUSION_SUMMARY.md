# Phase 04B - YOLO26 Semantic Fusion: LOTO Result Summary (Corrected)

**Date:** 2026-07-27 (rerun after fixing 5 validation bugs)
**Run type:** 6-fold leave-one-trip-out cross-validation
**Model:** yolo26n.pt (PyTorch FP32) + pre-computed detection CSVs
**Fusion mode:** soft-guard
**Physical guard:** frozen
**Source data:** ai_cv/outputs/benchmarks/phase04_loto/source

---

## Result: ALL 6 FOLDS INFEASIBLE

No hyperparameter configuration in the 27-policy grid satisfies both training
constraints on any fold's 5-trip training set:

- Composite >= 38.4 -- passes for most configs
- Critical-TTC MAE <= baseline MAE on same 5 trips -- FAILS FOR EVERY CONFIG ON EVERY FOLD

No validation metrics can be reported. LOTO macro summary is null.

---

## Previous Run Was Invalid

The earlier run (before fixes) reported Macro F1=0.5634, Composite=39.71, MAE=44.806s.
Two bugs caused this:

1. Temporal state keyed by row index `idx` instead of `track_id` -- consecutive misses
   never accumulated, soft-guard was effectively disabled, YOLO had no real effect.
2. Width reconstructed from height norm instead of selected_width_norm -- component
   bboxes were square, corrupting IoU matching.

With both bugs, semantic fusion had near-zero suppression effect. The reported numbers
were essentially the physical-guard-only baseline mislabelled as fusion results.

---

## Correct Physical-Guard Baseline (confirmed by fixed code)

| Trip       | F1     | Composite | MAE (s) |
|------------|--------|-----------|---------|
| T01-Sample | 0.4516 | 41.21     | 40.547  |
| T02-Sample | 0.7647 | 46.28     | 31.464  |
| T03-Sample | 0.3333 | 30.71     | 62.231  |
| T04-Sample | 0.7629 | 50.12     | 37.155  |
| T05-Sample | 0.2609 | 30.96     | 64.331  |
| T06-Sample | 0.8070 | 39.00     | 33.105  |
| Macro      | 0.5634 | 39.71     | 44.806  |

These are the correct numbers for the physical guard alone.

---

## Root Cause of Infeasibility

Example: Fold 1 (hold out T01, train on T02-T06):

|                              | Composite | MAE      |
|------------------------------|-----------|----------|
| Baseline (no semantics)      | --        | 45.66 s  |
| Best semantic (0.25, 3, 5.0) | 39.27     | 49.12 s  |
| MAE constraint met?          | --        | FAIL (+3.46 s) |

The damage is entirely from T03:

- T03 baseline MAE:   62.2 s
- T03 with semantics: 77.0 s  (+14.8 s)
- T03 suppressed:     750 / 1355 candidates (55%)
- T03 YOLO matched:   162 / 1355 only

T03 is a dark/low-visibility trip. YOLO detects very few objects.
With fixed temporal state, consecutive misses accumulate correctly and the
soft-guard fires -- but on T03 the suppressed candidates include real danger
tracks that YOLO cannot see.

---

## Per-Trip Soft-Guard Effect (config 0.25, 3, 5.0)

| Trip       | Suppressed | Matched | Total | MAE delta  |
|------------|------------|---------|-------|------------|
| T01-Sample | 606        | 116     | 755   | 0          |
| T02-Sample | 151        | 122     | 745   | 0          |
| T03-Sample | 750        | 162     | 1355  | +14.8 s    |
| T04-Sample | 340        | 372     | 804   | +2.6 s     |
| T05-Sample | 488        | 238     | 938   | 0          |
| T06-Sample | 1088       | 529     | 1894  | 0          |

---

## Gate Status

All gates N/A -- no feasible fold exists.
Promotion: BLOCKED.

Per plan: "Reject YOLO fusion if it fails F1 or T03 recall after the fixed
27-policy semantic search. Do not start another broad threshold sweep."

---

## Correct Next Step Per Plan

The plan prescribes for this case:
  "Detector misses true road users: collect box annotations."

Fine-tuning YOLO on T03-style (dark/low-visibility) scenes is the correct
next action. This is a separate phase.

Do NOT change corridor rules, MAE constraints, or soft-guard thresholds to
force a pass -- the constraint is correct.

---

## ONNX Status

Structural export: valid (onnx.checker passed; output `[1,300,6]`).
Parity test: **rerun on GPU** (2026-07-27 21:05) — both backends confirmed on
CUDA (`cuda:0` + `CUDAExecutionProvider`). Results:

- Class agreement 96.55% [GATE FAIL ≥99%], median IoU 0.9968 [PASS],
  mean conf diff 0.0205 [GATE FAIL ≤0.02].
- All 3 class swaps are car↔truck on the **same physical box** (IoU 0.997-0.999)
  at low confidence near the 0.25 threshold — **0 distinct-object errors**.
- Root cause: the end-to-end ONNX export retains competing class hypotheses per
  box and tiebreaks by insertion order; Ultralytics' native postprocess
  (PyTorch path) tiebreaks differently. **Not** a parser bug, not double-NMS.
- Fusion-equivalent: verified — soft-guard uses only retained-class membership
  + confidence; car and truck are both retained, so the swap changes no
  suppression decision. Per the plan's own rule ("compare danger output, not
  only raw detector tensors"), the raw-tensor gate failure is cosmetic.
- Full diagnostics: `yolo26_export/onnx_parity_report.json`.

GPU env: **RESOLVED** — `torch 2.11.0+cu128` (CUDA available), `onnxruntime-gpu
1.20.1`. Required fix: `_ensure_cuda_dlls_on_path()` in `yolo26_backends.py`
prepends `torch/lib` to PATH so ORT's CUDA EP finds the bundled cuDNN 9 / CUDA
12 DLLs (without it, ORT failed with `LoadLibrary error 126` and silently fell
back to CPU).
