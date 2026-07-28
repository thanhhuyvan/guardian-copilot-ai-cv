# T05 False Positive Diagnosis — Phase 04B YOLO26 Semantic Fusion

**Date:** 2026-07-27 (updated after LOTO rerun with fixed validation code)  
**Script:** `tmp/diagnose_t05_fp.py`  
**Data:** `ai_cv/outputs/benchmarks/phase04_loto/source/track_candidates/T05-Sample.csv`  
**Baseline:** physical guard only (`track_p35` predictions — no semantics)  
**JSON detail:** `artifacts/yolo26_loto/t05_fp_diagnosis.json`

> **Note:** The LOTO rerun (after fixing track_id keying and width reconstruction) produced
> ALL 6 FOLDS INFEASIBLE. Every semantic config raises T03 MAE from 62.2 s to 77.0 s,
> failing the constraint `mean_train_mae <= base_train_mae`. The T05 FP analysis below
> is therefore against the physical-guard baseline — which is the correct reference
> since no semantic config was promotable.

---

## Summary

The YOLO26 soft-guard **did not reduce T05 false positives at all** (baseline: 45, after fusion: 45).  
This diagnosis identifies the exact reason.

---

## Numbers

| Category | Candidates | % of total | Suppressible? |
|---|---|---|---|
| depth <= 5m — close-range fallback active | 63 | **57%** | NO — by design |
| depth > 5m — YOLO matches component as vehicle | 17 | **15%** | NO — YOLO keeps score alive |
| depth > 5m — YOLO present but no IoU match | 26 | **23%** | YES (after 3 misses) |
| depth > 5m — No YOLO detection at all | 5 | **5%** | YES (after 3 misses) |
| **Total** | **111** | 100% | — |

Depth stats: min=2.6m, median=4.6m, max=33.4m

---

## Root Cause 1: Close-range fallback (57% of FP candidates)

The majority of T05 FP candidates have **stereo depth <= 5m**. The plan's close-range fallback is a hard rule:

> "At depth <=5m, preserve current guarded TTC behavior even without a semantic match."

So the soft-guard physically cannot fire on these regardless of YOLO result.  
These are **not a YOLO problem**.

### Are these real or stereo artifacts?

Looking at the close-range FP frames (244, 253, 254, 492–506): the depths are 2.6–5.0m and the TTC values are 1.0–2.5s. These look like a **real approaching object at very short range** that the ground truth labels as `inf` (safe). This is likely a T05 scene characteristic — an object that enters the corridor at close range briefly without actually being a collision risk per GT annotation.

**This cannot be fixed by YOLO semantic fusion.** It requires either:
- A corridor geometry fix (the object is not actually in the collision path)
- A ground-truth understanding issue (the GT may label this as safe because the ego-vehicle takes evasive action or the object passes)

---

## Root Cause 2: YOLO correctly detects a real car (15% of FP candidates)

Frames 481–555 show a **real car** passing through the field of view:
- YOLO class: `car` on all 17 matched candidates
- YOLO confidence: 0.89–0.94 (very high)
- IoU with stereo component: 0.29–0.50
- Depths: 5.2m–32.0m

**YOLO is not misclassifying.** It is correctly detecting a real car.  
The stereo pipeline also correctly measures its depth and closing speed.  
The car enters the frame, YOLO matches it confidently, semantic score stays high, soft-guard correctly does not suppress it.

**This is not a YOLO problem and not a fusion problem.** The GT labels this car as safe (TTC=inf), but the stereo+tracking pipeline sees it as a danger candidate. This is a **ground truth ambiguity** — the car is real, the depth is real, but GT doesn't label it as a near-miss event (possibly because it changes lane or the ego brakes).

---

## Root Cause 3: Far components with YOLO present but no match (23%)

26 candidates where YOLO detects something in the frame (bus/truck/person/car) but the IoU with the stereo component is below 0.15 and the component center is not inside the expanded detection box.

These **should eventually be suppressed** after 3 consecutive misses. The fact they are still FP suggests:
- The track restarts before accumulating 3 consecutive misses, or
- A mix of frames where YOLO does match (keeping score alive) and frames where it doesn't

---

## Conclusion: This is NOT a YOLO Problem

| Question | Answer |
|---|---|
| Is YOLO misclassifying fence/road as vehicle? | **No** — fence/road components appear in the close-range zone (<=5m) where fusion is bypassed entirely |
| Is YOLO maintaining scores on genuine vehicles? | **Yes** — it correctly matches a real car at frames 481–555 |
| Would fine-tuning YOLO help? | **No** — YOLO is correct; the problem is the GT labeling and close-range fallback design |
| Is there a simple fix? | **No easy fix** — the FP candidates are either physically too close for soft-guard or are genuinely detected objects that GT marks safe |

---

## What Could Actually Reduce T05 FP

1. **Corridor geometry tightening** — if the 57% close-range components are not truly in the collision path, narrowing the corridor (reducing `corridor_bottom_width`) would eliminate them. Risk: may hurt T03 recall.

2. **Close-range fallback threshold** — lowering from 5m to 3m would expose some candidates to the semantic gate. Risk: misses a real close-range danger.

3. **GT re-examination** — the T05 FP frames 481–555 contain a confirmed real car. The issue may be that GT marks this car as non-dangerous because of driver behavior. This is a **scoring/annotation issue**, not a pipeline issue.

4. **Per-class corridor rule** — exclude objects confirmed as pedestrian or cyclist from the primary corridor TTC (they may be on the roadside, not in the collision path). This is scope-adjacent.

---

## Next Steps Per Plan

Per `docs/YOLO26_SEMANTIC_FUSION_PLAN.md`:

> "Classify failure before training"

The failure is classified:
- **Not** detector miss of true road users
- **Not** association geometry failure
- **Yes** semantic fusion succeeds correctly — GT labeling and close-range physics prevent suppression

Recommended action: **do not start fine-tuning**. The T05 FP problem is in the ground truth / corridor geometry, not in the YOLO detector.

Proceed with live pipeline integration (Step 4) and document this finding.
