# Phase 04B final decision — YOLO26 semantic fusion

Date: 2026-07-28  
Branch: `research/phase-04b-yolo26-fusion`  
Decision: **REJECT for promotion; retain the classical guarded pipeline**

## Diagnosis

The manual review covered all 23 critical T03 annotation rows:

| Label | Count | Share |
|---|---:|---:|
| Association failure | 16 | 69.6% |
| Stereo noise | 7 | 30.4% |
| Genuine YOLO miss | 0 | 0.0% |
| Unsure | 0 | 0.0% |

The dominant failure was component-to-detection association. In the affected
frames, a large merged stereo component contained the correctly detected lead
car. The frozen association rule rejected it because the component center was
outside the YOLO box, despite the YOLO box being fully contained by the
component.

## Targeted correction

Association now additionally accepts symmetric containment when:

- the detection center is inside the stereo component;
- the stereo component covers at least 50% of the detection box; and
- vertical overlap is at least 50%.

The original IoU and component-center rules remain unchanged. Three regression
tests cover containment, valid merged-component matching, and low-coverage
edge-contact rejection.

## Measured result

The fixed 27-policy sweep was rerun over all six 600-frame practice trips.

| Candidate | Macro F1 | Composite | Critical-TTC MAE | T05 FP | T03 recall |
|---|---:|---:|---:|---:|---:|
| Physical guard baseline | 0.5634 | 39.712 | 44.806 s | 45 | 0.276 |
| Global semantic best | **0.5745** | **40.234** | **44.806 s** | 45 | 0.276 |
| LOTO selected (5 feasible folds) | 0.5286 | 37.950 | 46.953 s | 45 | 0.241 |
| Per-trip oracle upper bound | **0.5745** | 40.181 | 44.806 s | 45 | 0.276 |

The correction is useful: the global semantic candidate gains 0.0111 absolute
F1 and removes eight false positives without worsening critical-TTC MAE.
However, it cannot pass the frozen promotion gates.

## Gate decision

| Hard gate | Requirement | Result | Status |
|---|---:|---:|---|
| LOTO macro danger-F1 | >= 0.60 | invalid full LOTO; 0.5286 over 5 feasible folds | FAIL |
| All LOTO folds feasible | 6 / 6 | 5 / 6 | FAIL |
| T05 false positives | <= 20 | 45 | FAIL |
| T03 recall | >= 0.276 | 0.241 in LOTO | FAIL |
| Composite | >= 38.4 | 37.950 in partial LOTO | FAIL |
| Critical-TTC MAE | <= 46.638 s | 46.953 s in partial LOTO | FAIL |

Even the test-label-selected oracle reaches only 0.5745 F1, below the 0.60
target. More threshold tuning cannot satisfy the gate with this pretrained
detector and fusion policy.

## Closure

- Do not promote YOLO26 semantic fusion.
- Do not begin TensorRT, INT8, or the official five-repeat latency benchmark
  for this rejected candidate; accuracy is an earlier hard stop.
- Do not fine-tune based on this sample: no reviewed critical row was a genuine
  detector miss.
- Keep the association correction as research code and preserve the measured
  artifacts for reproducibility.
- Continue with the classical guarded pipeline in Phase 05 confidence, risk
  state, and event handling.

The unresolved Ultralytics license is no longer a release blocker because no
YOLO26 dependency is being promoted.
