# T05 ground-leakage screening — no repair promoted

## Question

Do T05 V1 false alerts arise mainly because the global V-disparity ground
model leaks road pixels into classical obstacle components?

## Fixed diagnostic protocol

Rendered every V1 T05 false-alert frame (`47`) and every V1 T05 true-danger
anchor (`28`) from the same frozen Phase 14 run. Each overlay contains the
fresh dual-SGBM ground/obstacle masks, extracted components, selected classical
component, and frozen YOLO boxes. Ground-truth TTC only selects offline audit
strata; it is not an inference input.

## Screening result

Road/background support is visibly present in selected components, but it is
not unique to false alerts. Median diagnostics are similar across strata:

| Metric | False alerts | True-danger anchors |
|---|---:|---:|
| Ground confidence | 0.474 | 0.599 |
| Ground residual | 0.500 px | 0.433 px |
| Selected ground fraction | 0.221 | 0.321 |
| Selected obstacle fraction | 0.705 | 0.650 |
| Selected component quality | 0.843 | 0.846 |

Both strata contain broad components blending a real road user with road or
background structure. Therefore a global “low ground confidence,” “large
component,” or “ground fraction” gate cannot be justified: it would remove
some false alerts but likely remove genuine T05 danger as well.

## Decision

Do not implement a ground-removal gate or retune V-disparity values. This
screen does not meet the pre-registered condition that false alerts show a
separable ground-leakage signature absent from true danger.

Next useful upstream test remains object/component association: identify why
the selected broad classical component does not correspond tightly to the
YOLO road-user box. Generated overlays and the optional visual-label CSV are
ignored under `ai_cv/outputs/phase15_ground_leakage_t05/`.
