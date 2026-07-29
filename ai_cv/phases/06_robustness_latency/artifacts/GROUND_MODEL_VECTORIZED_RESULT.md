# Vectorized V-disparity line-fit — promoted pending full latency recertification

## Fixed experiment

The existing V-disparity estimator was split into histogram, row-mode, and
robust-line-fit work. A vectorized candidate enumerates the same deterministic
line pairs, selects the same maximum score, and retains the original weighted
three-pass refinement. It is a behavior-preserving implementation change, not
a new ground model or accuracy gate.

## Shadow parity

All six trips were processed with the frozen `2 OpenCV threads / 2 SGBM
matchers` configuration. Each frame compared the complete `GroundModel`
dataclass from the loop and vectorized fit before downstream masking.

| Trip | Frames checked | Ground-model mismatches | Reference fit P95 | Vectorized fit P95 |
|---|---:|---:|---:|---:|
| T01 | 600 | 0 | 20.97 ms | 4.63 ms |
| T02 | 600 | 0 | 17.00 ms | 3.90 ms |
| T03 | 600 | 0 | 17.59 ms | 4.50 ms |
| T04 | 600 | 0 | 22.01 ms | 6.46 ms |
| T05 | 600 | 0 | 22.96 ms | 7.12 ms |
| T06 | 600 | 0 | 16.02 ms | 5.00 ms |
| **Total** | **3,600** | **0** | — | — |

The profile includes an extra shadow fit per frame, so its total ground-stage
timing is not a replacement deployment latency claim. The precise conclusion
is limited to identical ground models and a substantially lower isolated
line-fit cost.

## Decision

The vectorized fit now replaces the loop implementation in
`estimate_ground_model()`. It cannot change masks, components, TTC, or risk
when parity holds. A clean all-trip/five-repeat decoded-pair-to-risk benchmark
is still required before updating the authoritative `65.91 ms` deployment
claim.
