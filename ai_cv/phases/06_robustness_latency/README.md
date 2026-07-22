# Phase 06 - Robustness and Latency

**Status:** NOT_STARTED  
**Depends on:** Phase 05

## Mục tiêu

Chứng minh pipeline ổn định trước input degraded và đo performance thực tế.

## Câu hỏi nghiên cứu

- Pipeline suy giảm thế nào với blur, dark, noise và frame drop?
- Stereo mất một camera thì fallback nào hợp lệ?
- Bottleneck latency nằm ở detector, depth hay serialization?

## Test cần có

- Missing left/right/both camera.
- Corrupt image và empty detection.
- Blur, brightness, noise, occlusion.
- Frame drop và irregular timestamp.
- Model exception không dừng trip.
- Determinism/reproducibility.

## Verification

- Robustness matrix theo perturbation và severity.
- Mean/worst-trip metric.
- P50/P95 latency, FPS, RAM/VRAM và hardware description.
- Output degraded/unknown đúng contract.

## Exit criteria

- Không crash trong test suite.
- Không công bố real-time nếu chưa đạt benchmark tương ứng.

