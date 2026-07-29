# Guardian latency board — measured work, critical path, and decisions

## Authoritative deployed V1 measurement

The deployment claim is the Phase 06 clean certification: native `640×360`,
batch 1, 100 warm-up frames, all 3,600 frames, five repeats, decoded stereo
pair through TTC/risk output. Disk loading is excluded.

| Gate | Measured result | Limit | Status |
|---|---:|---:|---|
| Compute P50 / P95 / P99 | 55.913 / **65.910** / 71.915 ms | P95 ≤ 75 ms | Pass |
| Mean FPS | about 18 FPS from P50 | — | Informational |
| Process RAM | 1,420.67 MB | — | Informational |
| GPU VRAM | 460.68 MB | ≤ 5,120 MB | Pass |
| Repeat mismatch | 0 / 43,200 | 0 | Pass |
| P95 headroom | **9.09 ms** | 75 ms | Limited |

This is the only latency number to use for V1 deployment claims. Desktop GPU
load, browser activity, thermal state, and a different OpenCV thread setting
can move a later one-pass observation; they do not invalidate this controlled
certification.

## Live pipeline stage board

The Phase 05 live YOLO26+SGBM benchmark used the same native resolution and
the frozen `2 OpenCV threads / 2 SGBM matchers` setting. Its stage timing is a
useful breakdown, but stages **must not be summed**: YOLO runs concurrently
with CPU SGBM.

| Work | Hardware | P50 | P95 | P99 | Critical-path role |
|---|---|---:|---:|---:|---|
| Full decoded-pair → risk pipeline | CPU + GPU | 53.36 | **63.22** | 70.30 | End-to-end observation |
| SGBM stereo (left/right + consistency) | CPU | 38.73 | **47.40** | 53.64 | Largest critical-path block |
| YOLO26n inference | GPU | 23.73 | 30.36 | 33.98 | Mostly overlaps SGBM |
| Concurrent inference wall | CPU + GPU | 39.29 | 48.01 | 54.29 | Stereo generally dominates overlap |
| Depth, tracking, TTC, risk postprocess | CPU | 14.03 | **17.24** | 19.29 | Second critical-path block |

`47.40 + 17.24 ≈ 64.64 ms`, which explains the observed full-pipeline P95.
Do **not** add YOLO's `30.36 ms` again: it overlaps the CPU stereo work.

## Classical-only work board

The earlier five-repeat SGBM-only reference isolates where the CPU time goes.

| Stage | P50 | P95 | What it does |
|---|---:|---:|---|
| Left matcher | 12.87 | 15.79 | 96-disparity SGBM dynamic program |
| Right matcher | 11.91 | 16.04 | Right-view consistency match |
| LR consistency | 2.43 | 3.47 | Reject incompatible disparities |
| Ground model | 13.77 | **18.60** | Ground/v-disparity removal |
| Components | 2.19 | 3.35 | Connected obstacle regions |
| Tracking + TTC | 0.35 | **0.85** | Track update, TTC, risk inputs |
| Full compute | 44.00 | 55.42 | Classical reference |
| Disk loading, excluded | 2.22 | 2.91 | JPEG/decode I/O |

### What the math costs

| Algorithm | Work proxy | Latency conclusion |
|---|---:|---|
| Dual SGBM | 44,236,800 disparity hypotheses/pair | Primary cost; CPU, branch/memory-heavy |
| YOLO26n | 6.119 GFLOPs at 640×640 | GPU cost, but overlaps SGBM |
| Ground processing | no portable FLOP count | Major CPU cost: 18.60 ms P95 alone |
| Components / tracker / TTC / FSM | small array and scalar operations | Under 1 ms P95 in classical reference |
| EKF, Frenet, CPA, occupancy experiments | small matrix/vector math | Not separately certified; expected small relative to SGBM, but not a deployment claim |

Therefore the V2 research math is **not** what consumes latency. It was kept
offline/shadow because its accuracy evidence failed, not because it was too
slow.

## Learned-stereo comparator

LightStereo-S FP32 was measured only on T05, once, with 20 warm-up frames. It
is an exploratory diagnostic—not a deployment benchmark.

| Work | P95 |
|---|---:|
| LightStereo preprocessing | 34.77 ms |
| LightStereo GPU inference | 66.03 ms |
| Ground model | 31.77 ms |
| Whole compute pipeline | **135.50 ms** |
| GPU VRAM | 388.68 MB |

It misses the 75 ms target and produced poor T05 danger output. TensorRT could
reduce *neural inference* but cannot remove the CPU ground/postprocess cost,
and it does not address the observed TTC/path error. Do not invest in it until
a model improves accuracy first.

## Real problem: accuracy, not latency

V1 already passes the latency gate. The remaining errors are:

1. **T01 lateral/turn traffic:** depth reduces even when the target path does
   not intersect ego path.
2. **T05 roadside pedestrians:** visible object is not in the host corridor.
3. **T05 leading vehicle:** object may be correctly detected and in lane but
   is not truly closing; raw depth-rate is unreliable.

Hard IoU, containment association, detector veto, EKF gating, and temporal
depth regression were tested and rejected on accuracy grounds. More latency
optimization will not increase F1.

## Deployment decisions

| Decision | Reason |
|---|---|
| Deploy V1 at 65.91 ms P95 | Meets ≤75 ms gate with 9.09 ms measured headroom |
| Keep 2 OpenCV threads / 2 SGBM matchers | 6 threads / 1 matcher created 101.65 ms P95 contention |
| Do not enable LightStereo | 135.50 ms exploratory P95 and poor T05 result |
| Do not add V2 risk math | Accuracy rejected; latency is not the blocker |
| Future speed work | Optimize SGBM + ground model first; TensorRT only helps neural lane |

