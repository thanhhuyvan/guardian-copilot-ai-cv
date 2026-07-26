# Phase 2B latency baseline 001

## Decision

Promote the optimized full-frame SGBM implementation as the Phase 2B
classical reference. It preserves the frozen Stage 2A predictions and passes
the revised deployment requirement of native batch-1 compute P95 below
`75 ms`.

The original `50 ms` stretch target was not met. On 2026-07-26 the deployment
gate was revised to strict P95 `< 75 ms`; the measured result did not need to
be rerun or extrapolated to establish that decision.

## Reproducible run

- Source base: `e066808b4ef39874d03b28447674abf514a04602`
- Branch: `research/phase-02b-latency`
- Input: six practice trips, 600 frames per trip, native `640x360`, batch 1
- Warm-up: 100 stereo pairs
- Measurement: all 3,600 frames, five repeats, 18,000 timing rows
- OpenCV configuration: six threads, one sequential matcher lane
- Timing boundary: decoded left/right pair through TTC/risk output
- Disk loading: measured separately and excluded from the compute gate
- Hardware: RTX 3060 Laptop GPU (6 GiB), NVIDIA driver 572.16; SGBM is CPU-only

## Results

| Candidate | Repeats | P50 (ms) | P95 (ms) | P99 (ms) | Mean FPS | Composite | Danger-F1 | Worst trip | Peak RAM | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Full-frame optimized SGBM | 5 | 44.00 | 55.42 | 62.38 | 22.38 | 28.7 | 0.402 | 4.6 | 302.3 MiB | Pass `<75 ms` |
| SGBM crop, top 96 rows | 1 | 36.32 | 46.58 | 53.58 | 26.91 | 27.3 | 0.400 | 8.5 | not sampled | Reject quality |

The crop saved latency but lost `1.4` composite points. That exceeds the
maximum allowed loss of `0.5`, so it was stopped before the five-repeat timing
protocol. The full-frame candidate lost `0.0` composite and `0.000` danger-F1.

The benchmark's direct, unrounded danger classification count was
`TP=137`, `FP=341`, `FN=67`. The official evaluator result remains the source
of truth for composite and danger-F1.

## Full-frame stage timing

| Stage | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---:|---:|---:|
| File loading (excluded) | 2.22 | 2.91 | 3.38 |
| Stereo total | 27.91 | 34.98 | 37.63 |
| Left matcher | 12.87 | 15.79 | 16.69 |
| Right matcher | 11.91 | 16.04 | 17.06 |
| LR consistency | 2.43 | 3.47 | 4.12 |
| Ground model | 13.77 | 18.60 | 23.44 |
| Components | 2.19 | 3.35 | 4.26 |
| Tracking/TTC | 0.35 | 0.85 | 1.19 |
| Decoded-pair compute | 44.00 | 55.42 | 62.38 |
| Compute plus file loading | 46.31 | 58.03 | 64.99 |

## Behavior parity

All 24 prediction CSV files from the optimized implementation had identical
SHA-256 content to the detached `e066808` reference run. The official
evaluation was unchanged:

- composite `28.7`
- danger-F1 `0.402`
- worst-trip composite `4.6`

## Next step

The classical lane now satisfies the revised deployment latency requirement.
ONNX Runtime and TensorRT work remains a comparator lane, not a blocker for
the baseline. It starts only after the documented WSL2/CUDA environment is
installed. Structured pruning remains disabled unless an unpruned learned
candidate is required and misses the active latency target.
