# Phase 02 - Target Extraction and Tracking

**Status:** COMPLETE — Stage 2A retained; Stage 2B YOLO26 candidate rejected  
**Depends on:** Phase 01

## Mục tiêu

Tạo target-level observation và track ổn định cho road objects. Stage 2A ưu
tiên geometry/classical; Stage 2B chỉ bổ sung deep learning khi có failure được
đo rõ.

Thứ tự component:
[`notes/COMPONENT_SEQUENCE.md`](notes/COMPONENT_SEQUENCE.md).

## Component 1 — SGBM confidence

```powershell
python ai_cv/phases/02_detection_tracking/src/analyze_stereo_confidence.py

python ai_cv/phases/02_detection_tracking/src/experiment_lr_consistency.py
```

Kết luận:
[`artifacts/STEREO_CONFIDENCE_001.md`](artifacts/STEREO_CONFIDENCE_001.md).

Hard left-right mask không được promote: composite giảm `19.7 -> 17.8`. LR
residual được giữ làm confidence signal. Ground removal sau đó được đánh giá
trong vertical slice bên dưới.

## Stage 2A classical vertical slice

```powershell
python ai_cv/phases/02_detection_tracking/src/analyze_ground_obstacles.py

python ai_cv/phases/02_detection_tracking/src/experiment_classical_vertical_slice.py

python ai_cv/phases/02_detection_tracking/src/analyze_vertical_slice_results.py
```

Kết luận:
[`artifacts/CLASSICAL_VERTICAL_SLICE_001.md`](artifacts/CLASSICAL_VERTICAL_SLICE_001.md).

Pipeline object-centric tăng danger F1 `0.220 -> 0.402`, tăng TP `80 -> 135` và
giảm FN `124 -> 69`. Tuy nhiên component identity bị fragment mạnh; composite
`28.7` và worst-trip `4.6` chưa vượt robust fixed-ROI reference (`32.2`,
`16.9`). Vì vậy geometry/tracking được giữ, connected components không được
promote làm target identity cuối.

Stage 2B phải tuân theo promotion gate:
[`notes/STAGE2B_EXPERIMENT_GATE.md`](notes/STAGE2B_EXPERIMENT_GATE.md).

The YOLO26 semantic-fusion comparator was diagnosed and corrected for symmetric
component/box containment. Its best global macro danger-F1 was `0.5745`, but
leave-one-trip-out validation remained infeasible and the frozen `0.60` F1
gate was not reachable even by the per-trip oracle. The candidate is therefore
not promoted; see
[`artifacts/yolo26_loto_association_v2/PHASE04B_FINAL_DECISION.md`](artifacts/yolo26_loto_association_v2/PHASE04B_FINAL_DECISION.md).

Vai trò và shortlist deep learning cho stereo/TTC:
[`artifacts/DEEP_STEREO_RESEARCH_001.md`](artifacts/DEEP_STEREO_RESEARCH_001.md).

## Câu hỏi nghiên cứu

- Confidence stereo nào tương quan với depth/TTC error?
- Instance-aware extractor có giảm identity fragmentation và T03 false positive?
- Object detector hay semantic mask tạo end-to-end TTC gain lớn hơn trên cùng
  compute budget?
- Geometry tracking chịu cut-in, occlusion và frame drop đến đâu sau khi có
  object identity ổn định hơn?

## Test cần có

- LR consistency đúng sign, correspondence và boundary.
- Ground removal không xóa obstacle support.
- Empty frame không crash.
- Track confirmation/expiry đúng config.
- Không tái sử dụng history của track hết hạn.

## Verification

- Mỗi component có visual, quantitative ablation và keep/reject decision.
- Báo cáo component coverage, track continuity và degraded behavior.
- Deep learning phải có comparison với classical candidate mạnh nhất.

## Phase 2B latency benchmark

The unified runner measures a decoded stereo pair through the unchanged
track-p35 Guardian TTC path. Disk I/O is reported separately. The active
deployment gate is strict compute P95 `<75 ms`; the earlier `50 ms` value is
retained only as a historical stretch target. See the measured
[Phase 2B latency baseline](artifacts/PHASE02B_LATENCY_BASELINE_001.md).

```powershell
python ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py `
  --backend sgbm `
  --precision fp32 `
  --repeats 5 `
  --warmup-frames 100 `
  --latency-target-ms 75
```

An official candidate must use exactly five repeats, at least 100 warm-up
frames, and exactly 600 ordered unique frames from each of all six trips.
Partial and smoke runs are diagnostic only and can never be accepted.

The same command supports `lightstereo-pytorch`, `lightstereo-onnx`, and
`lightstereo-tensorrt`. Learned runs require `--model-path`; converted
FP16/INT8 or ONNX/TensorRT runs additionally require
`--lane-reference-summary` from the matching PyTorch FP32 run,
`--parity-report` from the frozen 72-pair gate, and the generated artifact
sidecar (inferred as `<model-path>.manifest.json` or supplied with
`--artifact-manifest`). Their dependencies stay in the external OpenStereo
environment and are imported only when selected.

Generated frame timings, predictions, evaluation, environment metadata,
artifact checksum, acceptance gates, and `comparison.csv` are written under
`ai_cv/outputs/benchmarks/phase02b_latency/` and remain ignored by Git.
`nvidia-ml-py` supplies low-overhead process-wide VRAM sampling; if it is
unavailable, the runner polls `nvidia-smi` and records that fallback.

Before a converted backend enters the full benchmark, run its frozen 72-pair
conversion gate:

```powershell
python ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py `
  parity `
  --manifest ~/.local/state/guardian-phase02b/manifests/lightstereo_parity_72.json `
  --data-root ~/guardian-data/phase02b/Practice_Dataset `
  --reference-model-path ~/benchmarks/OpenStereo-assets/checkpoints/LightStereo-S-KITTI.ckpt `
  --candidate-backend lightstereo-onnx `
  --candidate-precision fp32 `
  --candidate-model-path ~/benchmarks/OpenStereo-assets/generated/LightStereo-S-KITTI.opset17.onnx
```

The command verifies the manifest and image hashes, compares the converted
result with PyTorch FP32 on all 72 pairs, and writes an auditable JSON report.
It passes only when mean absolute disparity error is at most `0.25 px`, the
fraction above `3 px` is at most `0.5%`, and missing reference-valid pixels are
at most `0.5%`.
For an INT8 benchmark, also pass the exact generated
`--calibration-manifest` and `--calibration-cache` so their hashes and cache
metadata can be bound to the engine sidecar.

After all candidates finish, merge them and apply the frozen selection rule:

```powershell
python ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py `
  aggregate `
  --summaries ai_cv/outputs/benchmarks/phase02b_latency
```

Candidates within 5% of the fastest accepted P95 are treated as a latency tie;
the candidate with higher danger-F1 wins.

## Exit criteria

- Geometry và causal TTC có implementation/test/report tái lập được.
- Classical identity limitation và Stage 2B promotion gate được ghi rõ.
