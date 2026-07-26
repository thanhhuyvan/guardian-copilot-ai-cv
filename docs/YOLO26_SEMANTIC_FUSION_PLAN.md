# Phase 04B - YOLO26 Semantic Fusion

## Goal

Add YOLO26n object semantics to Guardian's existing stereo-depth, tracking, and
TTC pipeline. Primary purpose: reject road/fence components responsible for
T05 false alarms without losing T03 danger recall.

YOLO supplies class, box, and confidence only. SGBM remains source of metric
depth. Existing temporal tracker remains source of closing speed and TTC.

## Frozen references and gates

- Branch from `main` at or after `48836b2`.
- Development branch: `research/phase-04b-yolo26-fusion`.
- Frozen non-semantic leave-one-trip-out reference:
  - Macro danger-F1: `0.531`
  - Composite: `38.4`
  - Critical-TTC MAE: `46.638 s`
  - T03 recall: `0.276`
  - T05 false positives: `45`
- Frozen runtime reference:
  - Full-frame SGBM compute P95: `54.40 ms`
  - Deployment gate: strict P95 `<75 ms`
  - Peak process VRAM: `<5 GB`
- Do not tune against scored/private test data.

## Licensing gate

YOLO26 code and official weights use Ultralytics AGPL-3.0 or a commercial
Ultralytics license. Before merging a YOLO26 dependency or derived model into a
product branch, choose one:

1. License Guardian and its complete corresponding source under AGPL-3.0; or
2. Obtain an Ultralytics commercial/R&D license.

Until resolved:

- Treat YOLO26 as a research comparator.
- Keep weights, ONNX files, TensorRT engines, calibration caches, and downloaded
  Ultralytics source out of Git.
- Record model hashes and generation commands.
- Do not describe YOLO26 as an NVIDIA open-source backbone.

Sources:

- https://docs.ultralytics.com/models/yolo26
- https://docs.ultralytics.com/integrations/tensorrt
- https://www.ultralytics.com/license

## Fixed model and runtime

- Model: `yolo26n.pt`, detection task, pretrained COCO weights.
- Classes retained:
  - `0 person`
  - `1 bicycle`
  - `2 car`
  - `3 motorcycle`
  - `5 bus`
  - `7 truck`
- Camera: left BGR frame only.
- Batch: static batch 1.
- First input candidate: letterboxed `640x640`, preserving aspect ratio.
- Second input candidate, tested only if officially supported by export:
  static `384x640`.
- Deployment order:
  1. PyTorch FP32 reference.
  2. ONNX FP32 parity candidate.
  3. TensorRT FP16 deployment candidate.
  4. TensorRT INT8 only if FP16 misses latency or VRAM gate.
- TensorRT engine must be built on target GPU/runtime. Never copy an RTX-built
  engine to Jetson.
- Keep YOLO inference on GPU and CPU SGBM stereo concurrent where safe.

## Interfaces

Add backend-neutral types:

```python
@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float

@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    backend: str
    precision: str
    input_shape: tuple[int, int, int, int]
    model_sha256: str
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

class ObjectDetector(Protocol):
    def infer(self, left_bgr: np.ndarray) -> DetectionResult: ...
    def close(self) -> None: ...
```

Implement backends:

- `none`: empty result; reproduces current Guardian behavior.
- `yolo26-pytorch`
- `yolo26-onnx`
- `yolo26-tensorrt`

Extend benchmark CLI:

```text
--semantic-detector
  none|yolo26-pytorch|yolo26-onnx|yolo26-tensorrt
--semantic-model-path PATH
--semantic-precision fp32|fp16|int8
--semantic-confidence 0.25
--semantic-stride 1
--semantic-fusion off|soft-guard
```

Every benchmark summary must record detector model hash, license note, input
shape, precision, detector stride, class allowlist, fusion thresholds, and
detector stage timings.

## Data flow

```text
left image -> YOLO26 detections -----------------------------+
                                                             |
stereo pair -> SGBM -> obstacle components -> track update --+-> semantic fusion
                                                                  |
                                                                  +-> TTC/risk
```

Run detector and stereo from the same decoded frame. Preserve frame ID and
timestamp. Never use future detections for current-frame output.

## Semantic association

For each disparity component/track:

1. Expand each accepted YOLO box by 10% width and height, clipped to image.
2. Match when either:
   - component/detection IoU is at least `0.15`; or
   - component center lies inside expanded box and vertical overlap is at
     least `0.50`.
3. If several detections match, select highest:

```text
0.60 * detection confidence + 0.40 * IoU
```

4. Store matched class, confidence, IoU, and frame ID on track observation.
5. Never copy YOLO box depth. Sample depth from existing disparity component.

## Temporal semantic state

Maintain per-track state:

```python
semantic_score_t = 0.4 * matched_confidence + 0.6 * semantic_score_t_minus_1
```

- Matched confidence is zero when no detection matches.
- Semantic support is present when score is at least `0.25`.
- Track begins with score zero.
- Reset semantic state when track ID expires.
- Record consecutive detector misses.

`soft-guard` rejects a TTC candidate only when all conditions hold:

- no semantic support;
- at least 3 consecutive detector misses;
- latest stereo depth is greater than `5 m`.

Close-range fallback:

- At depth `<=5 m`, preserve current guarded TTC behavior even without a
  semantic match.
- One missed YOLO frame must never remove a candidate.
- Unknown/unlisted YOLO classes provide no semantic support but do not bypass
  close-range fallback.

Keep current physical guard:

- corridor top width `0.10`
- corridor bottom width `0.50`
- minimum bottom fraction `0.50`
- minimum height fraction `0.05`
- minimum confidence `0.75`
- maximum closing speed `20 m/s`
- maximum depth `20 m`
- maximum motion residual `0.8 m`

## Implementation sequence

### 1. Environment and artifact provenance

- Create separate YOLO environment; do not modify frozen Guardian environment.
- Install pinned Ultralytics, PyTorch/CUDA, ONNX, ONNX Runtime GPU, and
  TensorRT versions.
- Download `yolo26n.pt`.
- Record package versions and SHA-256.
- Confirm `nvidia-smi`, PyTorch CUDA, ONNX CUDA provider, and TensorRT engine
  load.

### 2. Detector-only reference

- Run PyTorch FP32 over all 3,600 left frames.
- Write one detection CSV per trip:

```text
frame_id,timestamp,class_id,class_name,confidence,x0,y0,x1,y1
```

- Save detector-only P50/P95/P99, peak RAM, peak VRAM, and class counts.
- Produce overlays for 72 stratified frames: 12 per trip.
- Manually inspect all T03/T05 overlay frames before fusion.

Stop early if common cars/people are systematically missed. Do not hide poor
detector coverage by changing TTC thresholds.

### 3. Offline fusion replay

- Reuse frozen component/track candidate traces where possible.
- Add detection-to-component association diagnostics.
- Generate predictions for:
  - physical guard only;
  - hard one-frame semantic gate, diagnostic only;
  - specified `soft-guard`.
- Hard gate cannot be promoted.
- Run six-fold leave-one-trip-out selection only for:
  - semantic score threshold: `0.20, 0.25, 0.30`
  - consecutive misses: `2, 3, 4`
  - close fallback depth: `4, 5, 6 m`
- Keep physical guard frozen during this search.
- For each fold, tune on five trips and evaluate once on untouched sixth trip.

Selection objective:

1. Highest training macro danger-F1.
2. Training composite at least `38.4`.
3. Training critical-TTC MAE no worse than physical guard on same five trips.
4. Tie: higher recall, then simpler/default-nearest configuration.

### 4. Live pipeline integration

- Integrate chosen fusion into common `GuardianTtcPipeline`.
- Preserve `--semantic-detector none` bit-identical output.
- Reset detector/tracker/fusion state between trips and repeats.
- Report detector preprocessing, inference, postprocessing, association, and
  total pipeline timings separately.
- Run detector every frame first.
- Test stride 2 only if stride 1 misses latency gate. Reuse tracked semantic
  state on skipped frames; never reuse a future detection.

### 5. Conversion

- Export static ONNX and validate with `onnx.checker`.
- Compare PyTorch and ONNX on frozen 72-frame sample.
- Build TensorRT FP16 engine on RTX 3060.
- Compare TensorRT FP16 against PyTorch reference.
- Keep generated artifacts ignored; commit manifest containing hashes,
  commands, tool versions, and input/output tensor contracts.

INT8 activates only if FP16 fails P95 or VRAM:

- Calibrate with 300 unlabeled pairs, 50 per trip.
- Exclude frozen 72-frame parity sample.
- Re-run detection and end-to-end parity.

### 6. Official evaluation

- Warm up 100 frames.
- Reset all state.
- Run all six trips five times.
- First repeat produces predictions; all repeats produce latency.
- Require deterministic danger outputs across repeats.
- Report P50/P95/P99 and FPS excluding file I/O; report file I/O separately.

## Tests

Unit tests:

- Letterbox and inverse box mapping.
- Class allowlist.
- Box expansion and clipping.
- IoU/center/vertical-overlap association.
- Deterministic best-match selection.
- Semantic EMA.
- Consecutive-miss reset.
- Close-range fallback.
- Track expiration clears semantic state.
- `none` backend reproduces current output.
- NaN/Inf/empty detector output fails safely.

Conversion tests:

- Static shape and tensor names.
- Finite boxes/confidences.
- Boxes remain inside native image.
- Class agreement on matched detections at least `99%`.
- Median matched box IoU at least `0.98`.
- Mean confidence difference at most `0.02`.
- Compare danger output, not only raw detector tensors.

Integration tests:

- Detector miss for one frame does not remove track.
- Persistent unmatched far component becomes suppressed.
- Unmatched component at `<=5 m` survives.
- Trip reset prevents state contamination.
- Detector failure falls back to physical guard and records degraded mode.

## Acceptance

Promote only when every hard gate passes:

- Leave-one-trip-out macro danger-F1 `>=0.60`.
- Leave-one-trip-out composite `>=38.4`.
- Leave-one-trip-out critical-TTC MAE `<=46.638 s`.
- T05 false positives `<=20`.
- T03 recall `>=0.276`.
- Complete pipeline compute P95 `<75 ms`.
- Peak process VRAM `<5 GB`.
- No missing frames, crashes, NaN/Inf detector tensors, or state leakage.
- `none` backend remains bit-identical to commit `48836b2`.

Prefer FP16 over INT8 when latency differs by less than 5%.

Reject YOLO fusion if it fails F1 or T03 recall after the fixed 27-policy
semantic search. Do not start another broad threshold sweep.

## If pretrained YOLO26 fails

Classify failure before training:

- Detector misses true road users: collect box annotations.
- Detector sees object but association fails: fix fusion geometry.
- Semantic fusion succeeds but danger remains wrong: fix depth/motion/TTC.
- Latency fails: test stride 2, then INT8.

Fine-tuning is a separate phase:

- Annotate 100-200 diverse frames emphasizing T03/T05.
- Maintain trip-level separation.
- For unbiased leave-one-trip-out evaluation, train six fold-specific models;
  never train a fold model on its held-out trip.
- Use batch 1-2, FP16, frozen backbone first, and gradient accumulation.
- Distillation uses offline teacher pseudo-labels so teacher and student are not
  loaded together on 6 GB GPU.

## Proposal updates after measurement

Update proposal pages 14-15, 28, and 33:

- Replace unverified shared-backbone claim with modular prototype description.
- Name `YOLO26n + TensorRT FP16` and exact license.
- Replace estimated 35 ms/180 MB/28 MB values with measured results.
- Keep shared multi-task backbone as future production optimization.
- State measured NORMAL/ATTENTIVE/HIGH-RISK FPS; do not claim 30 FPS unless
  complete Fast Path P95 is below `33.3 ms`.

## Deliverables

- Detector adapter and three backends.
- Detector/fusion tests.
- 72-frame parity manifest and report.
- Six-fold leave-one-trip-out report and chart.
- Five-repeat latency report.
- Model provenance manifest.
- Updated metric benchmark Markdown.
- Updated proposal claims based on measured evidence.

