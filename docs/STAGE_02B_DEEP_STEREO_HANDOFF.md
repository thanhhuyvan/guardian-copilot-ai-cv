# Guardian CoPilot AI/CV - Stage 2B Deep Stereo Handoff

**Status:** Temporary working strategy  
**Branch:** `research/phase-02a-classical-ttc`  
**Last verified Stage 2A commit:** `8d1c420`  
**Purpose:** resume research quickly on a stronger GPU machine

## 1. Product scope

Guardian CoPilot is a **vision-only collision-risk analytics system** for fleet
operations.

Primary deployment direction:

- road-facing stereo video;
- telemetry such as ego speed, brake and steering;
- causal TTC estimation;
- risk events and evidence clips;
- post-trip fleet analysis, driver coaching and route-risk insight.

The current solution does **not** use radar. Future sensor-fusion compatibility
may be mentioned as a roadmap item, but it is not part of the hackathon
implementation or core pitch.

## 2. Measured starting point

### Official baseline

```text
SGBM -> fixed ROI -> median depth -> five-frame slope -> TTC
```

| Metric | Result |
|---|---:|
| Mean composite | 19.7 |
| Worst trip | 5.0 |
| Danger F1 | 0.220 |
| TP / FP / FN | 80 / 436 / 124 |
| Effective throughput | 18.28 FPS |

### Stage 1 robust fixed ROI

| Metric | Result |
|---|---:|
| Mean composite | 32.2 |
| Worst trip | 16.9 |
| Danger F1 | 0.258 |

This remains the strongest score/robustness reference.

### Stage 2A classical object-centric TTC

```text
SGBM
  -> V-disparity ground model
  -> vertical-support obstacle components
  -> causal association
  -> robust per-track motion
  -> collision corridor
  -> TTC
```

Best danger-recall variant:

| Metric | Track p35 |
|---|---:|
| Mean composite | 28.7 |
| Worst trip | 4.6 |
| Danger F1 | 0.402 |
| TP / FP / FN | 135 / 340 / 69 |

Stage 2A proves that object-centric geometry recovers threats, but connected
components merge and fragment. Selected track IDs are especially unstable on
T02, T03 and T06. T03 still produces 139 danger false positives.

## 3. Evidence from the supplied handbook chapters

Primary references:

- Chapter 18: *Machine Vision*.
- Chapter 19: *Stereo Vision for ADAS*.

Key lessons:

1. Deep stereo approximately halved historical KITTI errors and became dominant
   on the benchmark.
2. Successful systems often retain stereo geometry. MC-CNN learned matching
   cost but still used SGM regularization.
3. Stereo-specific architectures that preserve the epipolar constraint perform
   better than unconstrained two-image disparity regression.
4. Semantic evidence improves disparity in ambiguous and low-texture regions.
5. Semantic road evidence can remove ghost obstacles caused by road structure,
   reflections and slight decalibration.
6. Instance information separates adjacent same-class geometric elements.
7. Sub-pixel accuracy and calibration are critical. Depth error grows roughly
   quadratically with range for a fixed disparity error.
8. Temporal filtering is necessary because differentiating noisy distance
   directly produces unstable velocity and TTC.
9. Automotive real-time references target roughly 25-30 Hz, but this project
   currently consumes 20 FPS input.

These are literature findings, not reproduced results on the hackathon dataset.

## 4. Target architecture

```text
rectified stereo pair
  -> lightweight one-stage detector
  -> optional road semantics / instance mask
  -> SGBM or learned sub-pixel disparity
  -> instance-conditioned robust depth
  -> causal temporal track and motion state
  -> steering-conditioned collision corridor
  -> TTC distribution + degraded state
  -> event, evidence clip and fleet insight
```

Classical safeguards retained:

- calibrated epipolar geometry;
- metric triangulation using `depth = fx * baseline / disparity`;
- causal temporal filtering;
- physical closing-speed and TTC constraints;
- explicit degraded-state behavior.

## 5. What deep learning is expected to contribute

### Lightweight detector

- object identity, class and confidence;
- fewer component merges/fragments;
- stable association using appearance and class;
- object type required by TTC logs and fleet reports.

It does not repair incorrect disparity by itself.

### Road semantics or instance segmentation

- removes road/background pixels from object depth;
- targets T03 ghost obstacles;
- separates adjacent objects and improves depth boundaries.

Use only if detector boxes plus SGBM leave a measured mixed-depth failure.

### Learned stereo

- stronger matching in low texture, night, repeated texture and thin objects;
- denser object support;
- improved boundary and sub-pixel disparity.

It must improve final TTC, not only produce visually cleaner depth.

### Range-aware uncertainty

- predicts whether disparity/depth is trustworthy;
- prevents weak stereo support from becoming a precise TTC claim;
- supports `confidence`, TTC interval and degraded reason.

### Temporal stereo/filtering

- reduces frame-to-frame depth jitter;
- stabilizes closing speed;
- bridges short occlusion or weak-texture intervals.

All inference used for evaluation must remain causal.

## 6. Required ablation ladder

| ID | Instance source | Depth source | Question |
|---|---|---|---|
| A0 | Fixed ROI | SGBM | Frozen official/Stage 1 reference |
| A1 | Classical components | SGBM | Frozen Stage 2A recall reference |
| B1 | Lightweight detector | SGBM | Does instance identity solve fragmentation? |
| B1-S | Detector + road semantics | SGBM | Does road evidence remove T03 ghosts? |
| B2 | Detector/mask | Learned stereo | Does learned disparity improve TTC beyond B1? |
| B2-C | Detector/mask | Learned stereo + calibration monitor | Are long-range errors calibration-driven? |
| B3 | B2 winner | Range-aware uncertainty | Does confidence reduce FP and TTC spikes? |
| B4 | B3 winner | Causal temporal state | Does temporal consistency justify its cost? |

Do not jump directly to B4. Each stage isolates one source of improvement.

## 7. Initial model shortlist

| Role | Candidate | Usage |
|---|---|---|
| Lightweight detection | One-stage detector with vehicle/person/bicycle/motorcycle classes | B1 reference |
| Efficient stereo | HITNet | Latency-oriented control |
| Efficient zero-shot watchlist | Lite Any Stereo V2 | Verify code, license and reproducibility |
| Accuracy/compute challenger | IGEV-Stereo | Main learned-stereo challenger |
| Modern zero-shot challenger | Fast-FoundationStereo | Test if release is reproducible |
| Quality ceiling | FoundationStereo | 72-frame oracle only |

Do not test every historical model. Promote at most one efficient and one
accuracy-oriented learned-stereo candidate to the full run.

## 8. Falsification sequence

1. Six frozen failure cases:
   - T01 #324 pedestrian FN;
   - T03 #293 empty-road FP;
   - T04 #265 lead-car TP;
   - T05 #314 off-path pedestrian FP;
   - T05 #469 lead-car FN;
   - T06 #146 motorcycle FN.
2. Labeled danger frames plus matched empty-road negatives.
3. A 72-frame stratified sample.
4. Sub-pixel/range and small rectification-offset checks.
5. Full 3,600-frame practice evaluation.

A candidate that fails an earlier gate is not tuned on the full set.

## 9. Evaluation metrics

### Organizer metrics

- mean composite;
- worst-trip composite;
- critical-zone MAE;
- inverse-TTC MAE;
- danger precision, recall and F1;
- TP/FP/FN per trip.

### Robustness metrics

- thin object, night, empty-road and occlusion slices;
- disparity/depth error by distance;
- selected-track ID switches;
- TTC jitter;
- confidence calibration;
- missing detection/disparity behavior.

### Compute metrics

- end-to-end P50/P95/P99 latency;
- detector and stereo latency separately;
- peak GPU VRAM;
- model load time;
- throughput per trip for post-trip processing.

### Product metrics

- false events per trip/minute;
- estimated manual clip-review workload;
- fraction of events marked degraded;
- event coverage by object type.

F2 may be reported as a secondary safety diagnostic. F1/composite remain the
primary challenge metrics.

## 10. Promotion targets

A Stage 2B candidate should aim to satisfy all:

- mean composite greater than `32.2`;
- worst trip greater than `16.9`;
- danger F1 at least `0.402`;
- TP no lower than `130`;
- FN no higher than `74`;
- material reduction from 139 T03 false positives;
- fewer selected-ID switches on T02/T03/T06;
- causal inference;
- license and checkpoint provenance recorded;
- no evaluation-label use during inference.

For an in-car research claim at 20 FPS, target end-to-end P95 at or below
50 ms. For the current post-trip product, also report cost/throughput rather than
claiming real-time as a requirement.

## 11. Claims discipline

Every important statement must carry one status:

| Status | Meaning |
|---|---|
| `MEASURED` | Reproduced on the hackathon practice dataset |
| `PDF` | Supported by the two supplied handbook chapters |
| `LITERATURE` | Supported by an external primary paper/repository |
| `HYPOTHESIS` | Not yet demonstrated |

Examples:

- `MEASURED`: track p35 achieves danger F1 `0.402`.
- `PDF`: semantic road evidence suppresses ghost Stixels.
- `LITERATURE`: FoundationStereo targets zero-shot stereo generalization.
- `HYPOTHESIS`: detector-guided stereo cropping reduces latency without
  degrading TTC.

Do not convert a target or hypothesis into a pitch result.

## 12. Stronger-machine startup checklist

1. Clone and switch branch:

   ```powershell
   git clone https://github.com/thanhhuyvan/guardian-copilot-ai-cv.git
   cd guardian-copilot-ai-cv
   git switch research/phase-02a-classical-ttc
   ```

2. Copy datasets and the two reference PDFs separately. They are intentionally
   excluded from Git.
3. Record:
   - GPU model and VRAM;
   - `nvidia-smi`;
   - OS and driver;
   - CUDA/cuDNN;
   - Python, PyTorch and OpenCV versions.
4. Recreate the environment; do not copy `.venv`.
5. Run existing unit and structure checks.
6. Reproduce one frozen baseline before installing model candidates.
7. Start B1 with detector plus unchanged SGBM.
8. Do not commit datasets, generated predictions, videos or pretrained weights.

## 13. Product positioning

Use this temporary positioning:

> Guardian CoPilot is a vision-only fleet collision-risk analytics system. It
> combines lightweight instance perception with metric stereo, semantic road
> evidence and causal temporal filtering to produce TTC with explicit
> confidence. The system turns video into reviewable risk events, evidence clips
> and route/driver insights rather than returning an unexplained TTC number.

Radar or additional sensors belong only to the future integration roadmap.

## 14. Detailed project references

- `ai_cv/phases/01_data_baseline/artifacts/STAGE_01_REPORT.md`
- `ai_cv/phases/02_detection_tracking/artifacts/STEREO_CONFIDENCE_001.md`
- `ai_cv/phases/02_detection_tracking/artifacts/CLASSICAL_VERTICAL_SLICE_001.md`
- `ai_cv/phases/02_detection_tracking/artifacts/DEEP_STEREO_RESEARCH_001.md`
- `ai_cv/phases/02_detection_tracking/notes/STAGE2B_EXPERIMENT_GATE.md`

