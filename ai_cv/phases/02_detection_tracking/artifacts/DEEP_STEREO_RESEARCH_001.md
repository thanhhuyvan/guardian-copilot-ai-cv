# Deep Learning for the Stereo TTC Pipeline

**Status:** RESEARCH SHORTLIST  
**Date:** 2026-07-23  
**Goal:** determine where deep learning produces end-to-end TTC value, rather
than replacing every classical component at once

## 1. Primary evidence from the supplied handbook chapters

The main design reference is now the two supplied 2026 handbook chapters:

- Chapter 18, *Machine Vision*, by Stiller, Bachmann and Salscheider.
- Chapter 19, *Stereo Vision for ADAS*, by Gehrig and Franke.

They support a hybrid design rather than an unconstrained end-to-end
replacement.

### Benchmark and engineering observations

| Evidence from the chapters | Consequence for this project |
|---|---|
| Chapter 19 states that KITTI stereo error metrics were approximately halved after deep stereo became dominant, with most top-100 methods using deep learning. | Learned stereo is a credible accuracy challenger to SGBM. |
| MC-CNN learned the matching cost but retained SGM regularization and reached the top KITTI rank in 2015. | The first deep method should replace a weak component, not discard useful geometry. |
| DispNet demonstrated direct disparity regression, but the chapter reports that architectures explicitly exploiting the epipolar constraint and bounded disparity search perform better. | Prefer stereo-specific models over generic two-image regression. |
| GC-Net-style 3D cost-volume filtering and differentiable sub-pixel ArgMin improved accuracy. | Evaluate boundary quality and sub-pixel disparity, not only valid-pixel density. |
| SegStereo jointly used semantics and disparity to improve ambiguous and low-contrast regions. | Semantic priors are a targeted stereo regularizer, especially for road/object separation. |
| The cited CSPN result required about 1 s per pair on a contemporary high-end GPU. | Benchmark claims without latency are insufficient for ADAS/TTC. |
| Automotive stereo is expected to cover roughly 2-80 m, operate around 25-30 Hz and remain robust to rain, reflections, wipers, darkness and backlight. | Report range slices, adverse cases and P50/P95/P99 latency. |
| Spatially discrete methods achieve around 0.25 px average disparity accuracy even with good calibration; at long range this becomes several metres of depth uncertainty. | TTC confidence must depend on range and disparity uncertainty. |
| A 1 px disparity offset at 60 m can overestimate distance by about 20 m in the chapter's automotive example. | Online/residual calibration is a first-class safety check, not preprocessing trivia. |
| Semantic Stixels suppress ghost obstacles from decalibration, wet-road reflection and horizontal road structure. | Road semantics should specifically target T03 false positives. |
| Instance Stixels separate adjacent same-class Stixels into independent traffic participants. | Instance identity directly addresses Stage 2A merge/fragment failures. |
| Dense stereo-assisted classification reduced false alarms by a reported factor of five at the same detection rate in the cited automotive system. | Measure detector-only versus detector-plus-stereo false alarms. |
| Strong temporal 6D filtering was reported more effective than differential SceneFlow for velocity stability. | Estimate motion over causal tracks; do not derive TTC from two noisy depth frames. |

These are handbook-reported results and historical benchmarks, not reproduced
results on the hackathon dataset. They define hypotheses and evaluation gates,
not expected scores.

## 2. Current measured bottlenecks

The classical vertical slice established two different failure families:

1. **Instance failure**
   - connected components merge objects with background;
   - one physical object fragments into many track IDs;
   - off-path objects can enter the collision corridor.
2. **Stereo/depth failure**
   - thin pedestrians and motorcycles have weak or discontinuous support;
   - night, low-texture and occlusion boundaries produce unstable disparity;
   - per-frame depth noise becomes amplified when differentiated into closing
     speed and TTC.

These failures require different neural methods. A detector cannot repair bad
disparity, and a better stereo matcher cannot provide stable object identity by
itself.

## 3. What deep learning can add

### A. Lightweight object detection or instance segmentation

**Contribution**

- separates road users from geometrically similar road/background regions;
- provides class, confidence and instance boundary;
- stabilizes association using appearance/class in addition to IoU and depth;
- lets depth be estimated only inside the relevant object support.

**Expected impact**

- reduce T03 background false positives;
- reduce component fragmentation on T02/T03/T06;
- preserve T01 pedestrian and T06 motorcycle recall;
- provide the object type required by the product output.

**What it does not solve**

- incorrect stereo correspondence inside a valid detection;
- temporal depth flicker;
- calibrated TTC uncertainty.

This is the first integration step because it addresses the strongest measured
Stage 2A failure.

Chapter 18 also distinguishes one-stage from two-stage detection: the former is
normally faster, while the latter often trades additional compute for accuracy.
For the requested lightweight path, begin with a one-stage detector. Move to
instance segmentation only if box-contained road/background disparity remains a
measured TTC error.

### B. Learned stereo matching

Classical SGBM searches using hand-designed matching and smoothness costs.
Learned stereo uses image features, context and learned cost aggregation to
resolve low texture, repeated texture, exposure change and thin structure.

For this project it can:

- recover denser disparity on pedestrians, motorcycles and vehicle boundaries;
- reduce road/object disparity bleeding inside a box;
- improve zero-shot behavior in night or ambiguous regions;
- retain metric scale through the known stereo calibration:
  `depth = fx * baseline / disparity`.

The output must still be evaluated inside instances and through final TTC.
Pixel-level benchmark quality alone is not a promotion criterion.

### C. Learned confidence and uncertainty

A confidence head can predict where disparity is unreliable rather than using
LR consistency as a binary truth. The TTC layer can then:

- weight object-depth pixels instead of deleting all weak matches;
- reject implausible closing-speed updates;
- emit `confidence`, `ttc_low` and `ttc_high`;
- enter a degraded state instead of reporting a false precise TTC.

This directly supports the requested TTC stream confidence and is a stronger
product contribution than returning a single unqualified number.

The range-aware model should use disparity-domain uncertainty. Since
`Z = fx * baseline / d`, a fixed disparity error produces depth error that grows
approximately quadratically with range. A constant depth-confidence threshold
is therefore physically inappropriate.

### D. Temporal/video stereo

Video stereo reuses previous disparity/features and enforces temporal
consistency. It can reduce frame-to-frame depth jitter, which is especially
important because TTC depends on the derivative of distance.

Potential benefits:

- smoother object depth without a long output delay;
- fewer closing-speed spikes;
- persistent support through brief occlusion or weak texture;
- less computation if the previous state initializes the next frame.

Constraints:

- inference must remain causal;
- state must reset at trip boundaries and camera discontinuities;
- stale state must not survive long frame gaps or scene cuts.

### E. Monocular semantic/geometric priors

Modern zero-shot stereo models can use pretrained monocular features to resolve
ambiguous stereo regions. The monocular prior helps structure and boundaries;
stereo correspondence preserves metric scale.

This is useful for textureless or thin objects, but a pure monocular depth model
is not a direct replacement:

- absolute scale can drift;
- object motion and camera motion can be confused;
- metric TTC becomes harder to calibrate.

For this challenge, monocular depth should be a prior or degraded-mode fallback,
not the primary metric-depth source.

## 4. Architecture derived from the two chapters

The chapter-aligned target architecture is:

```text
rectified stereo pair
  -> lightweight one-stage detector / optional instance mask
  -> learned or SGBM disparity with sub-pixel output
  -> semantic road/background evidence
  -> instance-conditioned robust disparity aggregation
  -> causal temporal track and motion filter
  -> steering-conditioned collision corridor
  -> TTC distribution + degraded state
```

This resembles semantic/instance Stixels conceptually, but uses modern
detector/segmentation outputs and retains our official TTC evaluator.

Three explicit safeguards remain classical:

- calibrated epipolar geometry and metric triangulation;
- causal state estimation;
- physical TTC and collision-corridor constraints.

## 5. Model shortlist for the stronger machine

| Role | Candidate | Why test it | Position |
|---|---|---|---|
| Efficient learned stereo | HITNet | 2D geometric propagation without a full 3D cost volume; designed for real-time stereo | latency-oriented control |
| Efficient zero-shot watchlist | Lite Any Stereo V2 | 2D-only cost aggregation and efficiency-oriented model family | verify fresh release, code and license before promotion |
| Accuracy/compute balance | IGEV-Stereo | geometry encoding volume plus iterative refinement; public implementation and strong cross-dataset behavior | main research candidate |
| Modern zero-shot deployment | Fast-FoundationStereo | distilled/pruned zero-shot family reported more than 10x faster than FoundationStereo | preferred modern candidate if reproducible |
| Quality ceiling | FoundationStereo | strong cross-domain and difficult-material priors | offline oracle, not deployment target |
| Temporal challenger | Stereo Any Video or a causal temporal-stereo method | explicitly targets temporally consistent disparity | only after single-frame winner |

FoundationStereo itself reports roughly 0.7 s for a 375 x 1242 image on an A100,
so it is used to estimate the attainable depth ceiling, not as the default
20 FPS solution.

RAFT-Stereo and CREStereo remain useful fallbacks if the newer releases are
hard to reproduce, but testing every historical stereo model is not the goal.

Lite Any Stereo V2 was released in June 2026 and reports stronger zero-shot
accuracy than Fast-FoundationStereo at 1.8x/2.7x higher speed on the authors'
H200/Orin comparisons. Because it is very recent, those numbers are a shortlist
signal, not evidence on this dataset.

## 6. Required ablation ladder

The following order isolates each source of improvement:

| Experiment | Instance source | Depth source | Question |
|---|---|---|---|
| A0 | fixed ROI | SGBM | frozen Stage 1 reference |
| A1 | classical components | SGBM | frozen Stage 2A recall reference |
| B1 | detector boxes | SGBM | does semantic instance identity solve fragmentation? |
| B1-S | detector + road semantics | SGBM | does semantic road evidence remove T03 ghost obstacles? |
| B2 | detector mask/box | learned stereo | does learned disparity improve TTC beyond B1/B1-S? |
| B2-C | detector mask/box | learned stereo + residual calibration monitor | are long-range errors calibration-driven? |
| B3 | detector mask/box | learned stereo + range-aware uncertainty | does calibrated confidence reduce FP and spikes? |
| B4 | detector mask/box | learned stereo + causal temporal state | does temporal filtering improve TTC stability enough to justify cost? |

Do not jump directly from A0 to B4. Without B1/B2, detector, stereo and temporal
gains cannot be attributed.

## 7. Evaluation gates

### Frame/depth gate

- six frozen Stage 1 failure cases;
- 72 stratified frames;
- depth error against provided keyframes, evaluation-only;
- object support coverage, boundary bleeding and invalid-depth rate;
- sub-pixel disparity residual and error versus range;
- sensitivity to synthetic vertical/offset rectification perturbations;
- GPU latency, peak VRAM and model load time.

### TTC gate

- all 3,600 practice frames;
- official composite, worst trip, critical MAE, inverse-TTC MAE and danger F1;
- TP/FP/FN per trip;
- TTC jitter and selected-track ID switching;
- end-to-end P50/P95/P99 latency.

### Product/robustness gate

- confidence calibration versus actual depth/TTC error;
- night, thin object, occlusion and empty-road slices;
- wet-road/reflection/horizontal-structure and calibration slices when present;
- behavior when no detection, no disparity or model inference fails;
- license, checkpoint source and reproducible environment manifest.

## 8. Promotion targets

The first detector plus learned-stereo candidate should aim to satisfy all:

- mean composite greater than `32.2`;
- worst trip greater than `16.9`;
- danger F1 at least `0.402`;
- TP no lower than `130` and FN no higher than `74`;
- material reduction from the 139 T03 false positives of track p35;
- fewer selected-ID switches on T02/T03/T06;
- causal end-to-end throughput at or above 20 FPS on the target machine, or an
  explicit offline/post-trip deployment statement.

If a heavy stereo model improves quality but misses latency, it remains the
teacher/oracle. It can generate pseudo-labels or guide distillation; it is not
silently presented as the deployable system.

## 9. Recommended research sequence

1. Move the repository and datasets without committing datasets or weights.
2. Record GPU, CUDA, driver, PyTorch and VRAM in a machine manifest.
3. Start with zero-shot inference; do not train on practice depth/TTC labels.
4. Integrate one lightweight detector while keeping SGBM fixed.
5. Add a road-semantic ablation only on the measured T03 ghost failure.
6. Benchmark HITNet, Lite Any Stereo V2, or the most reproducible efficient
   candidate.
7. Benchmark IGEV-Stereo as the accuracy/compute challenger.
8. Run FoundationStereo on the 72-frame set only to estimate the quality ceiling.
9. Promote one learned stereo model to the full 3,600-frame TTC run.
10. Add range-aware uncertainty, then temporal stereo, only if their measured
    failure remains.

## 10. Research references

- Supplied Chapter 18, *Machine Vision*:
  `C:\Users\A\Downloads\978-3-658-45276-6_18.pdf`
- Supplied Chapter 19, *Stereo Vision for ADAS*:
  `C:\Users\A\Downloads\978-3-658-45276-6_19.pdf`
- MC-CNN:
  https://www.jmlr.org/papers/v17/15-535.html
- DispNet / Scene Flow dataset:
  https://openaccess.thecvf.com/content_cvpr_2016/html/Mayer_A_Large_Dataset_CVPR_2016_paper.html
- GC-Net:
  https://openaccess.thecvf.com/content_ICCV_2017/html/Kendall_End-To-End_Learning_of_ICCV_2017_paper.html
- SegStereo:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Guorun_Yang_SegStereo_Exploiting_Semantic_ECCV_2018_paper.html
- Semantic Stixels:
  https://arxiv.org/abs/1604.01715

- RAFT-Stereo:
  https://arxiv.org/abs/2109.07547
- HITNet:
  https://openaccess.thecvf.com/content/CVPR2021/html/Tankovich_HITNet_Hierarchical_Iterative_Tile_Refinement_Network_for_Real-time_Stereo_Matching_CVPR_2021_paper.html
- IGEV-Stereo:
  https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf
- FoundationStereo:
  https://arxiv.org/abs/2501.09898
- Fast-FoundationStereo:
  https://arxiv.org/abs/2512.11130
- Lite Any Stereo V2:
  https://arxiv.org/abs/2606.24457
- Joint disparity and uncertainty estimation:
  https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Learning_the_Distribution_of_Errors_in_Stereo_Matching_for_Joint_CVPR_2023_paper.html
- Stereo Any Video:
  https://arxiv.org/abs/2503.05549
