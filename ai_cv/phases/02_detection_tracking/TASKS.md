# Tasks - Phase 02

## Stage 2A — Classical object-centric TTC

- [x] Reproduce left SGBM and implement explicit right matcher.
- [x] Test LR sign, correspondence, threshold and boundary behavior.
- [x] Visualize six Stage 1 failure cases.
- [x] Validate raw/filtered depth on 72 sampled keyframes.
- [x] Run full 3,600-frame LR TTC ablation.
- [x] Record matcher-only P50/P95/P99.
- [x] Decide: keep LR residual as confidence; reject hard mask.
- [x] Analyze V-disparity and ground-plane removal.
- [x] Extract obstacle components with a vertical-support/Stixel-lite gate.
- [x] Implement causal component association.
- [x] Estimate robust per-track distance and closing speed.
- [x] Fuse the collision corridor and select path-relevant tracks.
- [x] Run four variants on all 3,600 frames with the official evaluator.
- [x] Report track continuity, confusion and P50/P95/P99 latency.
- [x] Decide: keep geometry/TTC blocks; reject components as final identity.
- [ ] Test optical expansion/flow fallback only if instance-aware stereo remains
  degraded.

## Stage 2B — Deep-learning augmentation gate

- [x] Identify measured failure: component merge/fragment and unstable identity,
  especially on T02/T03/T06.
- [x] Freeze reference metrics, falsification order and promotion criteria.
- [x] Research deep stereo, uncertainty and temporal augmentation roles.
- [x] Freeze stronger-machine ablation ladder and promotion targets.
- [x] Compare detector/semantic candidate with strongest classical pipeline.
- [x] Check license and class mapping; stop latency/robustness work after the
  candidate fails the earlier accuracy gate.
- [x] Decide promotion: reject pretrained YOLO26 fusion; retain the classical
  guarded pipeline for the next phase.
