# Separate Domain-Adaptive Stereo Experiment

**Branch:** `research/phase-07-domain-adaptive-stereo`  
**Status:** BLOCKED ON WSL SETUP — baseline remains unchanged

## Purpose

Test whether self-supervised domain adaptation of a learned stereo model can
improve adverse-image depth reliability (T03/rain/night and the measured
sensor-noise sensitivity) without regressing Guardian's frozen clean pipeline.

This is an experiment, not a replacement deployment. The current live
SGBM+YOLO candidate stays frozen on the Phase 06 branch and remains the
fallback until every promotion gate passes.

## Chosen method

Start with a **separate LightStereo-S KITTI checkpoint** and train only an
adapted copy using unlabeled rectified stereo pairs. The loss must contain:

1. left-right photometric reconstruction with an occlusion/validity mask;
2. left-right disparity consistency;
3. edge-aware smoothness;
4. confidence masking so uncertain pseudo-labels do not become supervision;
5. a clean-reference regularizer to limit catastrophic forgetting.

This uses the practical parts of AdaStereo and Learning-to-Adapt-for-Stereo:
input/feature alignment and confidence-masked self-supervision. It does not
claim to reproduce either paper exactly.

## Data split — mandatory

| Role | Data | Rule |
|---|---|---|
| Source pretraining | Official LightStereo-S KITTI checkpoint | External, SHA-256 verified |
| Adaptation input | Unlabeled Guardian stereo pairs from five trips per fold | No TTC labels or future information |
| Validation | The held-out sixth Guardian trip | Never used to select gradients, epochs, or augmentations |
| External stress | Adverse-weather stereo dataset, such as DrivingStereo | External and versioned separately |
| Clean reference | Frozen Phase 06 predictions and metrics | Must not regress |

Run six leave-one-trip-out folds. For a held-out trip, choose the epoch only
from a fixed split of the five training trips; evaluate the held-out trip once.
Do not adapt on all six trips then score all six as generalization.

## Promotion gates

The adapted model is rejected unless all are true:

- held-out macro danger-F1 improves over the current `0.6579` reference, or
  T03/noise recall improves materially without a macro regression;
- clean full-set macro F1 does not fall by more than `0.01`;
- clean composite does not fall by more than `0.5` points;
- critical-TTC MAE does not worsen;
- native `640×360`, batch one compute P95 remains at or below `75 ms`;
- peak process VRAM remains at or below `5 GB`;
- no NaN/Inf disparity beyond the current baseline's invalid-mask behavior;
- all output contracts and tracker resets remain valid.

If the model improves T03 but misses latency, retain it only as an offline
teacher/oracle. It must not replace the deployable backend.

## Required environment

The deliberate runtime boundary is WSL Ubuntu 22.04 on D:, using the existing
Windows NVIDIA driver. OpenStereo and assets stay outside the Guardian Git
checkout and on WSL ext4:

```text
~/benchmarks/OpenStereo
~/benchmarks/OpenStereo-assets
~/guardian-data/phase02b
```

The exact pinned OpenStereo commit is
`23d71c92e33ad1f80dfc42bf29f5c6a914d38769`; the expected LightStereo-S KITTI
checkpoint SHA-256 is
`3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a`.

## First unblock action

Run the following from an **Administrator PowerShell** at `D:\Python`:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\phase02b\bootstrap_wsl.ps1 -Apply
```

Windows may require a restart. After restart, run the same command again, then
launch Ubuntu once and run the pinned Linux bootstrap from the Phase 2B guide.
No password, token, model file, dataset, or experiment output belongs in Git.

## Evidence base

- AdaStereo: [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_AdaStereo_A_Simple_and_Efficient_Approach_for_Adaptive_Stereo_Matching_CVPR_2021_paper.pdf)
- Tonioni et al.: [Learning to Adapt for Stereo](https://openaccess.thecvf.com/content_CVPR_2019/html/Tonioni_Learning_to_Adapt_for_Stereo_CVPR_2019_paper.html)
- Phase 06: `ai_cv/phases/06_robustness_latency/artifacts/ROBUSTNESS_LATENCY_REPORT.md`
