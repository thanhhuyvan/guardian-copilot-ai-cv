# Execution Plan — Confidence Repair and Separate Domain Adaptation

**Baseline to preserve:** Phase 06 live candidate, macro danger-F1 `0.6579`,
compute P95 `65.91 ms`, VRAM `460.68 MB`.

## Decision rule

Run two independent lanes. Neither replaces the baseline automatically.

```text
Current SGBM + YOLO deployment (frozen)
        |
        +-- Lane A: confidence / temporal measurement repair
        |
        +-- Lane B: separately trained adapted LightStereo
                 |
                 +-- promote only if accuracy + clean regression + latency gates pass
```

## Lane A — start while WSL installs

### Goal

Reduce unreliable depth/TTC updates under T03 rain/night and synthetic noise,
without using a learned model or changing object semantics.

### Implementation steps

1. Record per-object signals already available from SGBM:
   left/right agreement, valid-support fraction, disparity spread, ground-model
   confidence, track age, and motion residual.
2. Define an interpretable `depth_quality` score. Do not fit a large classifier.
3. When quality is low, mark the measurement unreliable. Reuse a previous
   depth only if association and ego-motion-compensated residual agree; otherwise
   emit `UNKNOWN` rather than stale TTC.
4. Add bounded temporal smoothing for reliable tracks only; reset it at trips,
   missing frames, timestamp faults, and camera faults.
5. Run six-trip LOTO selection, Phase 06 medium noise screen, and the full
   clean latency benchmark.

### Gates

- Macro F1 must not fall below `0.6579`.
- Critical-TTC MAE must not worsen.
- Noise-screen F1 loss must decrease from the current `-0.1133`.
- No stale alert may cross an injected `UNKNOWN` frame.
- Compute P95 must remain at or below `75 ms`.

### Time estimate

| Task | Estimate |
|---|---:|
| Instrumentation and unit tests | 2–4 hours |
| LOTO/screening/full evaluation | 1–2 hours |
| Total | 1 focused workday |

### Realistic result estimate

This repairs measurements rather than collision-path logic. A plausible gain is
macro F1 `+0.01` to `+0.04` (`0.67–0.70`), principally through better T03/noise
recall and fewer TTC spikes. It may be rejected if it suppresses true hazards.

## Lane B — separate domain-adapted LightStereo

### Prerequisites

1. WSL Ubuntu 22.04 is installed under `D:\WSL\Ubuntu-22.04`.
2. Run the pinned Linux bootstrap and record environment manifests.
3. Clone external OpenStereo at commit
   `23d71c92e33ad1f80dfc42bf29f5c6a914d38769`.
4. Fetch the SHA-256-verified LightStereo-S KITTI checkpoint.
5. Stage the data to WSL ext4; never train from `/mnt/d`.
6. Run CUDA, PyTorch, ONNX Runtime, TensorRT, and baseline stereo smoke checks.

### Experiment ladder

| Stage | Training data | Purpose | Promotion evidence |
|---|---|---|---|
| B0 | None | LightStereo-S zero-shot parity and latency | 72-frame geometry/parity report |
| B1 | Five unlabeled Guardian trips | Self-supervised stereo adaptation | Held-out sixth-trip disparity/TTC result |
| B2 | External adverse-weather stereo data + B1 | Broader domain coverage | External stress + Guardian LOTO |
| B3 | Best B1/B2 checkpoint | ONNX/TensorRT deployment | Full clean quality + P95/VRAM |

### Training configuration

- static batch `1`, crop/pad compatible with `384×640`;
- start with `10` epochs, early stop from a fixed five-trip validation split;
- keep source-checkpoint regularization to limit clean-domain forgetting;
- use photometric reconstruction, left-right consistency, edge-aware
  smoothness, and confidence/occlusion masking;
- no TTC label, danger label, held-out-trip frame, or future frame enters the
  training loss.

### Gates

- Adapted model must beat its own B0 reference on held-out target data.
- Full Guardian macro F1 must improve over `0.6579`, or materially improve
  T03/noise recall without a macro regression.
- Clean macro F1 may lose at most `0.01`; composite at most `0.5` points.
- Critical-TTC MAE cannot worsen.
- Native `640×360`, batch-one P95 must be `≤75 ms`; VRAM `≤5 GB`.
- Failing a gate retains the current SGBM deployment unchanged.

### Time estimate after WSL is ready

| Task | Estimate |
|---|---:|
| WSL/OpenStereo/checkpoint/data bootstrap | 1–3 hours, network dependent |
| B0 parity and latency | 30–60 minutes |
| One 10-epoch fold on RTX 3060 6 GB | 30–90 minutes |
| Six LOTO folds | 3–9 hours, best run overnight |
| Full evaluation and deployment conversion | 1–2 hours |
| Total experiment cycle | 1–2 days |

### Realistic result estimate

Domain adaptation targets disparity under adverse appearance. It cannot decide
whether a real adjacent car is on a collision path, so it cannot resolve T05
alone. A successful adapted model is realistically worth `+0.00` to `+0.04`
macro F1 over the frozen candidate (`0.66–0.70`), with a stronger expectation
for T03/noise robustness than for global F1. A negative or neutral result is
credible and is a valid rejection outcome.

## Combined expectation

If both Lane A and a later future-corridor-overlap method pass LOTO, a
defensible macro F1 target is `0.70–0.75`. Domain adaptation by itself should
not be presented as a route to `>0.80`; that would require new representative
labelled trajectory/risk data and external validation.

## Immediate sequence

1. Let the WSL bootstrap finish; do not install Linux NVIDIA display drivers.
2. Implement and evaluate Lane A in the current Windows pipeline.
3. Complete B0 before any adaptation training.
4. Run B1 one fold first as a stop/go check; only then schedule all six folds.
5. Promote the lowest-latency candidate that passes every gate.
