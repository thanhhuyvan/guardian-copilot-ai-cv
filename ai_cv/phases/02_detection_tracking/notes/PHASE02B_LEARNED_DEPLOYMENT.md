# Phase 2B LightStereo deployment

`lightstereo_deployment.py` freezes the learned-stereo data selections and
generates deployment artifacts without adding OpenStereo source, weights, ONNX
files, TensorRT engines, or calibration caches to Guardian.

## Fixed contract

- OpenStereo commit:
  `23d71c92e33ad1f80dfc42bf29f5c6a914d38769`
- Official checkpoint SHA-256:
  `3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a`
- Config: `cfgs/lightstereo/lightstereo_s_kitti.yaml`
- Native pair: batch 1, `360x640` BGR images
- Model input: `left_img`, `right_img`, each `1x3x384x640`
- Model output: `disp_pred`
- Export: static ONNX opset 17
- TensorRT: major version 10, 2 GiB builder workspace
- INT8: entropy calibration over exactly 300 unlabeled pairs, 50 per trip,
  interleaved across trips and with no overlap with the frozen 72-pair parity
  sample

The calibration preprocessor reproduces the pinned OpenStereo evaluation
transform: BGR-to-RGB, edge replication above the image to `384x640`, division
by 255, and ImageNet mean/std normalization. It refuses to resize a source
image.

## Commands inside WSL2

These commands assume the setup and data-staging defaults in
`docs/PHASE_02B_WSL_SETUP.md`.

```bash
guardian_root=/mnt/d/Python
deploy_script="$guardian_root/ai_cv/phases/02_detection_tracking/src/lightstereo_deployment.py"
state_root="$HOME/.local/state/guardian-phase02b"
asset_root="$HOME/benchmarks/OpenStereo-assets"

"$HOME/.venvs/guardian-phase02b/bin/python" "$deploy_script" \
  generate-manifests \
  --data-root "$HOME/guardian-data/phase02b/Practice_Dataset" \
  --output-dir "$state_root/manifests"
```

The command deterministically writes:

- `lightstereo_parity_72.json`
- `lightstereo_int8_calibration_300.json`

Export and validate the pinned checkpoint:

```bash
"$HOME/.venvs/openstereo-phase02b/bin/python" "$deploy_script" \
  export-onnx \
  --openstereo-root "$HOME/benchmarks/OpenStereo" \
  --checkpoint "$asset_root/checkpoints/LightStereo-S-KITTI.ckpt" \
  --output "$asset_root/generated/LightStereo-S-KITTI.opset17.onnx" \
  --device 0
```

Build the FP16 engine:

```bash
"$HOME/.venvs/openstereo-phase02b/bin/python" "$deploy_script" \
  build-engine \
  --onnx "$asset_root/generated/LightStereo-S-KITTI.opset17.onnx" \
  --output "$asset_root/generated/LightStereo-S-KITTI.fp16.engine" \
  --precision fp16 \
  --device-id 0
```

Build the calibrated INT8 engine:

```bash
"$HOME/.venvs/openstereo-phase02b/bin/python" "$deploy_script" \
  build-engine \
  --onnx "$asset_root/generated/LightStereo-S-KITTI.opset17.onnx" \
  --output "$asset_root/generated/LightStereo-S-KITTI.int8.engine" \
  --precision int8 \
  --calibration-manifest \
    "$state_root/manifests/lightstereo_int8_calibration_300.json" \
  --calibration-cache \
    "$asset_root/generated/LightStereo-S-KITTI.calibration.cache" \
  --data-root "$HOME/guardian-data/phase02b/Practice_Dataset" \
  --device-id 0
```

Each ONNX or engine receives a `.manifest.json` sidecar containing its SHA-256,
source hashes, exact command arguments, dependency versions, names/shapes, and
build settings. An INT8 cache is reused only when its sidecar exactly matches
the ONNX hash, calibration-manifest hash, selection hash, and preprocessing
contract. Otherwise TensorRT must recalibrate all 300 pairs.

## Safety checks

- The ONNX wrapper rejects a non-pinned OpenStereo checkout.
- The checkout must have no tracked changes, the config must be the exact
  tracked in-tree blob, and the checkpoint must match the official SHA-256.
- Checkpoints load with `weights_only=True` on CPU. The audited importer loads
  only LightStereo's required source files and never executes
  `stereo/modeling/__init__.py` or its FoundationStereo/flash-attn imports.
- Timm backbone construction is forced to `pretrained=False`; all weights come
  from the checksum-locked checkpoint, with no implicit network download.
- ONNX validation requires exactly opset 17 and the static
  `left_img`, `right_img` to `disp_pred` interface.
- TensorRT parsing repeats the name and shape checks.
- The INT8 builder rejects missing pairs, an incorrect count, any trip count
  other than 50, a CPU-only PyTorch installation, or an empty calibration
  cache.
- Generated binary/cache artifacts are ignored by Guardian Git. Keep all
  generated files under `~/benchmarks/OpenStereo-assets`.
