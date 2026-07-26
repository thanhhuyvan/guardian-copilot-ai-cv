# Phase 2B WSL2 and OpenStereo setup

This runbook creates the reproducible Linux/GPU environment for the Phase 2B
latency benchmark. It does not replace the Guardian checkout at `D:\Python`.
OpenStereo stays external and the benchmark datasets stay out of Git.

These setup assets do not export LightStereo, build its FP16/INT8 engines, or
select the 300-pair INT8 calibration sample. Those model-specific operations
belong to the backend benchmark so that conversion, parity, and latency evidence
remain in one experiment path. The smoke check builds only a disposable,
one-operation FP32 engine to prove that `trtexec` can build and reload engines.

## Safety and storage rules

- Run Windows installation commands only from an **Administrator PowerShell**.
- The scripts are dry-run by default. A mutating setup requires `-Apply` or
  `--apply`.
- Keep the Ubuntu VHD below `D:\WSL\Ubuntu-22.04`. This machine has much less
  free space on C: than D:.
- Use the Windows NVIDIA driver exposed to WSL. Never install an Ubuntu package
  named `nvidia-driver-*`, the `cuda` meta-package, or the `cuda-12-8`
  meta-package. The bootstrap installs only `cuda-toolkit-12-8`.
- Keep OpenStereo at `~/benchmarks/OpenStereo`, pinned to commit
  `23d71c92e33ad1f80dfc42bf29f5c6a914d38769`.
- OpenStereo says its code is academic/non-commercial only. Do not vendor its
  code or weights into Guardian and do not use it as a commercial dependency.
- Do not put credentials in a command, script, Git remote, `.env`, or report.
  Rotate any credential that has previously been pasted into chat and enable
  two-factor authentication.

## 1. Install WSL and place Ubuntu on D:

From `D:\Python` in Administrator PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\phase02b\bootstrap_wsl.ps1

powershell -ExecutionPolicy Bypass -File `
  .\scripts\phase02b\bootstrap_wsl.ps1 -Apply
```

When WSL features are absent, the first applied pass enables WSL without
installing a distro. Restart Windows if requested, then run the same `-Apply`
command again. The second pass updates WSL and uses:

```text
wsl --install --distribution Ubuntu-22.04 \
  --location D:\WSL\Ubuntu-22.04 --no-launch
```

Launch Ubuntu once and create the normal Linux user:

```powershell
wsl.exe --distribution Ubuntu-22.04
```

The PowerShell script is intentionally not able to unregister a distribution.

### Fallback when `--location` is unavailable

First run `wsl.exe --update` and retry. If the updated build still lacks
`--location`, use the Microsoft-supported export/import flow **before**
installing CUDA, Python environments, or datasets:

1. Install and launch a minimal `Ubuntu-22.04` normally.
2. Run `wsl.exe --shutdown`.
3. Export it to `D:\WSL\Ubuntu-22.04.tar`:

   ```powershell
   wsl.exe --export Ubuntu-22.04 D:\WSL\Ubuntu-22.04.tar
   Get-Item D:\WSL\Ubuntu-22.04.tar
   Get-FileHash -Algorithm SHA256 D:\WSL\Ubuntu-22.04.tar
   ```

4. Confirm the archive exists, has a plausible nonzero size, and has a recorded
   hash. Only then run the destructive unregister command:

   ```powershell
   wsl.exe --unregister Ubuntu-22.04
   ```

5. Import the verified archive on D: and confirm WSL 2:

   ```powershell
   wsl.exe --import Ubuntu-22.04 D:\WSL\Ubuntu-22.04 `
     D:\WSL\Ubuntu-22.04.tar --version 2
   wsl.exe --list --verbose
   ```

`--unregister` destroys the registered copy. Never run it without the verified
export. An imported distribution may initially use root; set a normal default
user in `/etc/wsl.conf` before continuing.

## 2. Install the toolkit, Python environments, and OpenStereo

Inside Ubuntu:

```bash
cd /mnt/d/Python
bash scripts/phase02b/bootstrap_wsl.sh
bash scripts/phase02b/bootstrap_wsl.sh --apply
```

The bootstrap:

- installs build tools, Python 3.10, and the pinned CUDA 12.8 toolkit;
- creates `~/.venvs/guardian-phase02b`;
- creates a separate `~/.venvs/openstereo-phase02b`;
- installs NumPy 1.26.4 and OpenCV headless 4.11.0.86 in both environments so
  SGBM and learned-stereo postprocessing use the same numeric runtime;
- installs PyTorch 2.7.1 from the official CUDA 12.8 wheel index;
- installs pinned ONNX, ONNX Runtime GPU, TensorRT, and NVML Python packages;
- clones OpenStereo on the Linux filesystem and detaches it at the required
  commit;
- records resolved `pip freeze`, Ubuntu package, and source manifests under
  `~/.local/state/guardian-phase02b`.

The pinned OpenStereo commit has newer optional model imports than its stale
root `requirements.txt`. Therefore the Phase 2B requirements deliberately use
`timm==1.0.15`, which supplies `timm.layers`; do not replace this environment
with an unrecorded `pip install -r OpenStereo/requirements.txt`.

If system packages and CUDA are already ready, rerun only the Python/source
portion with `--skip-system --apply`.

### Reconfirm the Stage 2A reference inside WSL

Matching NumPy and OpenCV removes an avoidable fairness difference between the
two Phase 2B environments, but this WSL stack must not be assumed bit-identical
to the earlier Windows Stage 2A environment. Before accepting any latency or
quality comparison, run the unoptimized SGBM reference in
`~/.venvs/guardian-phase02b` over all 3,600 practice frames and reconfirm results
near composite `28.7` and danger-F1 `0.402`. Archive the exact package manifest,
predictions, evaluation, and timing report. Treat a material quality difference
as an environment-regression investigation; do not silently redefine the frozen
reference.

## 3. Fetch the exact LightStereo-S KITTI checkpoint

The model weight is an external, ignored asset. Review OpenStereo's
academic/non-commercial restriction, then run:

```bash
bash scripts/phase02b/fetch_openstereo_checkpoint.sh
bash scripts/phase02b/fetch_openstereo_checkpoint.sh --apply
```

Default destination:

```text
~/benchmarks/OpenStereo-assets/checkpoints/LightStereo-S-KITTI.ckpt
```

The fetcher accepts only the Hugging Face object at
`checkpoint/LightStereo/LightStereo-S-KITTI.ckpt`, verifies size `14,159,749`
bytes and SHA-256/LFS oid
`3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a`,
then writes a provenance sidecar. It refuses to overwrite a mismatched file and
refuses a Windows-mounted destination.

## 4. Install matching TensorRT and `trtexec`

The Python wheel supplies the TensorRT builder/runtime but not the `trtexec`
binary required by the hard smoke gate. Download the NVIDIA local-repository
package for:

```text
TensorRT 10.8.0.43
Ubuntu 22.04 x86_64
CUDA 12.8
nv-tensorrt-local-repo-ubuntu2204-10.8.0-cuda-12.8_1.0-1_amd64.deb
```

Accept NVIDIA's license yourself and store the large download on D:, not C:.
Then, inside Ubuntu:

```bash
bash scripts/phase02b/install_tensorrt.sh \
  --repo-deb /mnt/d/path/to/nv-tensorrt-local-repo-ubuntu2204-10.8.0-cuda-12.8_1.0-1_amd64.deb

bash scripts/phase02b/install_tensorrt.sh \
  --repo-deb /mnt/d/path/to/nv-tensorrt-local-repo-ubuntu2204-10.8.0-cuda-12.8_1.0-1_amd64.deb \
  --apply
```

The installer records the source `.deb` SHA-256 and refuses a different
repository identity or TensorRT/CUDA version.

## 5. Stage benchmark data onto WSL ext4

The current Guardian checkout already contains the ignored local practice data
and starter kit. Copy them from the D: mount to the distro filesystem:

```bash
bash scripts/phase02b/stage_data.sh
bash scripts/phase02b/stage_data.sh --apply
```

Default destination:

```text
~/guardian-data/phase02b/Practice_Dataset
~/guardian-data/phase02b/Package_starterkit
```

If the source is still in Downloads, pass the WSL form of that parent path:

```bash
bash scripts/phase02b/stage_data.sh \
  --source-root /mnt/c/Users/bugma/Downloads/guardian-data \
  --apply
```

The source root must directly contain both named directories. The script
refuses a Windows-mounted destination, keeps a 2 GiB free-space reserve,
performs an additive `rsync`, verifies source-file checksums, and writes
SHA-256 manifests. It never deletes stale destination files.

### Freeze and consume the 72-pair parity manifest

Generate the parity and INT8 calibration selections from the staged practice
root. The manifest generator writes the parity file with its fixed filename,
`lightstereo_parity_72.json`:

```bash
guardian_root=/mnt/d/Python
state_root="$HOME/.local/state/guardian-phase02b"

"$HOME/.venvs/guardian-phase02b/bin/python" \
  "$guardian_root/ai_cv/phases/02_detection_tracking/src/lightstereo_deployment.py" \
  generate-manifests \
  --data-root "$HOME/guardian-data/phase02b/Practice_Dataset" \
  --output-dir "$state_root/manifests"
```

Every learned conversion parity run must consume that generated file and the
same staged data root. For example, after producing the ONNX artifact:

```bash
"$HOME/.venvs/openstereo-phase02b/bin/python" \
  "$guardian_root/ai_cv/phases/02_detection_tracking/src/benchmark_stereo_latency.py" \
  parity \
  --manifest "$state_root/manifests/lightstereo_parity_72.json" \
  --data-root "$HOME/guardian-data/phase02b/Practice_Dataset" \
  --reference-model-path \
    "$HOME/benchmarks/OpenStereo-assets/checkpoints/LightStereo-S-KITTI.ckpt" \
  --candidate-backend lightstereo-onnx \
  --candidate-precision fp32 \
  --candidate-model-path \
    "$HOME/benchmarks/OpenStereo-assets/generated/LightStereo-S-KITTI.opset17.onnx" \
  --output \
    ai_cv/outputs/benchmarks/phase02b_latency/lightstereo-onnx/fp32/parity_report.json
```

## 6. Run the hard environment gate

```bash
bash scripts/phase02b/smoke_check.sh \
  --output ~/.local/state/guardian-phase02b/smoke-check.txt
```

Do not start timed benchmark runs until all checks pass:

- WSL2 and Ubuntu 22.04;
- Linux HOME and dataset on the distro filesystem;
- host-provided NVIDIA GPU visibility;
- CUDA 12.8 compiler;
- no Linux display driver;
- Guardian and OpenStereo imports with the exact shared NumPy 1.26.4 and
  OpenCV headless 4.11.0.86 package pins;
- an actual PyTorch `Conv2d` module forward synchronized on CUDA;
- a temporary opset-17 ONNX graph executed by an ONNX Runtime session with
  CPU execution-provider fallback disabled and CUDA active first;
- NVML initialization for low-overhead, process-wide peak-VRAM measurement;
- TensorRT Python builder and `trtexec`;
- a disposable FP32 ONNX opset-17 engine build and reload through `trtexec`;
- exact, clean OpenStereo commit.
- the reviewed `lightstereo_s_kitti.yaml` constructs LightStereo-S through
  Guardian's adapter, strictly loads the verified checkpoint, and completes one
  native `640x360` CUDA forward with finite `360x640` disparity output.

For every benchmark report, archive the smoke report plus the generated package
and source manifests. TensorRT engines remain GPU-, TensorRT-, and shape-specific
artifacts and must not be committed.

If the LightStereo integration smoke reports `FoundationStereo`,
`FastFoundationStereo`, or `flash_attn`, do not install those unrelated,
heavyweight model dependencies. The pinned OpenStereo package aggregator was
reached accidentally; keep the Guardian adapter on a LightStereo-only import
path. Other construction failures identify whether to check the source
revision, config, checkpoint checksum, CUDA-enabled PyTorch, or timm backbone
cache.
