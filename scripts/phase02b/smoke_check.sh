#!/usr/bin/env bash
set -Eeuo pipefail

readonly OPENSTEREO_COMMIT="23d71c92e33ad1f80dfc42bf29f5c6a914d38769"
readonly CHECKPOINT_SHA256="3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a"
readonly CHECKPOINT_SIZE_BYTES="14159749"
readonly SHARED_NUMPY_VERSION="1.26.4"
readonly SHARED_OPENCV_PACKAGE_VERSION="4.11.0.86"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
guardian_env="${HOME}/.venvs/guardian-phase02b"
openstereo_env="${HOME}/.venvs/openstereo-phase02b"
openstereo_root="${HOME}/benchmarks/OpenStereo"
checkpoint_path="${HOME}/benchmarks/OpenStereo-assets/checkpoints/LightStereo-S-KITTI.ckpt"
data_root="${HOME}/guardian-data/phase02b"
output_path=""
failures=0
openstereo_revision_ready=0
openstereo_source_ready=0
checkpoint_ready=0

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --guardian-env PATH" \
        "  --openstereo-env PATH" \
        "  --openstereo-root PATH" \
        "  --checkpoint PATH" \
        "  --data-root PATH" \
        "  --output PATH          Also save the text report." \
        "  -h, --help"
}

while (($#)); do
    case "$1" in
        --guardian-env)
            guardian_env="$2"
            shift 2
            ;;
        --openstereo-env)
            openstereo_env="$2"
            shift 2
            ;;
        --openstereo-root)
            openstereo_root="$2"
            shift 2
            ;;
        --checkpoint)
            checkpoint_path="$2"
            shift 2
            ;;
        --data-root)
            data_root="$2"
            shift 2
            ;;
        --output)
            output_path="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

report_tmp="$(mktemp)"
ort_smoke_dir=""
trtexec_smoke_dir=""
cleanup() {
    rm -f -- "$report_tmp"
    if [[ -n "$ort_smoke_dir" && -d "$ort_smoke_dir" ]]; then
        rm -f -- "${ort_smoke_dir}/add.onnx"
        rmdir -- "$ort_smoke_dir" 2>/dev/null || true
    fi
    if [[ -n "$trtexec_smoke_dir" && -d "$trtexec_smoke_dir" ]]; then
        rm -f -- \
            "${trtexec_smoke_dir}/add.onnx" \
            "${trtexec_smoke_dir}/add.engine"
        rmdir -- "$trtexec_smoke_dir" 2>/dev/null || true
    fi
}
trap cleanup EXIT

emit() {
    printf '%s\n' "$*" | tee -a "$report_tmp"
}

pass() {
    emit "PASS  $1"
}

fail() {
    emit "FAIL  $1"
    failures=$((failures + 1))
}

detail() {
    local label="$1"
    shift
    local value
    if value="$("$@" 2>&1)"; then
        emit "INFO  ${label}: ${value//$'\n'/; }"
        return 0
    fi
    emit "INFO  ${label}: unavailable (${value//$'\n'/; })"
    return 1
}

emit "Guardian Phase 2B environment smoke check"
emit "Checked UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    pass "Running inside WSL"
else
    fail "Running inside WSL"
fi

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]]; then
        pass "Ubuntu 22.04"
    else
        fail "Ubuntu 22.04 (detected ${PRETTY_NAME:-unknown})"
    fi
else
    fail "/etc/os-release is readable"
fi

home_fs="$(findmnt -n -o FSTYPE -T "$HOME" 2>/dev/null || true)"
case "$home_fs" in
    ""|9p|drvfs|fuseblk)
        fail "Linux HOME uses ext4-compatible storage (detected ${home_fs:-unknown})"
        ;;
    *)
        pass "Linux HOME uses $home_fs"
        ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi is available through the Windows host driver"
    detail "GPU" nvidia-smi \
        --query-gpu=name,memory.total,driver_version,compute_cap \
        --format=csv,noheader || true
else
    fail "nvidia-smi is available through the Windows host driver"
fi

if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    pass "CUDA 12.8 toolkit nvcc exists"
    detail "nvcc" /usr/local/cuda-12.8/bin/nvcc --version || true
else
    fail "CUDA 12.8 toolkit nvcc exists"
fi

if dpkg-query -W -f='${binary:Package}\n' 2>/dev/null |
    grep -Eq '^nvidia-driver(-|$)'; then
    fail "No Linux NVIDIA display-driver package is installed"
else
    pass "No Linux NVIDIA display-driver package is installed"
fi

if [[ -x "$guardian_env/bin/python" ]]; then
    if "$guardian_env/bin/python" -c \
        "import cv2,jsonschema,matplotlib,numpy,pandas,PIL,psutil; from importlib.metadata import version; assert version('numpy') == '${SHARED_NUMPY_VERSION}'; assert version('opencv-python-headless') == '${SHARED_OPENCV_PACKAGE_VERSION}'" \
        >/dev/null 2>&1; then
        pass "Guardian environment imports with shared NumPy/OpenCV pins"
        detail "Guardian Python" "$guardian_env/bin/python" -c \
            "import cv2,numpy,platform; print(platform.python_version(), 'opencv='+cv2.__version__, 'numpy='+numpy.__version__)" ||
            true
    else
        fail "Guardian environment imports with shared NumPy/OpenCV pins"
    fi
else
    fail "Guardian virtual environment exists"
fi

if [[ -x "$openstereo_env/bin/python" ]]; then
    if "$openstereo_env/bin/python" -c \
        "import cv2,numpy,onnx,onnxruntime,pynvml,tensorrt,timm,torch,torchvision; from importlib.metadata import version; assert version('numpy') == '${SHARED_NUMPY_VERSION}'; assert version('opencv-python-headless') == '${SHARED_OPENCV_PACKAGE_VERSION}'" \
        >/dev/null 2>&1; then
        pass "OpenStereo environment imports with shared NumPy/OpenCV pins"
        detail "OpenStereo Python" "$openstereo_env/bin/python" -c \
            "import cv2,numpy,platform; print(platform.python_version(), 'opencv='+cv2.__version__, 'numpy='+numpy.__version__)" ||
            true
    else
        fail "OpenStereo environment imports with shared NumPy/OpenCV pins"
    fi

    if "$openstereo_env/bin/python" -c \
        "import torch; assert torch.cuda.is_available(); module=torch.nn.Conv2d(3,4,3,padding=1).eval().cuda(); x=torch.ones((1,3,16,16),device='cuda'); y=module(x); torch.cuda.synchronize(); assert y.shape==(1,4,16,16) and torch.isfinite(y).all().item()" \
        >/dev/null 2>&1; then
        pass "PyTorch CUDA module forward"
        detail "PyTorch CUDA" "$openstereo_env/bin/python" -c \
            "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" ||
            true
    else
        fail "PyTorch CUDA module forward"
    fi

    ort_smoke_dir="$(mktemp -d)"
    ort_smoke_onnx="${ort_smoke_dir}/add.onnx"
    ort_output=""
    if ort_output="$(
        "$openstereo_env/bin/python" - "$ort_smoke_onnx" 2>&1 <<'PY'
import sys

import numpy as np
import onnx
# Load PyTorch first so its CUDA 12/cuDNN 9 wheel libraries are loaded before
# ONNX Runtime dynamically loads its CUDA execution provider.
import torch
import onnxruntime as ort
from onnx import TensorProto, helper

model_path = sys.argv[1]
input_info = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 1, 2, 2]
)
output_info = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, 1, 2, 2]
)
constant = helper.make_tensor(
    "constant", TensorProto.FLOAT, [1], [2.0]
)
node = helper.make_node("Add", ["input", "constant"], ["output"])
graph = helper.make_graph(
    [node], "phase02b_ort_cuda_smoke", [input_info], [output_info], [constant]
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 17)]
)
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, model_path)

options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(
    model_path,
    sess_options=options,
    providers=[("CUDAExecutionProvider", {"device_id": 0})],
)
session.disable_fallback()
providers = session.get_providers()
assert providers and providers[0] == "CUDAExecutionProvider", providers
output = session.run(
    ["output"], {"input": np.ones((1, 1, 2, 2), dtype=np.float32)}
)[0]
np.testing.assert_array_equal(
    output, np.full((1, 1, 2, 2), 3.0, dtype=np.float32)
)
print("active_provider=" + providers[0])
PY
    )"; then
        pass "ONNX Runtime CUDA-only session execution"
        emit "INFO  ONNX Runtime: ${ort_output//$'\n'/; }"
    else
        fail "ONNX Runtime CUDA-only session execution"
        emit "INFO  ONNX Runtime failure: ${ort_output//$'\n'/; }"
    fi

    if "$openstereo_env/bin/python" -c \
        "import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount()>0; pynvml.nvmlShutdown()" \
        >/dev/null 2>&1; then
        pass "NVML Python binding and GPU access"
        detail "NVML" "$openstereo_env/bin/python" -c \
            "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlSystemGetDriverVersion(), 'devices='+str(pynvml.nvmlDeviceGetCount())); pynvml.nvmlShutdown()" ||
            true
    else
        fail "NVML Python binding and GPU access"
    fi

    if "$openstereo_env/bin/python" -c \
        "import tensorrt as trt; assert trt.Builder(trt.Logger())" \
        >/dev/null 2>&1; then
        pass "TensorRT Python builder"
        detail "TensorRT Python" "$openstereo_env/bin/python" -c \
            "import tensorrt as trt; print(trt.__version__)" ||
            true
    else
        fail "TensorRT Python builder"
    fi
else
    fail "OpenStereo virtual environment exists"
fi

if [[ -d "$openstereo_root/.git" ]]; then
    actual_commit="$(git -C "$openstereo_root" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$actual_commit" == "$OPENSTEREO_COMMIT" ]]; then
        pass "OpenStereo is pinned to $OPENSTEREO_COMMIT"
        openstereo_revision_ready=1
    else
        fail "OpenStereo pin (detected ${actual_commit:-unknown})"
    fi
    if [[ -z "$(git -C "$openstereo_root" status --porcelain --untracked-files=no)" ]]; then
        pass "OpenStereo tracked source is clean"
        if [[ "$openstereo_revision_ready" -eq 1 ]]; then
            openstereo_source_ready=1
        fi
    else
        fail "OpenStereo tracked source is clean"
    fi
else
    fail "OpenStereo checkout exists"
fi

if [[ -f "$checkpoint_path" ]]; then
    checkpoint_size="$(stat --format='%s' "$checkpoint_path")"
    checkpoint_hash="$(sha256sum "$checkpoint_path" | awk '{print $1}')"
    if [[ "$checkpoint_size" == "$CHECKPOINT_SIZE_BYTES" &&
        "$checkpoint_hash" == "$CHECKPOINT_SHA256" ]]; then
        pass "LightStereo-S KITTI checkpoint size and SHA-256"
        checkpoint_ready=1
    else
        fail "LightStereo-S KITTI checkpoint provenance"
    fi
else
    fail "LightStereo-S KITTI checkpoint exists; run fetch_openstereo_checkpoint.sh"
fi

if [[ -x "$openstereo_env/bin/python" &&
    "$openstereo_source_ready" -eq 1 &&
    "$checkpoint_ready" -eq 1 ]]; then
    lightstereo_output=""
    if lightstereo_output="$(
        "$openstereo_env/bin/python" \
            "${script_dir}/lightstereo_runtime_smoke.py" \
            --openstereo-root "$openstereo_root" \
            --checkpoint "$checkpoint_path" \
            2>&1
    )"; then
        pass "Pinned LightStereo-S config, checkpoint, model, and CUDA forward"
        emit "INFO  LightStereo-S: ${lightstereo_output//$'\n'/; }"
    else
        fail "Pinned LightStereo-S config, checkpoint, model, and CUDA forward"
        emit "INFO  LightStereo-S failure: ${lightstereo_output//$'\n'/; }"
    fi
else
    fail "LightStereo-S runtime smoke prerequisites are ready"
fi

data_fs="$(findmnt -n -o FSTYPE -T "$data_root" 2>/dev/null || true)"
if [[ -d "$data_root/Practice_Dataset" &&
    -d "$data_root/Package_starterkit" &&
    "$data_fs" != "9p" &&
    "$data_fs" != "drvfs" &&
    "$data_fs" != "fuseblk" &&
    -n "$data_fs" ]]; then
    pass "Benchmark data is staged on $data_fs"
else
    fail "Benchmark data is staged on WSL ext4"
fi

trtexec_path="$(command -v trtexec || true)"
if [[ -z "$trtexec_path" && -x /usr/src/tensorrt/bin/trtexec ]]; then
    trtexec_path="/usr/src/tensorrt/bin/trtexec"
fi
if [[ -n "$trtexec_path" ]] && "$trtexec_path" --help >/dev/null 2>&1; then
    pass "trtexec is available"
    emit "INFO  trtexec: $("$trtexec_path" --help 2>&1 | sed -n '1p')"

    if [[ -x "$openstereo_env/bin/python" ]]; then
        trtexec_smoke_dir="$(mktemp -d)"
        smoke_onnx="${trtexec_smoke_dir}/add.onnx"
        smoke_engine="${trtexec_smoke_dir}/add.engine"
        if "$openstereo_env/bin/python" - "$smoke_onnx" <<'PY'
import sys

import onnx
from onnx import TensorProto, helper

input_info = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 1, 1, 1]
)
output_info = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, 1, 1, 1]
)
constant = helper.make_tensor(
    "constant", TensorProto.FLOAT, [1], [2.0]
)
node = helper.make_node("Add", ["input", "constant"], ["output"])
graph = helper.make_graph(
    [node], "phase02b_trtexec_smoke", [input_info], [output_info], [constant]
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 17)]
)
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
PY
        then
            if "$trtexec_path" \
                --onnx="$smoke_onnx" \
                --saveEngine="$smoke_engine" \
                --skipInference \
                >/dev/null 2>&1 &&
                "$trtexec_path" \
                    --loadEngine="$smoke_engine" \
                    --shapes=input:1x1x1x1 \
                    --warmUp=0 \
                    --duration=0 \
                    --iterations=1 \
                    >/dev/null 2>&1; then
                pass "trtexec builds and loads a temporary FP32 opset-17 engine"
            else
                fail "trtexec builds and loads a temporary FP32 opset-17 engine"
            fi
        else
            fail "Generate the temporary ONNX smoke model"
        fi
    else
        fail "OpenStereo Python is available for the trtexec smoke model"
    fi
else
    fail "trtexec is available; run install_tensorrt.sh with the NVIDIA local-repo .deb"
fi

emit "Failures: $failures"

if [[ -n "$output_path" ]]; then
    output_parent="$(dirname "$output_path")"
    mkdir -p "$output_parent"
    cp "$report_tmp" "$output_path"
    echo "Report saved: $output_path"
fi

if ((failures > 0)); then
    exit 1
fi

echo "All Phase 2B environment smoke checks passed."
