#!/usr/bin/env bash
set -Eeuo pipefail

readonly CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
readonly CUDA_KEYRING_SHA256="eea6cc5f0eaeb99082d054b8c05ae206a378e31e88048df0310d59f651dceed2"
readonly CUDA_TOOLKIT_PACKAGE="cuda-toolkit-12-8=12.8.1-1"
readonly PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
readonly NVIDIA_PYPI_INDEX="https://pypi.nvidia.com"
readonly PYTORCH_VERSION="2.7.1"
readonly TORCHVISION_VERSION="0.22.1"
readonly TORCHAUDIO_VERSION="2.7.1"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
guardian_requirements="${script_dir}/requirements/guardian-py310.txt"
openstereo_requirements="${script_dir}/requirements/openstereo-py310.txt"
guardian_env="${HOME}/.venvs/guardian-phase02b"
openstereo_env="${HOME}/.venvs/openstereo-phase02b"
openstereo_root="${HOME}/benchmarks/OpenStereo"
state_root="${HOME}/.local/state/guardian-phase02b"
apply=0
skip_system=0

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --guardian-env PATH    Guardian virtual environment." \
        "  --openstereo-env PATH  OpenStereo virtual environment." \
        "  --openstereo-root PATH External OpenStereo checkout." \
        "  --state-root PATH      Resolved-version manifests." \
        "  --skip-system          Skip apt/CUDA toolkit installation." \
        "  --apply                Perform changes; default is dry run." \
        "  -h, --help             Show this help."
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
        --state-root)
            state_root="$2"
            shift 2
            ;;
        --skip-system)
            skip_system=1
            shift
            ;;
        --apply)
            apply=1
            shift
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

for required_file in "$guardian_requirements" "$openstereo_requirements"; do
    [[ -f "$required_file" ]] || {
        echo "Missing setup input: $required_file" >&2
        exit 1
    }
done

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

if ((apply == 0)); then
    echo "Dry run only; no apt package, virtual environment, or checkout will change."
    if ((skip_system == 0)); then
        print_command sudo apt-get update
        print_command sudo apt-get install -y --no-install-recommends \
            build-essential ca-certificates cmake curl git gnupg jq \
            ninja-build pkg-config python3.10 python3.10-venv rsync
        print_command sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
        print_command sudo apt-get install -y --no-install-recommends \
            "$CUDA_TOOLKIT_PACKAGE"
    fi
    print_command python3.10 -m venv "$guardian_env"
    print_command "$guardian_env/bin/pip" install -r "$guardian_requirements"
    print_command python3.10 -m venv "$openstereo_env"
    print_command "$openstereo_env/bin/pip" install \
        --index-url "$PYTORCH_INDEX" \
        "torch==$PYTORCH_VERSION" \
        "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION"
    print_command "$openstereo_env/bin/pip" install \
        --extra-index-url "$NVIDIA_PYPI_INDEX" \
        -r "$openstereo_requirements"
    print_command bash "${script_dir}/clone_openstereo.sh" \
        --root "$openstereo_root" --apply
    echo "Resolved package versions would be written below $state_root."
    exit 0
fi

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "This bootstrap must run inside WSL2." >&2
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "Ubuntu 22.04 is required; detected ${PRETTY_NAME:-unknown}." >&2
    exit 1
fi

home_fs="$(findmnt -n -o FSTYPE -T "$HOME")"
case "$home_fs" in
    9p|drvfs|fuseblk)
        echo "Linux HOME is on $home_fs, not the distro ext4 filesystem." >&2
        exit 1
        ;;
esac

if ((skip_system == 0)); then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gnupg \
        jq \
        ninja-build \
        pkg-config \
        python3.10 \
        python3.10-venv \
        rsync

    keyring_tmp="$(mktemp --suffix=.deb)"
    cleanup_keyring() {
        rm -f -- "$keyring_tmp"
    }
    trap cleanup_keyring EXIT
    curl --fail --location --silent --show-error \
        "$CUDA_KEYRING_URL" \
        --output "$keyring_tmp"
    echo "${CUDA_KEYRING_SHA256}  ${keyring_tmp}" | sha256sum --check -
    sudo dpkg -i "$keyring_tmp"
    sudo apt-get update

    if ! apt-cache madison cuda-toolkit-12-8 |
        awk '{print $3}' |
        grep -Fxq '12.8.1-1'; then
        echo "Pinned CUDA toolkit 12.8.1-1 is unavailable from the NVIDIA repository." >&2
        exit 1
    fi
    sudo apt-get install -y --no-install-recommends "$CUDA_TOOLKIT_PACKAGE"
fi

if dpkg-query -W -f='${binary:Package}\n' 2>/dev/null |
    grep -Eq '^nvidia-driver(-|$)'; then
    echo "A Linux NVIDIA display-driver package is installed. Stop and remove it manually." >&2
    exit 1
fi

mkdir -p "$(dirname "$guardian_env")" "$(dirname "$openstereo_env")" "$state_root"

python3.10 -m venv "$guardian_env"
"$guardian_env/bin/python" -m pip install \
    --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
"$guardian_env/bin/python" -m pip install -r "$guardian_requirements"

python3.10 -m venv "$openstereo_env"
"$openstereo_env/bin/python" -m pip install \
    --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
"$openstereo_env/bin/python" -m pip install \
    --index-url "$PYTORCH_INDEX" \
    "torch==$PYTORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"
"$openstereo_env/bin/python" -m pip install \
    --extra-index-url "$NVIDIA_PYPI_INDEX" \
    -r "$openstereo_requirements"

bash "${script_dir}/clone_openstereo.sh" \
    --root "$openstereo_root" \
    --apply

LC_ALL=C "$guardian_env/bin/python" -m pip freeze --all |
    sort >"${state_root}/guardian-pip-freeze.txt"
LC_ALL=C "$openstereo_env/bin/python" -m pip freeze --all |
    sort >"${state_root}/openstereo-pip-freeze.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\n' |
    LC_ALL=C sort >"${state_root}/ubuntu-packages.txt"
{
    printf 'url=%s\n' "https://github.com/XiandaGuo/OpenStereo.git"
    printf 'commit=%s\n' "$(git -C "$openstereo_root" rev-parse HEAD)"
    printf 'checked_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${state_root}/openstereo-source.txt"

echo "Phase 2B Python environments and OpenStereo source are ready."
echo "Resolved manifests: $state_root"
echo "TensorRT Python is installed. Use install_tensorrt.sh for matching trtexec."
