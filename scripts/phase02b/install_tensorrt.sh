#!/usr/bin/env bash
set -Eeuo pipefail

readonly TENSORRT_DEBIAN_VERSION="10.8.0.43-1+cuda12.8"
readonly EXPECTED_REPO_PATTERN='nv-tensorrt-local-repo-ubuntu2204-10\.8\.0-cuda-12\.8_1\.0-1_amd64\.deb'

repo_deb=""
state_root="${HOME}/.local/state/guardian-phase02b"
apply=0

usage() {
    printf '%s\n' \
        "Usage: $0 --repo-deb PATH [--state-root PATH] [--apply]" \
        "" \
        "Install the NVIDIA TensorRT 10.8 Ubuntu 22.04 local repository and" \
        "matching trtexec. Download the repository .deb from NVIDIA to D: first." \
        "Without --apply, this command is a non-mutating validation/dry run."
}

while (($#)); do
    case "$1" in
        --repo-deb)
            repo_deb="$2"
            shift 2
            ;;
        --state-root)
            state_root="$2"
            shift 2
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

[[ -n "$repo_deb" ]] || {
    echo "--repo-deb is required." >&2
    usage >&2
    exit 2
}
[[ -f "$repo_deb" ]] || {
    echo "TensorRT repository package not found: $repo_deb" >&2
    exit 1
}
[[ "$(basename "$repo_deb")" =~ ^${EXPECTED_REPO_PATTERN}$ ]] || {
    echo "Unexpected TensorRT repository filename: $(basename "$repo_deb")" >&2
    exit 1
}

repo_hash="$(sha256sum "$repo_deb" | awk '{print $1}')"
repo_package="$(dpkg-deb --field "$repo_deb" Package)"
case "$repo_package" in
    nv-tensorrt-local-repo-ubuntu2204-10.8.0-cuda-12.8)
        ;;
    *)
        echo "Unexpected Debian package identity: $repo_package" >&2
        exit 1
        ;;
esac

echo "TensorRT repository package: $repo_deb"
echo "SHA-256: $repo_hash"
echo "Pinned Debian version: $TENSORRT_DEBIAN_VERSION"

if ((apply == 0)); then
    echo "Dry run only; no repository or apt package was changed."
    printf '+ sudo dpkg -i %q\n' "$repo_deb"
    printf '+ sudo apt-get install -y --no-install-recommends %q\n' \
        "tensorrt=$TENSORRT_DEBIAN_VERSION"
    exit 0
fi

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "TensorRT must be installed inside WSL2." >&2
    exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "This repository package requires Ubuntu 22.04." >&2
    exit 1
fi

sudo dpkg -i "$repo_deb"
repo_keyring="$(find /var/nv-tensorrt-local-repo-ubuntu2204-10.8.0-cuda-12.8 \
    -maxdepth 1 -type f -name '*-keyring.gpg' -print -quit)"
[[ -n "$repo_keyring" ]] || {
    echo "TensorRT repository keyring was not installed." >&2
    exit 1
}
sudo install -m 0644 "$repo_keyring" /usr/share/keyrings/
sudo apt-get update

if ! apt-cache madison tensorrt |
    awk '{print $3}' |
    grep -Fxq "$TENSORRT_DEBIAN_VERSION"; then
    echo "Pinned TensorRT package is unavailable in the supplied local repository." >&2
    exit 1
fi

sudo apt-get install -y --no-install-recommends \
    "tensorrt=$TENSORRT_DEBIAN_VERSION"

if dpkg-query -W -f='${binary:Package}\n' 2>/dev/null |
    grep -Eq '^nvidia-driver(-|$)'; then
    echo "A Linux NVIDIA display driver was installed unexpectedly. Stop here." >&2
    exit 1
fi

trtexec_path="$(command -v trtexec || true)"
if [[ -z "$trtexec_path" && -x /usr/src/tensorrt/bin/trtexec ]]; then
    trtexec_path="/usr/src/tensorrt/bin/trtexec"
fi
[[ -n "$trtexec_path" ]] || {
    echo "TensorRT installed but trtexec was not found." >&2
    exit 1
}
"$trtexec_path" --version

mkdir -p "$state_root"
{
    printf 'source_file=%s\n' "$repo_deb"
    printf 'source_sha256=%s\n' "$repo_hash"
    printf 'debian_version=%s\n' "$TENSORRT_DEBIAN_VERSION"
    printf 'trtexec=%s\n' "$trtexec_path"
    printf 'installed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${state_root}/tensorrt-source.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\n' \
    'tensorrt*' 'libnvinfer*' 2>/dev/null |
    LC_ALL=C sort >"${state_root}/tensorrt-packages.txt"

echo "TensorRT and trtexec are ready."
