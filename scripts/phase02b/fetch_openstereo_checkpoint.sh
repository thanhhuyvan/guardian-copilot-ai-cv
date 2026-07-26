#!/usr/bin/env bash
set -Eeuo pipefail

readonly CHECKPOINT_URL="https://huggingface.co/XiandaGuo/OpenStereo/resolve/main/checkpoint/LightStereo/LightStereo-S-KITTI.ckpt"
readonly CHECKPOINT_SHA256="3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a"
readonly CHECKPOINT_SIZE_BYTES="14159749"

destination="${HOME}/benchmarks/OpenStereo-assets/checkpoints/LightStereo-S-KITTI.ckpt"
apply=0

usage() {
    printf '%s\n' \
        "Usage: $0 [--destination PATH] [--apply]" \
        "" \
        "Fetch the checksum-locked LightStereo-S KITTI checkpoint into an" \
        "external WSL workspace. Without --apply, print a dry run."
}

while (($#)); do
    case "$1" in
        --destination)
            [[ $# -ge 2 ]] || {
                echo "--destination requires a path" >&2
                exit 2
            }
            destination="$2"
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

verify_checkpoint() {
    local path="$1"
    local actual_size actual_hash
    actual_size="$(stat --format='%s' "$path")"
    actual_hash="$(sha256sum "$path" | awk '{print $1}')"
    [[ "$actual_size" == "$CHECKPOINT_SIZE_BYTES" ]] || {
        echo "Checkpoint size mismatch: $actual_size bytes" >&2
        return 1
    }
    [[ "$actual_hash" == "$CHECKPOINT_SHA256" ]] || {
        echo "Checkpoint SHA-256 mismatch: $actual_hash" >&2
        return 1
    }
}

echo "Source: $CHECKPOINT_URL"
echo "Destination: $destination"
echo "Expected bytes: $CHECKPOINT_SIZE_BYTES"
echo "Expected SHA-256: $CHECKPOINT_SHA256"

if [[ -f "$destination" ]]; then
    if verify_checkpoint "$destination"; then
        echo "Checkpoint already exists and is verified."
        exit 0
    fi
    echo "Refusing to overwrite an existing mismatched checkpoint." >&2
    exit 1
fi

if ((apply == 0)); then
    echo "Dry run only; no checkpoint will be downloaded."
    printf '+ curl --fail --location %q --output %q\n' \
        "$CHECKPOINT_URL" \
        "$destination"
    exit 0
fi

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "The checkpoint must be stored in the external WSL workspace." >&2
    exit 1
fi

destination_parent="$(dirname "$destination")"
filesystem_probe="$destination_parent"
while [[ ! -e "$filesystem_probe" ]]; do
    next_probe="$(dirname "$filesystem_probe")"
    [[ "$next_probe" != "$filesystem_probe" ]] || break
    filesystem_probe="$next_probe"
done
destination_fs="$(findmnt -n -o FSTYPE -T "$filesystem_probe")"
case "$destination_fs" in
    9p|drvfs|fuseblk)
        echo "Refusing to store the checkpoint on $destination_fs." >&2
        exit 1
        ;;
esac
mkdir -p "$destination_parent"

download_tmp="$(mktemp --tmpdir="$destination_parent" .LightStereo-S-KITTI.XXXXXX)"
cleanup() {
    rm -f -- "$download_tmp"
}
trap cleanup EXIT

curl \
    --fail \
    --location \
    --retry 3 \
    --retry-all-errors \
    --show-error \
    "$CHECKPOINT_URL" \
    --output "$download_tmp"
verify_checkpoint "$download_tmp"
mv "$download_tmp" "$destination"
trap - EXIT

{
    printf 'source_url=%s\n' "$CHECKPOINT_URL"
    printf 'source_path=%s\n' \
        'checkpoint/LightStereo/LightStereo-S-KITTI.ckpt'
    printf 'size_bytes=%s\n' "$CHECKPOINT_SIZE_BYTES"
    printf 'sha256=%s\n' "$CHECKPOINT_SHA256"
    printf 'downloaded_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${destination}.provenance.txt"

echo "Checkpoint downloaded and verified: $destination"
echo "It remains external to Guardian and OpenStereo source control."
