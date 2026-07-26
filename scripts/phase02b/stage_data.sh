#!/usr/bin/env bash
set -Eeuo pipefail

source_root="/mnt/d/Python"
destination_root="${HOME}/guardian-data/phase02b"
state_root="${HOME}/.local/state/guardian-phase02b/data"
apply=0
hash_files=1

readonly DATA_DIRECTORIES=(
    "Practice_Dataset"
    "Package_starterkit"
)

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --source-root PATH       Root containing Practice_Dataset and Package_starterkit." \
        "  --destination-root PATH  WSL ext4 destination." \
        "  --state-root PATH        Manifest destination." \
        "  --skip-hash              Skip SHA-256 manifests." \
        "  --apply                  Copy/update files; default is dry run." \
        "  -h, --help               Show this help."
}

while (($#)); do
    case "$1" in
        --source-root)
            source_root="$2"
            shift 2
            ;;
        --destination-root)
            destination_root="$2"
            shift 2
            ;;
        --state-root)
            state_root="$2"
            shift 2
            ;;
        --skip-hash)
            hash_files=0
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

for directory in "${DATA_DIRECTORIES[@]}"; do
    [[ -d "${source_root}/${directory}" ]] || {
        echo "Missing source directory: ${source_root}/${directory}" >&2
        exit 1
    }
done

echo "Source: $source_root"
echo "WSL ext4 destination: $destination_root"
echo "Directories: ${DATA_DIRECTORIES[*]}"
echo "Sync policy: additive/update only; stale destination files are never deleted."

if ((apply == 0)); then
    echo "Dry run only; no data will be copied."
    for directory in "${DATA_DIRECTORIES[@]}"; do
        printf '+ rsync --archive --no-owner --no-group %q/ %q/\n' \
            "${source_root}/${directory}" \
            "${destination_root}/${directory}"
    done
    exit 0
fi

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "Data staging must run inside WSL2." >&2
    exit 1
fi

filesystem_probe="$destination_root"
while [[ ! -e "$filesystem_probe" ]]; do
    next_probe="$(dirname "$filesystem_probe")"
    [[ "$next_probe" != "$filesystem_probe" ]] || break
    filesystem_probe="$next_probe"
done
destination_fs="$(findmnt -n -o FSTYPE -T "$filesystem_probe")"
case "$destination_fs" in
    9p|drvfs|fuseblk)
        echo "Refusing destination filesystem $destination_fs; use WSL ext4." >&2
        exit 1
        ;;
esac
mkdir -p "$destination_root" "$state_root"

required_bytes=0
for directory in "${DATA_DIRECTORIES[@]}"; do
    directory_bytes="$(du -sb "${source_root}/${directory}" | awk '{print $1}')"
    required_bytes=$((required_bytes + directory_bytes))
done
available_bytes="$(df -B1 --output=avail "$destination_root" | tail -n 1 | tr -d ' ')"
reserve_bytes=$((2 * 1024 * 1024 * 1024))
if ((available_bytes < required_bytes + reserve_bytes)); then
    echo "Insufficient ext4 space: need source size plus a 2 GiB reserve." >&2
    exit 1
fi

for directory in "${DATA_DIRECTORIES[@]}"; do
    source_directory="${source_root}/${directory}"
    destination_directory="${destination_root}/${directory}"
    mkdir -p "$destination_directory"
    rsync \
        --archive \
        --no-owner \
        --no-group \
        --human-readable \
        --info=stats2 \
        "${source_directory}/" \
        "${destination_directory}/"

    pending="$(
        rsync \
            --archive \
            --no-owner \
            --no-group \
            --checksum \
            --dry-run \
            --itemize-changes \
            "${source_directory}/" \
            "${destination_directory}/"
    )"
    if [[ -n "$pending" ]]; then
        echo "Post-copy verification found mismatched source files in $directory:" >&2
        echo "$pending" >&2
        exit 1
    fi

    if ((hash_files == 1)); then
        manifest_tmp="${state_root}/${directory}.sha256.tmp"
        (
            cd "$destination_directory"
            find . -type f -print0 |
                LC_ALL=C sort -z |
                xargs -0 -r sha256sum
        ) >"$manifest_tmp"
        mv "$manifest_tmp" "${state_root}/${directory}.sha256"
    fi
done

{
    printf 'source_root=%s\n' "$source_root"
    printf 'destination_root=%s\n' "$destination_root"
    printf 'destination_filesystem=%s\n' "$destination_fs"
    printf 'staged_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hash_manifests=%s\n' "$hash_files"
} >"${state_root}/stage-source.txt"

echo "Practice data and starter kit are synchronized to WSL ext4."
echo "Manifests: $state_root"
