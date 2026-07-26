#!/usr/bin/env bash
set -Eeuo pipefail

readonly OPENSTEREO_URL="https://github.com/XiandaGuo/OpenStereo.git"
readonly OPENSTEREO_COMMIT="23d71c92e33ad1f80dfc42bf29f5c6a914d38769"

target_root="${HOME}/benchmarks/OpenStereo"
apply=0

usage() {
    printf '%s\n' \
        "Usage: $0 [--root PATH] [--apply]" \
        "" \
        "Without --apply, prints a non-mutating dry run."
}

while (($#)); do
    case "$1" in
        --root)
            [[ $# -ge 2 ]] || { echo "--root requires a path" >&2; exit 2; }
            target_root="$2"
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

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

if ((apply == 0)); then
    echo "Dry run only; no repository will be cloned or changed."
    print_command git clone --filter=blob:none "$OPENSTEREO_URL" "$target_root"
    print_command git -C "$target_root" fetch origin "$OPENSTEREO_COMMIT"
    print_command git -C "$target_root" switch --detach "$OPENSTEREO_COMMIT"
    exit 0
fi

if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "OpenStereo must be cloned inside WSL2." >&2
    exit 1
fi

target_parent="$(dirname "$target_root")"
filesystem_probe="$target_parent"
while [[ ! -e "$filesystem_probe" ]]; do
    next_probe="$(dirname "$filesystem_probe")"
    [[ "$next_probe" != "$filesystem_probe" ]] || break
    filesystem_probe="$next_probe"
done
target_fs="$(findmnt -n -o FSTYPE -T "$filesystem_probe")"
case "$target_fs" in
    9p|drvfs|fuseblk)
        echo "Refusing to place OpenStereo on a Windows-mounted filesystem ($target_fs)." >&2
        exit 1
        ;;
esac
mkdir -p "$target_parent"

if [[ -e "$target_root" && ! -d "$target_root/.git" ]]; then
    echo "Target exists but is not a Git repository: $target_root" >&2
    exit 1
fi

if [[ ! -d "$target_root/.git" ]]; then
    git clone --filter=blob:none "$OPENSTEREO_URL" "$target_root"
else
    origin_url="$(git -C "$target_root" remote get-url origin)"
    normalized_origin="${origin_url%.git}"
    normalized_expected="${OPENSTEREO_URL%.git}"
    if [[ "$normalized_origin" != "$normalized_expected" ]]; then
        echo "Unexpected OpenStereo origin: $origin_url" >&2
        exit 1
    fi
    if [[ -n "$(git -C "$target_root" status --porcelain --untracked-files=no)" ]]; then
        echo "Tracked OpenStereo changes exist; refusing to change commits." >&2
        exit 1
    fi
fi

git -C "$target_root" fetch origin "$OPENSTEREO_COMMIT"
git -C "$target_root" switch --detach "$OPENSTEREO_COMMIT"

actual_commit="$(git -C "$target_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$OPENSTEREO_COMMIT" ]]; then
    echo "OpenStereo commit mismatch: $actual_commit" >&2
    exit 1
fi

echo "OpenStereo source ready: $target_root"
echo "Pinned commit: $actual_commit"
echo "License boundary: external academic/non-commercial benchmark only."
