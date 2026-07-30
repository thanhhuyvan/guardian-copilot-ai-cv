"""Validate and freeze Challenge-1 TTC CSVs from a causal Guardian run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path


TRIPS = tuple(f"T{index:02d}d" for index in range(1, 11))
FIELDS = ("frame_id", "timestamp", "predicted_ttc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ttc(value: str) -> float:
    text = value.strip().lower()
    if text in {"inf", "+inf", "infinity"}:
        return math.inf
    result = float(text)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"TTC must be a positive finite value or inf, got {value!r}")
    return result


def validate(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path.name}: expected columns {FIELDS}, got {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 1800:
        raise ValueError(f"{path.name}: expected 1800 frames, got {len(rows)}")
    dangers = 0
    prior_time = -math.inf
    for expected_id, row in enumerate(rows):
        if int(row["frame_id"]) != expected_id:
            raise ValueError(f"{path.name}: frame sequence is not 0..1799")
        timestamp = float(row["timestamp"])
        if not math.isfinite(timestamp) or timestamp <= prior_time:
            raise ValueError(f"{path.name}: timestamps are not strictly increasing")
        prior_time = timestamp
        dangers += int(parse_ttc(row["predicted_ttc"]) < 2.0)
    return {"rows": len(rows), "danger_frames": dangers, "sha256": sha256(path)}


def git_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    predictions = output / "predictions" / "guardian_v1"
    predictions.mkdir(parents=True, exist_ok=True)
    files: dict[str, object] = {}
    for trip_id in TRIPS:
        input_path = source / f"{trip_id}.csv"
        if not input_path.is_file():
            raise FileNotFoundError(f"missing {input_path}")
        files[trip_id] = validate(input_path)
        shutil.copy2(input_path, predictions / input_path.name)
    extras = sorted(path.name for path in source.glob("*.csv") if path.stem not in TRIPS)
    if extras:
        raise ValueError(f"unexpected submission CSVs: {extras}")
    manifest = {
        "schema": "guardian.challenge1-submission-manifest.v1",
        "selected_policy": "conservative_union",
        "source_directory": str(source),
        "repository_commit": git_commit(args.repository.resolve()),
        "trips": files,
        "validation": "10 files; 1800 sequential frames/trip; monotonic timestamps; positive finite or inf TTC",
        "ground_truth": "redacted; this manifest is format/provenance validation, not an F1 result",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
