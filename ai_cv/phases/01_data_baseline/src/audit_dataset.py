"""Audit the local GuardianCoPilot practice and scored datasets.

The script is read-only with respect to dataset roots. It writes only compact CSV,
JSON and Markdown evidence to the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PRACTICE_IDS = [f"T{i:02d}-Sample" for i in range(1, 7)]
SCORED_IDS = [f"T{i:02d}d" for i in range(1, 11)]
DEPTH_SENTINEL_MIN_M = 999.0
EXPECTED_IMAGE_SIZE = (640, 360)

FIELD_SPECS = [
    ("frames[].ego.speed_kmh", "deployable_core", "Current/past telemetry"),
    ("frames[].ego.longitudinal_accel", "deployable_core", "Current/past telemetry"),
    ("frames[].ego.lateral_accel", "deployable_core", "Current/past telemetry"),
    ("frames[].targets[].target_id", "organizer_auxiliary", "Not deployment-realistic by default"),
    ("frames[].targets[].target_class", "organizer_auxiliary", "Not deployment-realistic by default"),
    ("events_log[].type", "organizer_auxiliary", "Causal mode may consume only after event time"),
    ("events_log[].t", "organizer_auxiliary", "Causal mode may consume only after event time"),
    ("events_log[].params", "practice_ground_truth", "Redacted from scored trips"),
    ("frames[].targets[].rel_pos", "practice_ground_truth", "Never predictor input"),
    ("frames[].targets[].rel_velocity", "practice_ground_truth", "Never predictor input"),
    ("frames[].targets[].longitudinal_distance", "practice_ground_truth", "Never predictor input"),
    ("frames[].targets[].lateral_distance", "practice_ground_truth", "Never predictor input"),
    ("frames[].targets[].closing_speed", "practice_ground_truth", "Never predictor input"),
    ("frames[].targets[].ttc_simple", "practice_ground_truth", "Never predictor input"),
    ("frames[].targets[].ttc_2d", "practice_ground_truth", "Never predictor input"),
    ("frames[].targets[].in_collision_cone", "practice_ground_truth", "Never predictor input"),
    ("frames[].min_ttc", "practice_ground_truth", "Evaluation only"),
    ("frames[].headway_sec", "practice_ground_truth", "Evaluation only"),
    ("frames[].driver.state", "out_of_core_ground_truth", "DMS outside Stage 01 TTC core"),
    ("frames[].behavior_flags.harsh_brake", "practice_ground_truth", "Evaluation only"),
    ("frames[].risk.final_risk_score", "practice_ground_truth", "Evaluation only"),
    ("trip_aggregate.safe_driving_score", "practice_ground_truth", "Evaluation only"),
    ("driver_summary.state_distribution_pct", "out_of_core_ground_truth", "DMS outside TTC core"),
]


def load_document(trip_dir: Path) -> dict[str, Any]:
    trip_id = trip_dir.name
    gz_path = trip_dir / f"{trip_id}.json.gz"
    json_path = trip_dir / f"{trip_id}.json"
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    if json_path.exists():
        with json_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(f"Missing {trip_id}.json(.gz) in {trip_dir}")


def parse_frame_id(path: Path, prefix: str = "") -> int | None:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return int(stem) if stem.isdigit() else None


def collect_ids(directory: Path, suffix: str, prefix: str = "") -> tuple[list[int], list[Path]]:
    if not directory.is_dir():
        return [], []
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix)
    ids = [frame_id for path in paths if (frame_id := parse_frame_id(path, prefix)) is not None]
    return ids, paths


def contiguous(ids: Iterable[int], expected_count: int) -> bool:
    values = list(ids)
    return values == list(range(expected_count))


def collapse_ranges(values: Iterable[int]) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ";".join(ranges)


def as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def count_episodes(values: list[float | None], threshold: float) -> int:
    active = False
    episodes = 0
    for value in values:
        now_active = value is not None and value < threshold
        if now_active and not active:
            episodes += 1
        active = now_active
    return episodes


def path_exists(document: Any, dotted_path: str) -> bool:
    nodes = [document]
    for token in dotted_path.split("."):
        is_array = token.endswith("[]")
        key = token[:-2] if is_array else token
        next_nodes: list[Any] = []
        for node in nodes:
            if not isinstance(node, dict) or key not in node:
                continue
            value = node[key]
            if is_array:
                if isinstance(value, list):
                    next_nodes.extend(value)
            else:
                next_nodes.append(value)
        nodes = next_nodes
        if not nodes:
            return False
    return True


def inspect_images(paths: list[Path], mode: str) -> dict[str, Any]:
    if not paths or mode == "none":
        return {"checked": 0, "invalid": 0, "sizes": "", "formats": ""}
    selected = paths
    if mode == "sample" and len(paths) > 3:
        selected = [paths[0], paths[len(paths) // 2], paths[-1]]
    sizes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    invalid = 0
    for path in selected:
        try:
            with Image.open(path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
                formats[str(image.format or path.suffix.lstrip(".")).upper()] += 1
                image.verify()
        except Exception:
            invalid += 1
    return {
        "checked": len(selected),
        "invalid": invalid,
        "sizes": ";".join(sorted(sizes)),
        "formats": ";".join(sorted(formats)),
    }


def inspect_depth(paths: list[Path]) -> dict[str, Any]:
    total_pixels = 0
    zero_pixels = 0
    sentinel_pixels = 0
    nonfinite_pixels = 0
    valid_pixels = 0
    all_zero_ids: list[int] = []
    all_sentinel_ids: list[int] = []
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    valid_medians: list[float] = []
    global_valid_min = math.inf
    global_valid_max = -math.inf

    for path in paths:
        frame_id = parse_frame_id(path)
        array = np.load(path, mmap_mode="r")
        shapes["x".join(str(dim) for dim in array.shape)] += 1
        dtypes[str(array.dtype)] += 1
        pixel_count = int(array.size)
        total_pixels += pixel_count
        zero_count = int(np.count_nonzero(array == 0))
        sentinel_count = int(np.count_nonzero(array >= DEPTH_SENTINEL_MIN_M))
        finite_mask = np.isfinite(array)
        nonfinite_count = pixel_count - int(np.count_nonzero(finite_mask))
        valid_mask = finite_mask & (array > 0) & (array < DEPTH_SENTINEL_MIN_M)
        valid_count = int(np.count_nonzero(valid_mask))

        zero_pixels += zero_count
        sentinel_pixels += sentinel_count
        nonfinite_pixels += nonfinite_count
        valid_pixels += valid_count
        if frame_id is not None and zero_count == pixel_count:
            all_zero_ids.append(frame_id)
        if frame_id is not None and sentinel_count == pixel_count:
            all_sentinel_ids.append(frame_id)
        if valid_count:
            valid_values = np.asarray(array[valid_mask])
            valid_medians.append(float(np.median(valid_values)))
            global_valid_min = min(global_valid_min, float(valid_values.min()))
            global_valid_max = max(global_valid_max, float(valid_values.max()))

    def ratio(count: int) -> float:
        return count / total_pixels if total_pixels else 0.0

    return {
        "depth_shape": ";".join(f"{key}:{value}" for key, value in sorted(shapes.items())),
        "depth_dtype": ";".join(f"{key}:{value}" for key, value in sorted(dtypes.items())),
        "depth_all_zero_count": len(all_zero_ids),
        "depth_all_zero_ranges": collapse_ranges(all_zero_ids),
        "depth_all_sentinel_count": len(all_sentinel_ids),
        "depth_all_sentinel_ranges": collapse_ranges(all_sentinel_ids),
        "depth_zero_ratio": round(ratio(zero_pixels), 8),
        "depth_sentinel_ratio": round(ratio(sentinel_pixels), 8),
        "depth_nonfinite_ratio": round(ratio(nonfinite_pixels), 8),
        "depth_valid_ratio": round(ratio(valid_pixels), 8),
        "depth_valid_median_of_keyframes_m": round(statistics.median(valid_medians), 4) if valid_medians else None,
        "depth_valid_min_m": round(global_valid_min, 4) if math.isfinite(global_valid_min) else None,
        "depth_valid_max_m": round(global_valid_max, 4) if math.isfinite(global_valid_max) else None,
    }


def audit_trip(trip_dir: Path, split: str, image_mode: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    trip_id = trip_dir.name
    document = load_document(trip_dir)
    metadata = document.get("metadata", {})
    frames = document.get("frames", [])
    frame_ids = [int(frame.get("frame_id", -1)) for frame in frames]
    timestamps = [float(frame.get("timestamp", math.nan)) for frame in frames]
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]

    modality_specs = {
        "left": (trip_dir / "kitti" / "image_2", ".jpg", ""),
        "right": (trip_dir / "kitti" / "image_3", ".jpg", ""),
        "driver": (trip_dir / "driver", ".jpg", "frame_"),
        "depth": (trip_dir / "kitti" / "depth", ".npy", ""),
        "calib": (trip_dir / "kitti" / "calib", ".txt", ""),
        "label": (trip_dir / "kitti" / "label_2", ".txt", ""),
    }
    modality: dict[str, tuple[list[int], list[Path]]] = {
        name: collect_ids(directory, suffix, prefix)
        for name, (directory, suffix, prefix) in modality_specs.items()
    }

    calibration_path = trip_dir / "kitti" / "calibration_info.txt"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    fx = calibration.get("K_left", [[None]])[0][0] if calibration.get("K_left") else None
    baseline_m = calibration.get("baseline_m")

    left_info = inspect_images(modality["left"][1], image_mode)
    right_info = inspect_images(modality["right"][1], image_mode)
    driver_info = inspect_images(modality["driver"][1], image_mode)
    depth_info = inspect_depth(modality["depth"][1])

    ttc_values = [as_finite_float(frame.get("min_ttc")) for frame in frames]
    target_classes: Counter[str] = Counter()
    for frame in frames:
        for target in frame.get("targets", []):
            target_classes[str(target.get("target_class", "unknown"))] += 1
    event_types = [str(event.get("type", "unknown")) for event in document.get("events_log", [])]

    observed_duration = (timestamps[-1] - timestamps[0] + statistics.median(deltas)) if timestamps and deltas else 0.0
    description_seconds = None
    match = re.search(r"\b(\d+)s\b", str(metadata.get("description", "")), flags=re.IGNORECASE)
    if match:
        description_seconds = int(match.group(1))

    row: dict[str, Any] = {
        "trip_id": trip_id,
        "split": split,
        "n_frames": len(frames),
        "frame_id_first": frame_ids[0] if frame_ids else None,
        "frame_id_last": frame_ids[-1] if frame_ids else None,
        "frame_ids_contiguous": contiguous(frame_ids, len(frames)),
        "timestamps_monotonic": all(delta > 0 for delta in deltas),
        "timestamp_step_median_sec": round(statistics.median(deltas), 6) if deltas else None,
        "timestamp_step_min_sec": round(min(deltas), 6) if deltas else None,
        "timestamp_step_max_sec": round(max(deltas), 6) if deltas else None,
        "duration_metadata_sec": metadata.get("duration_sec"),
        "duration_observed_sec": round(observed_duration, 3),
        "fps_metadata": metadata.get("fps"),
        "description": metadata.get("description", ""),
        "description_seconds": description_seconds,
        "left_count": len(modality["left"][0]),
        "right_count": len(modality["right"][0]),
        "driver_count": len(modality["driver"][0]),
        "depth_count": len(modality["depth"][0]),
        "calib_count": len(modality["calib"][0]),
        "label_count": len(modality["label"][0]),
        "left_ids_contiguous": contiguous(modality["left"][0], len(frames)),
        "right_ids_contiguous": contiguous(modality["right"][0], len(frames)),
        "driver_ids_contiguous": contiguous(modality["driver"][0], len(frames)),
        "calib_ids_contiguous": contiguous(modality["calib"][0], len(frames)),
        "label_ids_contiguous": contiguous(modality["label"][0], len(frames)),
        "image_size": left_info["sizes"],
        "image_format": left_info["formats"],
        "images_checked": left_info["checked"] + right_info["checked"] + driver_info["checked"],
        "invalid_images": left_info["invalid"] + right_info["invalid"] + driver_info["invalid"],
        "fx": fx,
        "baseline_m": baseline_m,
        "n_events": len(event_types),
        "event_types": ";".join(event_types),
        "target_classes": ";".join(f"{key}:{value}" for key, value in sorted(target_classes.items())),
        "finite_gt_count": sum(value is not None for value in ttc_values),
        "gt_lt_3_count": sum(value is not None and value < 3.0 for value in ttc_values),
        "gt_lt_2_count": sum(value is not None and value < 2.0 for value in ttc_values),
        "gt_lt_1_5_count": sum(value is not None and value < 1.5 for value in ttc_values),
        "gt_lt_3_episodes": count_episodes(ttc_values, 3.0),
        "gt_lt_2_episodes": count_episodes(ttc_values, 2.0),
        **depth_info,
    }

    anomalies: list[dict[str, str]] = []

    def add_anomaly(severity: str, category: str, detail: str) -> None:
        anomalies.append({"trip_id": trip_id, "severity": severity, "category": category, "detail": detail})

    expected_count_fields = ("left_count", "right_count", "driver_count", "calib_count", "label_count")
    for field in expected_count_fields:
        if row[field] != len(frames):
            add_anomaly("ERROR", "modality_count", f"{field}={row[field]} but n_frames={len(frames)}")
    if not row["frame_ids_contiguous"] or not row["timestamps_monotonic"]:
        add_anomaly("ERROR", "timeline", "Frame IDs or timestamps are not strictly contiguous/monotonic")
    if deltas and not all(math.isclose(delta, 0.05, abs_tol=1e-6) for delta in deltas):
        add_anomaly("WARN", "timestamp_step", f"Observed delta range {min(deltas):.6f}-{max(deltas):.6f}s")
    if row["invalid_images"]:
        add_anomaly("ERROR", "image_decode", f"{row['invalid_images']} image files failed verification")
    if row["image_size"] and row["image_size"] != f"{EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]}":
        add_anomaly("ERROR", "image_size", f"Unexpected left-image sizes: {row['image_size']}")
    if fx is None or not math.isclose(float(fx), 320.0, rel_tol=1e-6):
        add_anomaly("ERROR", "calibration", f"Unexpected fx={fx}")
    if baseline_m is None or not math.isclose(float(baseline_m), 0.3, rel_tol=1e-6):
        add_anomaly("ERROR", "calibration", f"Unexpected baseline_m={baseline_m}")
    if row["depth_all_zero_count"]:
        add_anomaly("WARN", "depth_all_zero", f"{row['depth_all_zero_count']} keyframes: {row['depth_all_zero_ranges']}")
    if description_seconds is not None and not math.isclose(description_seconds, observed_duration, abs_tol=1.0):
        add_anomaly(
            "WARN",
            "metadata_description",
            f"Description says {description_seconds}s but observed duration is {observed_duration:.1f}s",
        )
    if split == "scored" and row["finite_gt_count"]:
        add_anomaly("ERROR", "redaction", f"Scored trip exposes {row['finite_gt_count']} finite min_ttc values")

    field_presence = {path: path_exists(document, path) for path, _, _ in FIELD_SPECS}
    return row, field_presence, anomalies


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_dir: Path, rows: list[dict[str, Any]], anomalies: list[dict[str, str]]) -> None:
    practice = [row for row in rows if row["split"] == "practice"]
    scored = [row for row in rows if row["split"] == "scored"]
    errors = [item for item in anomalies if item["severity"] == "ERROR"]
    warnings = [item for item in anomalies if item["severity"] == "WARN"]

    lines = [
        "# Dataset Audit Report",
        "",
        "Generated by `src/audit_dataset.py`. Dataset files were read-only and are not committed.",
        "",
        "## Summary",
        "",
        f"- Trips audited: {len(rows)} ({len(practice)} practice, {len(scored)} scored).",
        f"- Frames audited: {sum(int(row['n_frames']) for row in rows):,}.",
        f"- Depth keyframes audited: {sum(int(row['depth_count']) for row in rows):,}.",
        f"- Image files verified: {sum(int(row['images_checked']) for row in rows):,}.",
        f"- Findings: {len(errors)} error(s), {len(warnings)} warning(s).",
        "",
        "## Trip inventory",
        "",
        "| Trip | Split | Frames | Left/Right | Depth | GT finite | <3s | <2s | Zero-depth keyframes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trip_id']} | {row['split']} | {row['n_frames']} | "
            f"{row['left_count']}/{row['right_count']} | {row['depth_count']} | "
            f"{row['finite_gt_count']} | {row['gt_lt_3_count']} | {row['gt_lt_2_count']} | "
            f"{row['depth_all_zero_count']} |"
        )

    lines.extend(["", "## Findings", ""])
    if anomalies:
        lines.extend(["| Severity | Trip | Category | Detail |", "|---|---|---|---|"])
        for finding in anomalies:
            detail = finding["detail"].replace("|", "\\|")
            lines.append(f"| {finding['severity']} | {finding['trip_id']} | {finding['category']} | {detail} |")
    else:
        lines.append("No anomalies detected.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `finite_gt_count=0` is expected for scored/redacted trips.",
            "- Depth values `>=999 m` are treated as sentinel/invalid for quality statistics.",
            "- All-zero depth keyframes are degraded inputs even though the `.npy` files exist.",
            "- Field presence does not imply deployment-realistic use; see `field_availability.csv`.",
            "- This audit does not yet prove stereo pixel alignment or baseline accuracy.",
            "",
            "## Gate decision",
            "",
            "S1.1 inventory/redaction audit passes only when there are no unexplained ERROR findings. "
            "WARN findings remain explicit degraded-mode requirements for later phases.",
        ]
    )
    (output_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(practice_root: Path, scored_root: Path, output_dir: Path, image_mode: str) -> int:
    rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, str]] = []
    presence_by_trip: dict[str, dict[str, bool]] = {}

    for split, root, trip_ids in (
        ("practice", practice_root, PRACTICE_IDS),
        ("scored", scored_root, SCORED_IDS),
    ):
        for trip_id in trip_ids:
            trip_dir = root / trip_id
            if not trip_dir.is_dir():
                anomalies.append(
                    {"trip_id": trip_id, "severity": "ERROR", "category": "missing_trip", "detail": str(trip_dir)}
                )
                continue
            print(f"Auditing {trip_id} ({split})...", flush=True)
            row, presence, trip_anomalies = audit_trip(trip_dir, split, image_mode)
            rows.append(row)
            presence_by_trip[trip_id] = presence
            anomalies.extend(trip_anomalies)

    rows.sort(key=lambda row: (0 if row["split"] == "practice" else 1, row["trip_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "dataset_inventory.csv", rows)

    depth_fields = [
        "trip_id",
        "split",
        "depth_count",
        "depth_shape",
        "depth_dtype",
        "depth_all_zero_count",
        "depth_all_zero_ranges",
        "depth_all_sentinel_count",
        "depth_all_sentinel_ranges",
        "depth_zero_ratio",
        "depth_sentinel_ratio",
        "depth_nonfinite_ratio",
        "depth_valid_ratio",
        "depth_valid_median_of_keyframes_m",
        "depth_valid_min_m",
        "depth_valid_max_m",
    ]
    write_csv(output_dir / "depth_keyframe_summary.csv", [{key: row.get(key) for key in depth_fields} for row in rows])

    gt_fields = [
        "trip_id",
        "n_frames",
        "finite_gt_count",
        "gt_lt_3_count",
        "gt_lt_2_count",
        "gt_lt_1_5_count",
        "gt_lt_3_episodes",
        "gt_lt_2_episodes",
        "event_types",
        "target_classes",
    ]
    practice_rows = [{key: row.get(key) for key in gt_fields} for row in rows if row["split"] == "practice"]
    totals = {key: "" for key in gt_fields}
    totals["trip_id"] = "TOTAL"
    for key in (
        "n_frames",
        "finite_gt_count",
        "gt_lt_3_count",
        "gt_lt_2_count",
        "gt_lt_1_5_count",
        "gt_lt_3_episodes",
        "gt_lt_2_episodes",
    ):
        totals[key] = sum(int(row[key]) for row in practice_rows)
    write_csv(output_dir / "practice_gt_distribution.csv", [*practice_rows, totals], gt_fields)

    field_rows: list[dict[str, Any]] = []
    for path, use_class, note in FIELD_SPECS:
        practice_present = sum(presence_by_trip.get(trip_id, {}).get(path, False) for trip_id in PRACTICE_IDS)
        scored_present = sum(presence_by_trip.get(trip_id, {}).get(path, False) for trip_id in SCORED_IDS)
        field_rows.append(
            {
                "field": path,
                "practice_trips_present": f"{practice_present}/{len(PRACTICE_IDS)}",
                "scored_trips_present": f"{scored_present}/{len(SCORED_IDS)}",
                "use_class": use_class,
                "note": note,
            }
        )
    write_csv(output_dir / "field_availability.csv", field_rows)

    report = {
        "schema_version": "dataset_audit.v1",
        # Keep reports portable and avoid committing developer-specific paths.
        "practice_root": str(practice_root),
        "scored_root": str(scored_root),
        "image_verification_mode": image_mode,
        "summary": {
            "trips_audited": len(rows),
            "frames_audited": sum(int(row["n_frames"]) for row in rows),
            "depth_keyframes_audited": sum(int(row["depth_count"]) for row in rows),
            "images_verified": sum(int(row["images_checked"]) for row in rows),
            "error_count": sum(item["severity"] == "ERROR" for item in anomalies),
            "warning_count": sum(item["severity"] == "WARN" for item in anomalies),
        },
        "trips": rows,
        "anomalies": anomalies,
    }
    (output_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(output_dir, rows, anomalies)

    print(f"Audit artifacts written to {output_dir}")
    print(json.dumps(report["summary"], indent=2))
    return 1 if report["summary"]["error_count"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit GuardianCoPilot practice and scored datasets.")
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-images", choices=("none", "sample", "all"), default="sample")
    args = parser.parse_args()
    return run_audit(args.practice_root, args.scored_root, args.output_dir, args.verify_images)


if __name__ == "__main__":
    sys.exit(main())
