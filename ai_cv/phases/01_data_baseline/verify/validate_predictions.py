"""Strict Challenge 1 prediction validation.

The organizer evaluator is intentionally permissive: malformed rows can become
``inf`` and duplicate frame IDs overwrite earlier rows.  This validator fails
closed before evaluation so a good-looking score cannot come from an incomplete
CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ("frame_id", "timestamp", "predicted_ttc")


@dataclass(frozen=True)
class ValidationSummary:
    trip_id: str
    rows: int
    finite_predictions: int
    infinite_predictions: int


def _load_source_frames(trip_dir: Path) -> list[dict]:
    metadata_path = trip_dir / f"{trip_dir.name}.json.gz"
    if not metadata_path.is_file():
        matches = list(trip_dir.glob("*.json.gz"))
        if len(matches) != 1:
            raise ValueError(
                f"{trip_dir}: expected exactly one trip JSON, found {len(matches)}"
            )
        metadata_path = matches[0]

    with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{metadata_path}: missing non-empty frames list")
    return frames


def _parse_ttc(raw: str | None, row_number: int) -> float:
    if raw is None:
        raise ValueError(f"row {row_number}: missing predicted_ttc")
    value = raw.strip().lower()
    if value in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if value in {"", "nan", "-inf", "-infinity"}:
        raise ValueError(f"row {row_number}: invalid predicted_ttc {raw!r}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: predicted_ttc is not numeric: {raw!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"row {row_number}: invalid predicted_ttc {raw!r}")
    return parsed


def validate_prediction_file(
    csv_path: Path,
    trip_dir: Path,
    *,
    timestamp_tolerance: float = 1e-6,
    allow_extra_columns: bool = False,
) -> ValidationSummary:
    source_frames = _load_source_frames(trip_dir)
    expected_ids = [int(frame["frame_id"]) for frame in source_frames]
    expected_timestamps = [float(frame["timestamp"]) for frame in source_frames]

    if csv_path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise ValueError(
            f"{csv_path}: UTF-8 BOM is not accepted because the organizer "
            "evaluator misreads the frame_id header"
        )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if len(set(columns)) != len(columns):
            raise ValueError(f"{csv_path}: duplicate CSV header detected")
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        extras = [column for column in columns if column not in REQUIRED_COLUMNS]
        if missing:
            raise ValueError(f"{csv_path}: missing columns: {', '.join(missing)}")
        if extras and not allow_extra_columns:
            raise ValueError(
                f"{csv_path}: unexpected columns: {', '.join(extras)}; "
                "submission CSV must contain Challenge 1 columns only"
            )
        rows = list(reader)

    if len(rows) != len(source_frames):
        raise ValueError(
            f"{csv_path}: expected {len(source_frames)} rows, found {len(rows)}"
        )

    finite_count = 0
    actual_ids: list[int] = []
    for index, (row, expected_id, expected_timestamp) in enumerate(
        zip(rows, expected_ids, expected_timestamps), start=2
    ):
        try:
            frame_id = int(row["frame_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {index}: invalid frame_id {row['frame_id']!r}") from exc
        actual_ids.append(frame_id)
        if frame_id != expected_id:
            raise ValueError(
                f"row {index}: frame_id {frame_id}, expected {expected_id}; "
                "IDs must be complete, unique, and source-ordered"
            )

        try:
            timestamp = float(row["timestamp"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"row {index}: invalid timestamp {row['timestamp']!r}"
            ) from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"row {index}: timestamp must be finite")
        if abs(timestamp - expected_timestamp) > timestamp_tolerance:
            raise ValueError(
                f"row {index}: timestamp {timestamp}, expected "
                f"{expected_timestamp} ± {timestamp_tolerance}"
            )

        if math.isfinite(_parse_ttc(row["predicted_ttc"], index)):
            finite_count += 1

    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError(f"{csv_path}: duplicate frame_id detected")

    return ValidationSummary(
        trip_id=trip_dir.name,
        rows=len(rows),
        finite_predictions=finite_count,
        infinite_predictions=len(rows) - finite_count,
    )


def validate_prediction_set(
    predictions_root: Path,
    data_root: Path,
    *,
    timestamp_tolerance: float = 1e-6,
    allow_extra_columns: bool = False,
) -> list[ValidationSummary]:
    if not predictions_root.is_dir():
        raise ValueError(f"{predictions_root}: prediction directory not found")
    if not data_root.is_dir():
        raise ValueError(f"{data_root}: data directory not found")

    csv_paths = sorted(predictions_root.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"{predictions_root}: no prediction CSV files")

    expected_trip_ids = sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and any(path.glob("*.json.gz"))
    )
    if not expected_trip_ids:
        raise ValueError(f"{data_root}: no trip directories with JSON metadata")
    prediction_trip_ids = [path.stem for path in csv_paths]
    if prediction_trip_ids != expected_trip_ids:
        missing = sorted(set(expected_trip_ids) - set(prediction_trip_ids))
        extra = sorted(set(prediction_trip_ids) - set(expected_trip_ids))
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if extra:
            details.append(f"extra={','.join(extra)}")
        raise ValueError(
            f"{predictions_root}: prediction trip set does not match data root "
            f"({'; '.join(details)})"
        )

    summaries = []
    for csv_path in csv_paths:
        trip_dir = data_root / csv_path.stem
        if not trip_dir.is_dir():
            raise ValueError(f"{csv_path}: source trip not found at {trip_dir}")
        summaries.append(
            validate_prediction_file(
                csv_path,
                trip_dir,
                timestamp_tolerance=timestamp_tolerance,
                allow_extra_columns=allow_extra_columns,
            )
        )
    return summaries


def _format_summaries(summaries: Iterable[ValidationSummary]) -> str:
    return "\n".join(
        f"PASS {item.trip_id}: rows={item.rows}, finite={item.finite_predictions}, "
        f"inf={item.infinite_predictions}"
        for item in summaries
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--timestamp-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--allow-extra-columns",
        action="store_true",
        help="Permit local diagnostic columns. Never use for final submission.",
    )
    args = parser.parse_args()

    try:
        summaries = validate_prediction_set(
            args.predictions_root,
            args.data_root,
            timestamp_tolerance=args.timestamp_tolerance,
            allow_extra_columns=args.allow_extra_columns,
        )
    except ValueError as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(_format_summaries(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
