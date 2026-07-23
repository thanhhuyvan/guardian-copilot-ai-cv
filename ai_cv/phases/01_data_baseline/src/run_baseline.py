"""Run, normalize, strictly validate, and evaluate the organizer TTC baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "verify"))
from validate_predictions import validate_prediction_set  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_challenge1_csv(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_id", "timestamp", "predicted_ttc"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice-root", type=Path, default=Path("Practice_Dataset"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ai_cv/outputs/predictions/baseline_official"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("ai_cv/outputs/reports/baseline_official/evaluation.json"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("ai_cv/outputs/reports/baseline_official/run_manifest.json"),
    )
    parser.add_argument(
        "--team-kit-dir",
        type=Path,
        default=Path("Package_starterkit/package_starterkit/team_kit"),
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Skip SGBM and validate/evaluate existing normalized CSV files.",
    )
    args = parser.parse_args()

    predictor = args.team_kit_dir / "baseline_ttc_predictor.py"
    loader = args.team_kit_dir / "dataset_loader.py"
    evaluator = args.team_kit_dir / "evaluation.py"
    required_paths = [args.practice_root, predictor, loader, evaluator]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        parser.error(f"missing required path(s): {', '.join(missing)}")

    trips = sorted(
        path for path in args.practice_root.iterdir()
        if path.is_dir() and path.name.endswith("-Sample")
    )
    if not trips:
        parser.error(f"no *-Sample trips found in {args.practice_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    trip_runtime: dict[str, float] = {}
    started = time.perf_counter()

    if not args.reuse_predictions:
        with tempfile.TemporaryDirectory(prefix="phase01-baseline-") as temp:
            temp_root = Path(temp)
            for trip in trips:
                raw_output = temp_root / f"{trip.name}.csv"
                trip_started = time.perf_counter()
                run_command(
                    [
                        sys.executable,
                        str(predictor),
                        "--trip-dir",
                        str(trip),
                        "--output",
                        str(raw_output),
                    ]
                )
                trip_runtime[trip.name] = time.perf_counter() - trip_started
                normalize_challenge1_csv(
                    raw_output, args.output_root / f"{trip.name}.csv"
                )

    summaries = validate_prediction_set(args.output_root, args.practice_root)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            str(evaluator),
            "--predictions",
            str(args.output_root),
            "--data-dir",
            str(args.practice_root),
            "--output",
            str(args.report_output),
        ]
    )

    try:
        import cv2
        import numpy as np
        import pandas as pd
    except ImportError:
        cv2 = np = pd = None

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "reuse_predictions" if args.reuse_predictions else "full_recompute",
        "command": "python ai_cv/phases/01_data_baseline/src/run_baseline.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dependencies": {
            "opencv": getattr(cv2, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "pandas": getattr(pd, "__version__", None),
        },
        "source_sha256": {
            "baseline_ttc_predictor.py": sha256(predictor),
            "dataset_loader.py": sha256(loader),
            "evaluation.py": sha256(evaluator),
        },
        "trips": [
            {
                "trip_id": item.trip_id,
                "rows": item.rows,
                "finite_predictions": item.finite_predictions,
                "infinite_predictions": item.infinite_predictions,
                "wall_time_sec": trip_runtime.get(item.trip_id),
            }
            for item in summaries
        ],
        "total_wall_time_sec": time.perf_counter() - started,
        "evaluation_report": str(args.report_output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"PASS: validated and evaluated {len(summaries)} trips; "
        f"report={args.report_output}; manifest={args.manifest_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
