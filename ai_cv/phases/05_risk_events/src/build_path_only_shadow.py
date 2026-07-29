"""Build direct ego-path lateral-offset telemetry; never changes TTC/risk."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from path_relative_state import (
    camera_measurement_to_planar,
    host_lateral_displacement_m,
    yaw_rate_rps,
)

FIELDS = [
    "trip_id", "frame_id", "track_id", "depth_m", "object_lateral_m",
    "ego_path_lateral_m", "path_offset_m", "yaw_rate_rps", "available",
]


def read_evidence(evidence_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Read only the frozen Phase 17 per-frame evidence used for review."""
    evidence: dict[tuple[str, int], dict[str, str]] = {}
    for path in sorted(evidence_root.glob("T*-Sample.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                evidence[(path.stem, int(row["frame_id"]))] = row
    if not evidence:
        raise FileNotFoundError(f"No T*-Sample.csv files in {evidence_root}")
    return evidence


def empty_result(label: dict[str, str], track_id: int) -> dict[str, str]:
    return {
        "trip_id": label["trip_id"], "frame_id": label["frame_id"],
        "track_id": str(track_id), "depth_m": "", "object_lateral_m": "",
        "ego_path_lateral_m": "", "path_offset_m": "", "yaw_rate_rps": "",
        "available": "false",
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--focal-px", type=float, default=320.0)
    parser.add_argument("--principal-x-px", type=float, default=320.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = read_evidence(args.evidence_root)
    rows: list[dict[str, str]] = []
    with args.labels.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    for label in labels:
        key = (label["trip_id"], int(label["frame_id"]))
        row = evidence.get(key)
        if row is None:
            raise KeyError(f"Missing frozen evidence row: {key}")
        track_id = int(label["track_id"])
        update = next(
            (
                candidate
                for candidate in json.loads(row.get("v2_shadow_updates_json", "[]"))
                if int(candidate.get("track_id", -1)) == track_id
                and candidate.get("measurement_source") == "yolo_box_median_disparity"
            ),
            None,
        )
        result = empty_result(label, track_id)
        if update:
            box = update["bbox_xyxy"]
            depth = float(update["depth_m"])
            centre = (float(box[0]) + float(box[2])) / 2.0
            measurement = camera_measurement_to_planar(
                depth_m=depth, center_x_px=centre, focal_length_px=args.focal_px,
                principal_x_px=args.principal_x_px,
            )
            speed = float(row["ego_speed_mps"])
            yaw = yaw_rate_rps(speed, float(row["lateral_accel_mps2"]))
            if measurement and speed>1 and yaw is not None:
                horizon = measurement[0] / speed
                host = host_lateral_displacement_m(speed, yaw, horizon)
                result.update({
                    "depth_m": f"{depth:.6f}",
                    "object_lateral_m": f"{measurement[1]:.6f}",
                    "ego_path_lateral_m": f"{host:.6f}",
                    "path_offset_m": f"{measurement[1] - host:.6f}",
                    "yaw_rate_rps": f"{yaw:.8f}", "available": "true",
                })
        rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "available": sum(r["available"] == "true" for r in rows)}))


if __name__ == "__main__":
    main()
