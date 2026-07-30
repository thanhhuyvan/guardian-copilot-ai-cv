"""Build blind short-clip labels for object-event correspondence validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PHASE02_SRC = REPOSITORY_ROOT / "ai_cv" / "phases" / "02_detection_tracking" / "src"
if str(PHASE02_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE02_SRC))

from evaluate_path_only_gate import load_truth, parse_ttc  # noqa: E402


FIELDS = [
    "trip_id", "frame_id", "left_image_path", "selected_track_id",
    "object_id_window", "event_owner", "path_relation", "relative_motion",
    "cpa_distance_m", "occluded", "review_confidence", "notes",
]


def spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    positions = {round(index * (len(rows) - 1) / (count - 1)) for index in range(count)}
    return [row for index, row in enumerate(rows) if index in positions]


def selected_box(row: dict[str, str]) -> tuple[int, int, int, int] | None:
    raw = row.get("union_selected_bbox_xyxy") or row.get("classical_selected_bbox_xyxy")
    if not raw:
        return None
    try:
        box = json.loads(raw)
        if len(box) != 4:
            return None
        return tuple(round(float(value)) for value in box)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def render_clip(
    *, practice_root: Path, trip_id: str, frame_id: int,
    target_box: tuple[int, int, int, int], radius: int, output: Path,
) -> None:
    writer = None
    contact_frames = []
    for current in range(max(0, frame_id - radius), frame_id + radius + 1):
        path = practice_root / trip_id / "kitti" / "image_2" / f"{current:06d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            continue
        if writer is None:
            height, width = image.shape[:2]
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"could not create {output}")
        if current == frame_id:
            x0, y0, x1, y1 = target_box
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(image, "review object", (x0, max(18, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"frame {current}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(image)
        if current in {max(0, frame_id - radius), frame_id, frame_id + radius}:
            contact_frames.append(image.copy())
    if writer is not None:
        writer.release()
    if contact_frames:
        while len(contact_frames) < 3:
            contact_frames.append(contact_frames[-1])
        sheet = cv2.hconcat(contact_frames[:3])
        contact_path = output.parent.parent.parent / "contact_sheets" / trip_id / f"{frame_id:06d}.jpg"
        contact_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(contact_path), sheet):
            raise RuntimeError(f"could not create {contact_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-outcome", type=int, default=4)
    parser.add_argument("--radius", type=int, default=5)
    args = parser.parse_args()
    if args.per_outcome < 1 or args.radius < 1:
        raise ValueError("per-outcome and radius must be positive")

    selected: list[dict[str, str]] = []
    for evidence_path in sorted(args.evidence_root.glob("T*-Sample.csv")):
        with evidence_path.open(encoding="utf-8", newline="") as handle:
            evidence = list(csv.DictReader(handle))
        for row in evidence:
            row["_trip_id"] = evidence_path.stem
        truth = load_truth(args.practice_root, evidence_path.stem)
        outcomes: dict[str, list[dict[str, str]]] = {"danger": [], "non_danger": []}
        for row, target_ttc in zip(evidence, truth, strict=True):
            if parse_ttc(row["union_predicted_ttc"]) >= 2.0 or selected_box(row) is None:
                continue
            outcomes["danger" if target_ttc < 2.0 else "non_danger"].append(row)
        for rows in outcomes.values():
            selected.extend(spaced(rows, args.per_outcome))

    label_rows = []
    for row in selected:
        trip_id, frame_id = row["_trip_id"], int(row["frame_id"])
        label_rows.append({
            "trip_id": trip_id, "frame_id": frame_id,
            "left_image_path": str(args.practice_root / trip_id / "kitti" / "image_2" / f"{frame_id:06d}.jpg"),
            "selected_track_id": row.get("union_selected_track_id", ""),
            "object_id_window": "", "event_owner": "", "path_relation": "",
            "relative_motion": "", "cpa_distance_m": "", "occluded": "",
            "review_confidence": "", "notes": "",
        })
        box = selected_box(row)
        assert box is not None
        render_clip(
            practice_root=args.practice_root, trip_id=trip_id, frame_id=frame_id,
            target_box=box, radius=args.radius,
            output=args.output_dir / "clips" / trip_id / f"{frame_id:06d}.mp4",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "object_event_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(label_rows)
    print(json.dumps({"label_rows": len(label_rows), "blind": True, "radius_frames": args.radius}))


if __name__ == "__main__":
    main()
