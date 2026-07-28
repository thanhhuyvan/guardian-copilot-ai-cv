"""Run PyTorch YOLO26 detector reference over left video frames."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np

from yolo26_backends import get_detector_backend, PyTorchYolo26Detector


TRIPS = [f"T0{i}-Sample" for i in range(1, 7)]


def run_detector_reference(
    dataset_root: Path,
    output_dir: Path,
    model_path: str = "yolo26n.pt",
    confidence_threshold: float = 0.25,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "detections"
    csv_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays_72"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO26 detector from {model_path}...")
    detector = get_detector_backend(
        backend_name="yolo26-pytorch",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )

    all_timings_ms = []
    class_counts = {}
    trip_summaries = {}

    for trip_id in TRIPS:
        trip_path = dataset_root / trip_id
        left_img_dir = trip_path / "kitti" / "image_2"
        json_path = trip_path / "telemetry.json"

        if not left_img_dir.is_dir():
            print(f"Skipping missing trip dir: {left_img_dir}")
            continue

        frame_files = sorted(left_img_dir.glob("*.png")) + sorted(left_img_dir.glob("*.jpg"))
        if not frame_files:
            print(f"No frames found in {left_img_dir}")
            continue

        print(f"Processing {trip_id}: {len(frame_files)} frames...")
        out_csv_path = csv_dir / f"{trip_id}.csv"

        # Stratified frame indices for 12 overlay frames per trip (out of ~600)
        num_frames = len(frame_files)
        stratified_indices = set(np.linspace(0, num_frames - 1, 12, dtype=int))

        csv_rows = []
        trip_timings = []

        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame_id", "timestamp", "class_id", "class_name", "confidence", "x0", "y0", "x1", "y1"]
            )

            for idx, img_file in enumerate(frame_files):
                frame_id = idx + 1
                timestamp = idx * 0.05  # 20 FPS

                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                res = detector.infer(img)
                total_frame_ms = res.preprocess_ms + res.inference_ms + res.postprocess_ms
                trip_timings.append(total_frame_ms)
                all_timings_ms.append(total_frame_ms)

                for det in res.detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                    x0, y0, x1, y1 = det.bbox_xyxy
                    writer.writerow(
                        [
                            frame_id,
                            f"{timestamp:.3f}",
                            det.class_id,
                            det.class_name,
                            f"{det.confidence:.4f}",
                            f"{x0:.2f}",
                            f"{y0:.2f}",
                            f"{x1:.2f}",
                            f"{y1:.2f}",
                        ]
                    )

                # Save overlay if in 72 stratified set
                if idx in stratified_indices:
                    overlay_img = img.copy()
                    for det in res.detections:
                        x0, y0, x1, y1 = map(int, det.bbox_xyxy)
                        cv2.rectangle(overlay_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        label = f"{det.class_name} {det.confidence:.2f}"
                        cv2.putText(
                            overlay_img,
                            label,
                            (x0, max(15, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                    out_img_path = overlay_dir / f"{trip_id}_frame_{frame_id:04d}.jpg"
                    cv2.imwrite(str(out_img_path), overlay_img)

        p50 = float(np.percentile(trip_timings, 50))
        p95 = float(np.percentile(trip_timings, 95))
        p99 = float(np.percentile(trip_timings, 99))
        trip_summaries[trip_id] = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}

    detector.close()

    p50_all = float(np.percentile(all_timings_ms, 50)) if all_timings_ms else 0.0
    p95_all = float(np.percentile(all_timings_ms, 95)) if all_timings_ms else 0.0
    p99_all = float(np.percentile(all_timings_ms, 99)) if all_timings_ms else 0.0

    summary = {
        "model_path": model_path,
        "confidence_threshold": confidence_threshold,
        "total_frames_processed": len(all_timings_ms),
        "overall_latency_ms": {"p50": p50_all, "p95": p95_all, "p99": p99_all},
        "class_counts": class_counts,
        "trip_latency_ms": trip_summaries,
    }

    summary_path = output_dir / "detector_reference_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetector reference complete!")
    print(f"P50: {p50_all:.2f} ms, P95: {p95_all:.2f} ms, P99: {p99_all:.2f} ms")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO26 PyTorch detector reference.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Practice_Dataset"),
        help="Path to Practice_Dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference"),
        help="Output directory for CSVs, overlays and summary",
    )
    parser.add_argument("--model-path", type=str, default="yolo26n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    run_detector_reference(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        confidence_threshold=args.confidence,
    )


if __name__ == "__main__":
    main()
