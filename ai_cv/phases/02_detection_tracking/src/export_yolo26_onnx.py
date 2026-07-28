"""Export YOLO26 to static ONNX FP32 and validate parity against PyTorch reference.

Parity is measured on raw (unannotated) source frames, not on detector overlays.
Zero matched detections is a hard failure — the ONNX model must detect at least
one object that PyTorch also detects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from semantic_fusion import compute_iou

# Minimum number of cross-matched detection pairs required for the parity
# report to be considered valid.  Zero matches = the ONNX parser is broken.
MIN_MATCHED_DETECTIONS = 10


def collect_raw_source_frames(
    practice_root: Path,
    n_per_trip: int = 12,
) -> list[Path]:
    """Return up to n_per_trip raw left-camera frames from each practice trip.

    Uses Practice_Dataset/T0{1-6}-Sample/kitti/image_2/*.png (or .jpg).
    Falls back to any image found recursively if the standard path is absent.
    """
    trips = [f"T0{i}-Sample" for i in range(1, 7)]
    frames: list[Path] = []
    for trip in trips:
        img_dir = practice_root / trip / "kitti" / "image_2"
        if img_dir.is_dir():
            images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        else:
            # Fallback: search under trip root
            images = sorted((practice_root / trip).rglob("*.png"))[:n_per_trip * 2]
        if not images:
            print(f"  WARNING: no source images found for {trip} under {img_dir}")
            continue
        # Evenly-spaced sample
        step = max(1, len(images) // n_per_trip)
        frames.extend(images[::step][:n_per_trip])
    return frames


def export_and_validate_onnx(
    pytorch_model_path: str = "yolo26n.pt",
    onnx_output_path: str = "yolo26n.onnx",
    practice_root: Path = Path("Practice_Dataset"),
    output_dir: Path = Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_export"),
    n_per_trip: int = 12,
    skip_export: bool = False,
) -> None:
    import onnx
    from ultralytics import YOLO
    from yolo26_backends import ONNXYolo26Detector, PyTorchYolo26Detector

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(onnx_output_path)

    # ------------------------------------------------------------------ export
    if skip_export:
        if not onnx_path.is_file():
            print(f"ERROR: --skip-export set but '{onnx_path}' does not exist.")
            sys.exit(1)
        print(f"Skipping export; using existing '{onnx_path}'.")
    else:
        print(f"Exporting '{pytorch_model_path}' -> ONNX static FP32 ...")
        model = YOLO(pytorch_model_path)
        exported_path = model.export(format="onnx", dynamic=False, imgsz=640, simplify=True)
        if exported_path and Path(exported_path).is_file():
            if Path(exported_path).resolve() != onnx_path.resolve():
                Path(exported_path).rename(onnx_path)

        print(f"Validating ONNX structure with onnx.checker on '{onnx_path}' ...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX checker: VALID")

    # -------------------------------------------------------- collect frames
    print(f"\nCollecting raw source frames from {practice_root} ...")
    source_frames = collect_raw_source_frames(practice_root, n_per_trip=n_per_trip)
    if not source_frames:
        print("ERROR: No source frames found. Cannot run parity test.")
        sys.exit(1)
    print(f"  {len(source_frames)} frames from {len(set(p.parts[-4] for p in source_frames))} trips")

    # ----------------------------------------------------------- parity test
    print("\nRunning parity comparison (PyTorch FP32 vs ONNX FP32) ...")
    torch_det = PyTorchYolo26Detector(model_path=pytorch_model_path)
    onnx_det  = ONNXYolo26Detector(model_path=str(onnx_path))
    print(f"  PyTorch device: {torch_det.device}")
    print(f"  ONNX providers: {onnx_det.providers}")

    class_matches = 0
    total_matched = 0
    ious: list[float] = []
    conf_diffs: list[float] = []
    frames_with_torch_dets = 0
    class_swaps: list[dict] = []

    for img_path in source_frames:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: could not read {img_path}")
            continue

        t_res = torch_det.infer(img)
        o_res = onnx_det.infer(img)

        if t_res.detections:
            frames_with_torch_dets += 1

        for t_det in t_res.detections:
            best_iou = 0.0
            best_o = None
            for o_det in o_res.detections:
                v = compute_iou(t_det.bbox_xyxy, o_det.bbox_xyxy)
                if v > best_iou:
                    best_iou = v
                    best_o = o_det
            if best_o is not None and best_iou >= 0.50:
                total_matched += 1
                ious.append(best_iou)
                conf_diffs.append(abs(t_det.confidence - best_o.confidence))
                if t_det.class_id == best_o.class_id:
                    class_matches += 1
                else:
                    # Same physical object (IoU~1) but different class label.
                    # The end-to-end ONNX export retains competing class
                    # hypotheses per box and tiebreaks by insertion order;
                    # native Ultralytics postprocess (PyTorch) tiebreaks
                    # differently. Downstream fusion treats car/truck
                    # identically (only retained-class membership + confidence
                    # matter), so these are flagged but not blocking.
                    class_swaps.append({
                        "frame": img_path.name,
                        "torch_class": t_det.class_name,
                        "torch_conf": round(t_det.confidence, 4),
                        "onnx_class": best_o.class_name,
                        "onnx_conf": round(best_o.confidence, 4),
                        "iou": round(best_iou, 4),
                        "same_box": best_iou >= 0.95,
                    })

    torch_det.close()
    onnx_det.close()

    # ---------------------------------------------------------- hard failure
    # Zero matched detections means the ONNX parser is broken — the model
    # produced output but none matched the PyTorch reference.  This must be a
    # hard failure so it is never silently reported as "100% agreement".
    if total_matched == 0:
        msg = (
            f"PARITY HARD FAILURE: 0 matched detections across {len(source_frames)} frames "
            f"({frames_with_torch_dets} frames had PyTorch detections). "
            "The ONNX output parser is likely incorrect."
        )
        print(f"\n{msg}")
        parity_summary = {
            "total_matched_detections": 0,
            "frames_with_pytorch_detections": frames_with_torch_dets,
            "class_agreement_ratio": None,
            "median_matched_box_iou": None,
            "mean_confidence_diff": None,
            "parity_valid": False,
            "failure_reason": msg,
            "gates_passed": {
                "min_matched_detections": False,
                "class_agreement_ge_99": False,
                "median_iou_ge_98": False,
                "mean_conf_diff_le_0_02": False,
            },
        }
        out_path = output_dir / "onnx_parity_report.json"
        with out_path.open("w") as f:
            json.dump(parity_summary, f, indent=2)
        print(f"Report saved to: {out_path}")
        sys.exit(1)

    if total_matched < MIN_MATCHED_DETECTIONS:
        print(
            f"WARNING: only {total_matched} matched detections (minimum expected: {MIN_MATCHED_DETECTIONS}). "
            "Results may be unreliable."
        )

    class_agreement = class_matches / total_matched
    median_iou      = float(np.median(ious))
    mean_conf_diff  = float(np.mean(conf_diffs))
    conf_arr = np.array(conf_diffs)

    # A class swap where IoU>=0.95 is the same physical box with a competing
    # class label, not a localization error. These arise from the end-to-end
    # ONNX export retaining competing class hypotheses per box.
    same_box_swaps = [s for s in class_swaps if s["same_box"]]
    distinct_object_swaps = [s for s in class_swaps if not s["same_box"]]

    parity_summary = {
        "total_matched_detections": total_matched,
        "frames_with_pytorch_detections": frames_with_torch_dets,
        "pytorch_device": torch_det.device,
        "onnx_providers": onnx_det.providers,
        "class_agreement_ratio": class_agreement,
        "median_matched_box_iou": median_iou,
        "mean_confidence_diff": mean_conf_diff,
        "confidence_diff_distribution": {
            "p50": round(float(np.percentile(conf_arr, 50)), 4),
            "p90": round(float(np.percentile(conf_arr, 90)), 4),
            "p95": round(float(np.percentile(conf_arr, 95)), 4),
            "p99": round(float(np.percentile(conf_arr, 99)), 4),
            "max": round(float(conf_arr.max()), 4),
            "count_gt_0_02": int((conf_arr > 0.02).sum()),
        },
        "class_swaps": {
            "total": len(class_swaps),
            "same_box_competing_class": len(same_box_swaps),
            "distinct_object": len(distinct_object_swaps),
            "details": class_swaps,
        },
        "parity_valid": True,
        "gates_passed": {
            "min_matched_detections": total_matched >= MIN_MATCHED_DETECTIONS,
            "class_agreement_ge_99": class_agreement >= 0.99,
            "median_iou_ge_98": median_iou >= 0.98,
            "mean_conf_diff_le_0_02": mean_conf_diff <= 0.02,
        },
    }

    out_path = output_dir / "onnx_parity_report.json"
    with out_path.open("w") as f:
        json.dump(parity_summary, f, indent=2)

    print("\n================ ONNX Parity Report ================")
    print(f"PyTorch device:         {torch_det.device}")
    print(f"ONNX providers:         {onnx_det.providers}")
    print(f"Frames tested:          {len(source_frames)}  ({frames_with_torch_dets} with PyTorch detections)")
    print(f"Matched detection pairs:{total_matched}  (min required: {MIN_MATCHED_DETECTIONS})")
    print(f"Class Agreement:        {class_agreement * 100:.2f}%  (Gate: >= 99%)")
    print(f"Median Matched IoU:     {median_iou:.4f}   (Gate: >= 0.98)")
    print(f"Mean Confidence Diff:   {mean_conf_diff:.4f}   (Gate: <= 0.02)")
    print(f"Conf diff p50/p95/max:  {parity_summary['confidence_diff_distribution']['p50']}"
          f" / {parity_summary['confidence_diff_distribution']['p95']}"
          f" / {parity_summary['confidence_diff_distribution']['max']}")
    print(f"Class swaps:            {len(class_swaps)} total "
          f"({len(same_box_swaps)} same-box competing-class, "
          f"{len(distinct_object_swaps)} distinct-object)")
    all_pass = all(parity_summary["gates_passed"].values())
    print(f"Overall: {'ALL GATES PASSED' if all_pass else 'GATES FAILED'}")
    print(f"Report saved to: {out_path}")
    if not all_pass:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO26 to ONNX and validate parity.")
    parser.add_argument("--model-path",     type=str,  default="yolo26n.pt")
    parser.add_argument("--onnx-path",      type=str,  default="yolo26n.onnx")
    parser.add_argument("--practice-root",  type=Path, default=Path("Practice_Dataset"))
    parser.add_argument("--n-per-trip",     type=int,  default=12)
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Reuse the existing ONNX file instead of re-exporting.",
    )
    args = parser.parse_args()

    export_and_validate_onnx(
        pytorch_model_path=args.model_path,
        onnx_output_path=args.onnx_path,
        practice_root=args.practice_root,
        n_per_trip=args.n_per_trip,
        skip_export=args.skip_export,
    )


if __name__ == "__main__":
    main()
