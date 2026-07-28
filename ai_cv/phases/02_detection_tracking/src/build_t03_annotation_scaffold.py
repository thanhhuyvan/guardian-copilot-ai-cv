"""Stratified T03 annotation scaffold for Phase 04B YOLO26 Semantic Fusion.

Purpose
-------
The reproducible sweep (sweep_yolo26_fusion.py) proved that no frozen-contract
semantic configuration can lift macro F1 to 0.60: the oracle upper bound
(0.5577) is below the physical baseline (0.5634). The dominant degradation is
on T03, where every semantic config suppresses 750/1355 candidates and loses
8 true positives (FN 21 -> 29, MAE 62 -> 77s).

Before deciding whether to fine-tune YOLO26n (separate phase), this script
produces a human-labeling scaffold so the failure can be CLASSIFIED into one of
the three buckets the plan prescribes:

    genuine YOLO miss  -> fine-tune YOLO26 (separate phase)
    stereo noise        -> improve stereo confidence / fusion
    association failure -> improve bbox / track matching

It does NOT fine-tune and does NOT change any frozen threshold.

Outputs (all under --output-dir):
  - t03_annotation_template.csv      one row per suppressed candidate, with
    pre-populated diagnostic context (depth, TTC, YOLO match, score, misses,
    GT label) and an empty `label` column the human fills.
  - overlays/<frame_id>.jpg          rendered overlays for the sampled frames
    showing the stereo component box, the YOLO detection boxes, the depth/TTC
    text and a label legend.
  - t03_annotation_scaffold_report.json
    reproducibility record: which frames were sampled, sampling rule, counts.

Sampling rule
-------------
T03 has 497 frames carrying >=1 suppressed candidate and 750 suppressed
candidates total. We sample frames (not candidates) so a human labels
efficiently in one pass. Stratification:

    S1  all GT-danger frames (TTC<2s) carrying a suppressed candidate -- the
        recall loss; these are the highest priority. (18 frames in T03.)
    S2  remaining suppressed-only frames with no GT-danger, downsampled by
        uniform stride to reach the target ~200-300 distinct frames total.

Each sampled frame's overlay shows every suppressed candidate on that frame
alongside the YOLO detections for that frame, so the human can mark every
suppressed candidate's `label` in the CSV in one frame-by-frame pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cv2

from cross_validate_guarded_ttc import CURRENT_GUARD, GuardConfig, load_trip
from cross_validate_yolo26_fusion import (
    SemanticConfig,
    load_candidate_extras,
    load_detections_csv,
    predict_with_semantic_fusion,
)

TRIP_ID = "T03-Sample"
# Soft-guard config used by the reproducible sweep that produced
# the 750 suppressed candidates (sweep_yolo26_fusion.py oracle winner).
ANNOTATION_CFG = SemanticConfig(semantic_score_threshold=0.20,
                                consecutive_misses=2,
                                close_fallback_depth_m=5.0)
IMAGE_SHAPE = (360, 640, 3)


def _annotation_row(s: dict, k: int, fid: int, gt_danger_frames: set[int]) -> dict:
    """Build one label-template CSV row from a suppressed-candidate record.

    ``k`` is the 1-based box index printed on the red overlay box (the SAME
    enumerate() drives both the on-image label and this row's
    ``overlay_box_index``), so a CSV row can be matched to its box by either
    ``overlay_box_index`` or ``track_id``.
    """
    return {
        "frame_id": s["frame_id"],
        "track_id": s["track_id"],
        "overlay_box_index": k,
        "depth_m": f"{s['depth_m']:.3f}",
        "candidate_ttc": f"{s['candidate_ttc']:.3f}",
        "ground_truth_ttc": f"{s['ground_truth_ttc']:.3f}",
        "ground_confidence": f"{s['ground_confidence']:.3f}",
        "closing_speed_mps": f"{s['closing_speed_mps']:.3f}",
        "motion_residual_m": f"{s['motion_residual_m']:.3f}",
        "yolo_matched_class": s.get("matched_class", "") or "",
        "yolo_matched_confidence": "",
        "iou": "",
        "semantic_score": f"{s['semantic_score']:.4f}",
        "misses": s["misses"],
        "selected_center_x_norm": f"{s['selected_center_x_norm']:.4f}",
        "selected_bottom_y_norm": f"{s['selected_bottom_y_norm']:.4f}",
        "selected_width_norm": f"{s['selected_width_norm']:.4f}",
        "selected_height_norm": f"{s['selected_height_norm']:.4f}",
        "is_gt_danger_frame": int(fid in gt_danger_frames),
        "label": "",
        "annotator_notes": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the stratified T03 annotation scaffold."
    )
    parser.add_argument("--source-root", type=Path,
                        default=Path("ai_cv/outputs/benchmarks/phase04_loto/source"))
    parser.add_argument("--detections-dir", type=Path,
                        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/detections"))
    parser.add_argument("--practice-root", type=Path,
                        default=Path("Practice_Dataset"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_annotation"))
    parser.add_argument("--max-frames", type=int, default=240,
                        help="Soft cap on distinct frames sampled.")
    parser.add_argument("--danger-ttc-s", type=float, default=2.0,
                        help="GT TTC threshold for the 'danger' stratum (s).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    guard = GuardConfig(**CURRENT_GUARD)
    data = load_trip(args.source_root, TRIP_ID)
    dets_dir = args.detections_dir
    dets = load_detections_csv(dets_dir / f"{TRIP_ID}.csv")
    t_ids, widths = load_candidate_extras(args.source_root / "track_candidates" / f"{TRIP_ID}.csv")

    # Also load the full candidate CSV so we can recover the per-row geometry
    # (center_x_norm, bottom_y_norm, width_norm, height_norm, depth, candidate_ttc,
    # track_id) needed for accurate overlays and the label template.
    cand_path = args.source_root / "track_candidates" / f"{TRIP_ID}.csv"
    cand_rows = []
    with cand_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cand_rows.append(r)
    # Index: (frame_id, track_id) -> candidate row (last wins; duplicates are
    # handled by priority order below).
    cand_index: dict[tuple[int, int], dict] = {}
    for r in cand_rows:
        cand_index[(int(r["frame_id"]), int(r["track_id"]))] = r

    # Run the soft-guard with the annotation config to get suppressed candidates.
    preds, diag = predict_with_semantic_fusion(
        data, guard, ANNOTATION_CFG, dets, track_ids=t_ids, widths=widths,
        image_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
    )
    suppressed = [d for d in diag if d["suppressed"]]

    # Map each suppressed candidate to its canonical candidate row by
    # (frame_id, track_id). If multiple candidates share the same (frame, track),
    # we keep the diagnostic row but pick up geometry from the candidate CSV.
    sup_with_geom = []
    for d in suppressed:
        key = (d["frame_id"], d["track_id"])
        c = cand_index.get(key)
        if c is None:
            continue
        sup_with_geom.append({
            **d,
            "cand_idx": int(c.get("frame_id", 0)),  # not unique; debug only
            "depth_m": float(c["depth_m"]),
            "candidate_ttc": float(c["candidate_ttc"]),
            "ground_truth_ttc": float(c["ground_truth_ttc"]),
            "ground_confidence": float(c["ground_confidence"]),
            "closing_speed_mps": float(c["closing_speed_mps"]),
            "motion_residual_m": float(c["motion_residual_m"]),
            "selected_center_x_norm": float(c["selected_center_x_norm"]),
            "selected_bottom_y_norm": float(c["selected_bottom_y_norm"]),
            "selected_width_norm": float(c["selected_width_norm"]),
            "selected_height_norm": float(c["selected_height_norm"]),
        })

    # -------- Stratified frame sampling -------------------------------------
    gt_danger = (data.ground_truth < args.danger_ttc_s) & (data.ground_truth > 0.0)
    frame_to_id = {i: int(f) for i, f in enumerate(data.frame_ids)}
    gt_danger_frames = {frame_to_id[i] for i in np.where(gt_danger)[0]}

    # Group suppressed candidates by frame_id.
    sup_by_frame: dict[int, list[dict]] = {}
    for s in sup_with_geom:
        sup_by_frame.setdefault(s["frame_id"], []).append(s)

    sup_frames = set(sup_by_frame.keys())

    # Stratum S1: GT-danger + suppressed (the recall loss) -- ALL of them.
    s1 = sorted(sup_frames & gt_danger_frames)
    # Stratum S2: suppressed-only, no GT-danger, strided to fill the budget.
    s2_pool = sorted(sup_frames - gt_danger_frames)
    budget_s2 = max(0, args.max_frames - len(s1))
    if s2_pool:
        stride = max(1, len(s2_pool) // budget_s2) if budget_s2 > 0 else len(s2_pool)
        s2 = s2_pool[::stride][:budget_s2]
    else:
        s2 = []

    sampled_frames = s1 + s2
    print(f"Stratified sampling for {TRIP_ID}:")
    print(f"  S1 (GT-danger + suppressed): {len(s1)} frames (all)")
    print(f"  S2 (suppressed-only, strided): {len(s2)} frames")
    print(f"  total sampled frames:           {len(sampled_frames)}")
    print(f"  total suppressed candidates on sampled frames: "
          f"{sum(len(sup_by_frame[f]) for f in sampled_frames)} / {len(suppressed)} total")

    # -------- Render overlays AND write label template in one pass -----------
    # Crucial: the red overlay box label "#k" and the CSV "overlay_box_index"
    # MUST come from the same enumerate() call so a row can be tied to its box.
    # We therefore render each frame and emit its CSV rows together, in the
    # candidate order shown on the overlay (frame_id ascending -> red box #1,
    # #2, #3 ...). The CSV's "overlay_box_index" column is the same k printed
    # on the box, and "track_id" is also printed alongside it on the box, so
    # the row can be matched by either key.
    csv_path = args.output_dir / "t03_annotation_template.csv"
    cols = [
        "frame_id", "track_id", "overlay_box_index",  # match overlay "#k"
        "depth_m", "candidate_ttc", "ground_truth_ttc", "ground_confidence",
        "closing_speed_mps", "motion_residual_m",
        "yolo_matched_class", "yolo_matched_confidence", "iou",
        "semantic_score", "misses",
        "selected_center_x_norm", "selected_bottom_y_norm",
        "selected_width_norm", "selected_height_norm",
        "is_gt_danger_frame",
        "label",            # genuine_miss | stereo_noise | association_failure | unsure
        "annotator_notes",
    ]
    img_dir = args.practice_root / TRIP_ID / "kitti" / "image_2"
    missing = 0
    rendered = 0
    rows_written = 0
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for fid in sampled_frames:
            img_path = img_dir / f"{fid:06d}.jpg"
            if not img_path.is_file():
                missing += 1
                # Even if the source frame is missing we still emit its CSV rows
                # so the candidate inventory is complete; overlay_box_index is
                # still assigned deterministically (matches what WOULD have been
                # drawn).
                for k, s in enumerate(sup_by_frame[fid], start=1):
                    writer.writerow(_annotation_row(s, k, fid, gt_danger_frames))
                    rows_written += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                missing += 1
                for k, s in enumerate(sup_by_frame[fid], start=1):
                    writer.writerow(_annotation_row(s, k, fid, gt_danger_frames))
                    rows_written += 1
                continue
            h, w = img.shape[:2]

            # Draw YOLO detections for this frame in light blue.
            for det in dets.get(fid, []):
                x0, y0, x1, y1 = det.bbox_xyxy
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                              (255, 200, 0), 1)
                cv2.putText(img, f"Y:{det.class_name[:4]}/{det.confidence:.2f}",
                            (int(x0), max(10, int(y0) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1, cv2.LINE_AA)

            # Draw each suppressed candidate in red. THE SAME enumerate() drives
            # both the on-image "#k #tid" label and the CSV "overlay_box_index".
            for k, s in enumerate(sup_by_frame[fid], start=1):
                cx = s["selected_center_x_norm"] * w
                by = s["selected_bottom_y_norm"] * h
                bw = s["selected_width_norm"] * w
                bh = s["selected_height_norm"] * h
                x0 = max(0, int(cx - bw / 2))
                y0 = max(0, int(by - bh))
                x1 = min(w, int(cx + bw / 2))
                y1 = min(h, int(by))
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                # Print BOTH the box index AND the track_id so the annotator can
                # match the CSV row by either key when the box label is too long.
                cv2.putText(img,
                            f"#{k} tid={s['track_id']} d={s['depth_m']:.1f}m "
                            f"TTC={s['candidate_ttc']:.2f}s GT={s['ground_truth_ttc']:.2f}",
                            (x0, max(12, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
                # Emit the matching CSV row.
                writer.writerow(_annotation_row(s, k, fid, gt_danger_frames))
                rows_written += 1

            # Header bar with frame id and count.
            cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
            cv2.putText(img, f"{TRIP_ID} frame {fid:06d}  |  sup={len(sup_by_frame[fid])}  "
                              f"GT danger={'Y' if fid in gt_danger_frames else 'N'}  |  "
                              f"RED=suppressed component  CYAN=YOLO det",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            out_path = overlay_dir / f"{fid:06d}.jpg"
            cv2.imwrite(str(out_path), img)
            rendered += 1

    print(f"\nOverlays: rendered={rendered} missing_src={missing}")
    print(f"Label template: {csv_path} ({rows_written} rows, {len(cols)} cols)")
    print("  -> row matcher: overlay_box_index (matches the red box '#k' on the "
          "overlay) + track_id (also printed on the box)")

    # -------- Reproducibility report ----------------------------------------
    report = {
        "trip_id": TRIP_ID,
        "annotation_config": {
            "semantic_score_threshold": ANNOTATION_CFG.semantic_score_threshold,
            "consecutive_misses": ANNOTATION_CFG.consecutive_misses,
            "close_fallback_depth_m": ANNOTATION_CFG.close_fallback_depth_m,
        },
        "danger_ttc_threshold_s": args.danger_ttc_s,
        "sampling_rule": {
            "S1": "all GT-danger frames (< danger_ttc_s) with >=1 suppressed candidate",
            "S2": "remaining suppressed-only frames, uniform-strided to budget",
            "max_frames": args.max_frames,
        },
        "counts": {
            "suppressed_candidates_total": len(suppressed),
            "suppressed_candidates_on_sampled_frames": rows_written,
            "candidate_rows_in_track_csv": len(cand_rows),
            "frames_total_t03_gt_danger": int(gt_danger.sum()),
            "frames_suppressed": len(sup_frames),
            "frames_sampled_S1": len(s1),
            "frames_sampled_S2": len(s2),
            "frames_sampled_total": len(sampled_frames),
            "overlays_rendered": rendered,
            "overlays_missing_src": missing,
        },
        "label_codes": {
            "genuine_miss": "the suppressed candidate IS a real road user that "
                           "YOLO failed to detect -> fine-tune YOLO26",
            "stereo_noise": "the suppressed candidate is NOT a real object, "
                            "just stereo/depth noise -> improve stereo confidence/fusion",
            "association_failure": "YOLO did detect the object but the "
                                   "component-detection match failed (IoU/center) "
                                   "-> improve bbox/track matching",
            "unsure": "ambiguous or needs more context",
        },
        "outputs": {
            "label_template": str(csv_path),
            "overlays_dir": str(overlay_dir),
            "label_matcher_columns": ["overlay_box_index", "track_id"],
            "overlay_box_label_format": "#{overlay_box_index} tid={track_id} d=... TTC=... GT=...",
            "next_step_validator": "src/validate_t03_annotation.py --csv <label_template>",
        },
    }
    report_path = args.output_dir / "t03_annotation_scaffold_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Scaffold report: {report_path}")

    print("\n================ Annotation summary ================")
    print(f"Sampled frames:    {len(sampled_frames)}  "
          f"(S1 danger={len(s1)}, S2 strided={len(s2)})")
    print(f"Suppressed cands:   {rows_written}/{len(suppressed)} on sampled frames")
    print(f"Overlays rendered:  {rendered}  (missing source imgs: {missing})")
    print(f"Label template:     {csv_path}")
    print(f"Label the S1 rows first (is_gt_danger_frame=1, the recall loss):")
    print("  1. Open the CSV; for each row, open overlays/<frame_id>.jpg.")
    print("  2. Match the row by overlay_box_index (the '#k' on the red box) and")
    print("     track_id (also printed on the box as 'tid=...').")
    print("  3. Fill 'label': genuine_miss | stereo_noise | association_failure | unsure.")
    print("  4. Add a short 'annotator_notes' when uncertain.")
    print("  5. Save as CSV WITHOUT changing headers.")
    print("  6. Run: python src/validate_t03_annotation.py --csv <template>")


if __name__ == "__main__":
    main()
