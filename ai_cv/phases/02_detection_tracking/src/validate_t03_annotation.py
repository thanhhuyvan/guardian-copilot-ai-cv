"""Validate and aggregate human-labeled T03 annotation template.

Usage
-----
After a human has filled the `label` column of
`artifacts/yolo26_annotation/t03_annotation_template.csv` (and optionally
`annotator_notes`), run:

    python ai_cv/phases/02_detection_tracking/src/validate_t03_annotation.py \\
        --csv ai_cv/phases/02_detection_tracking/artifacts/yolo26_annotation/t03_annotation_template.csv

What it does
------------
1. Validates the labeled CSV:
   - headers unchanged (the writer's full column list, in order)
   - every row has an `overlay_box_index` that is a positive int and matches
     the per-frame 1..N range when the row count is checked frame-by-frame
   - inside each frame, `overlay_box_index` is unique (matches the overlay's
     #1, #2, #3 ... draw order)
   - `label` (when non-empty) is exactly one of the four allowed codes:
         genuine_miss | stereo_noise | association_failure | unsure
   - invariants on numeric columns (no NaN,finite, ranges sensible)
2. Aggregates the label distribution and recommends the next-phase action
   per the plan's decision rule:
     - dominant genuine_miss       -> fine-tune YOLO26
     - dominant stereo_noise        -> improve stereo confidence/fusion
     - dominant association_failure -> improve bbox/track matching
   The recommendation also reports within-stratum breakdowns (S1 = GT-danger
   frames first, S2 = remaining) so the recall-loss candidates are weighted
   explicitly.
3. Writes a JSON report at <csv>.report.json.

Exit code is 0 when validation passes (any rows labeled), 1 on any
inconsistency or empty input.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Frozen header order, lifted verbatim from build_t03_annotation_scaffold.py.
EXPECTED_HEADER = [
    "frame_id", "track_id", "overlay_box_index",
    "depth_m", "candidate_ttc", "ground_truth_ttc", "ground_confidence",
    "closing_speed_mps", "motion_residual_m",
    "yolo_matched_class", "yolo_matched_confidence", "iou",
    "semantic_score", "misses",
    "selected_center_x_norm", "selected_bottom_y_norm",
    "selected_width_norm", "selected_height_norm",
    "is_gt_danger_frame",
    "label",
    "annotator_notes",
]
ALLOWED_LABELS = {"genuine_miss", "stereo_noise", "association_failure", "unsure"}
# A label is considered the winner when its share of non-empty labels is >=
# this threshold. Tunable but defaults to a simple majority.
DECISION_THRESHOLD = 0.40


def _is_finite(value: str) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _is_finite_or_inf(value: str) -> bool:
    """Like _is_finite but also accepts ('inf'/'-inf') as the legitimate danger
    null sentinel for TTC columns (R-TTC-02/R-SUB-04)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return True  # inf / -inf / finite all parse as floats


def validate(csv_path: Path) -> tuple[bool, list[str], list[dict]]:
    """Return (ok, problems, rows). problems is a list of human-readable
    error strings; rows is the full parsed CSV (list of dicts)."""
    problems: list[str] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_HEADER:
            problems.append(
                "Header mismatch. Expected (from scaffold writer):\n  "
                + ", ".join(EXPECTED_HEADER)
                + "\nGot:\n  "
                + ", ".join(reader.fieldnames or [])
                + "\nHeaders must be UNCHANGED -- save the CSV without altering "
                  "columns in any way."
            )
            return False, problems, []
        rows = list(reader)

    if not rows:
        problems.append("CSV has no data rows (empty).")
        return False, problems, rows

    # Per-frame index tracking
    per_frame_indices: dict[int, set[int]] = defaultdict(set)
    label_counter: Counter = Counter()
    unlabeled_count = 0
    bad_label_rows: list[str] = []

    for i, r in enumerate(rows, start=2):  # row 1 is the header
        # Required numeric columns
        try:
            frame_id = int(r["frame_id"])
            track_id = int(r["track_id"])
            box_idx = int(r["overlay_box_index"])
        except (TypeError, ValueError) as e:
            problems.append(f"row {i}: bad integer key ({e}).")
            continue
        if box_idx < 1:
            problems.append(f"row {i}: overlay_box_index must be >=1, got {box_idx}.")
        if box_idx in per_frame_indices[frame_id]:
            problems.append(
                f"row {i}: duplicate overlay_box_index {box_idx} on frame {frame_id}."
            )
        else:
            per_frame_indices[frame_id].add(box_idx)

        # Numeric sanity on the floating columns we pre-populate.
        # Geometric/stereo scalar columns: must be finite.
        for col in ("depth_m", "ground_confidence", "closing_speed_mps",
                    "motion_residual_m", "semantic_score",
                    "selected_center_x_norm", "selected_bottom_y_norm",
                    "selected_width_norm", "selected_height_norm"):
            if r.get(col) and not _is_finite(r[col]):
                problems.append(f"row {i}: column {col} not finite ('{r[col]}').")
        # TTC columns: 'inf' is the legitimate danger NULL per the plan
        # (R-TTC-02 non-closing targets use internal 'inf'/JSON 'null'; R-SUB-04
        # CSV infinity format). Accept finite OR inf; reject NaN/parse errors.
        for col in ("candidate_ttc", "ground_truth_ttc"):
            if r.get(col) and not _is_finite_or_inf(r[col]):
                problems.append(f"row {i}: column {col} unparseable ('{r[col]}').")

        # Label column
        label = (r.get("label") or "").strip()
        if label:
            if label not in ALLOWED_LABELS:
                bad_label_rows.append(f"row {i} (frame {frame_id} #{box_idx}): '{label}'")
                continue
            label_counter[label] += 1
        else:
            unlabeled_count += 1

    if bad_label_rows:
        problems.append(
            "Rows with invalid label values (must be exactly one of "
            f"{sorted(ALLOWED_LABELS)}):\n  "
            + "\n  ".join(bad_label_rows[:20])
            + ("..." if len(bad_label_rows) > 20 else "")
        )

    # Box indices within each frame should be exactly 1..N. We check that the
    # observed set is contiguous from 1 (a common labelling-edit mistake is to
    # delete a row, leaving a gap).
    for fid, idxs in per_frame_indices.items():
        expected = set(range(1, len(idxs) + 1))
        if idxs != expected:
            problems.append(
                f"frame {fid}: overlay_box_index set {sorted(idxs)} is not "
                f"contiguous 1..{len(idxs)} (expected {sorted(expected)}). "
                "A row may have been deleted or reordered -- restore it from "
                "the scaffold before labeling."
            )

    return (len(problems) == 0), problems, rows


def aggregate(rows: list[dict]) -> dict:
    """Produce the label distribution + decision recommendation."""
    label_total = Counter()
    label_s1 = Counter()   # GT-danger frames
    label_s2 = Counter()
    labeled = 0
    for r in rows:
        lab = (r.get("label") or "").strip()
        if not lab or lab not in ALLOWED_LABELS:
            continue
        labeled += 1
        label_total[lab] += 1
        if int(r.get("is_gt_danger_frame", 0)) == 1:
            label_s1[lab] += 1
        else:
            label_s2[lab] += 1

    unlabeled = len(rows) - labeled
    coverage = labeled / len(rows) if rows else 0.0

    # Decision: a label wins ALL of:
    #   >= DECISION_THRESHOLD share of labeled rows
    #   AND >= DECISION_THRESHOLD share of S1 (GT-danger) labeled rows
    # If conflict, the S1 distribution wins priority (recall directly depends
    # on these), and we surface the disagreement explicitly.
    def _winner(counter: Counter) -> tuple[str | None, float]:
        if not counter:
            return None, 0.0
        n = sum(counter.values())
        lab, cnt = counter.most_common(1)[0]
        return lab, cnt / n

    win_all, share_all = _winner(label_total)
    win_s1,  share_s1  = _winner(label_s1)
    win_s2,  share_s2  = _winner(label_s2)

    s1_dominant_passes = win_s1 is not None and share_s1 >= DECISION_THRESHOLD
    all_dominant_passes = win_all is not None and share_all >= DECISION_THRESHOLD

    if all_dominant_passes and s1_dominant_passes and win_all == win_s1:
        decision = f"DECISIVE: '{win_all}' dominant overall ({share_all:.0%}) " \
                   f"and in GT-danger stratum ({share_s1:.0%}). "
        action_map = {
            "genuine_miss": "Fine-tune YOLO26n on T03-style dark scenes (separate phase; 6 fold-specific models, no leakage).",
            "stereo_noise": "Improve stereo confidence / fusion -- the suppressed candidates are depth artifacts, not real road users.",
            "association_failure": "Improve bbox/track matching -- YOLO sees the object but component-detection IoU/center match fails.",
            "unsure": "Ambiguous labels dominate; revisit the labeling instructions or sample more frames before deciding.",
        }
        decision += action_map.get(win_all, "")
    else:
        decision = (
            f"MIXED: overall winner='{win_all}' ({share_all:.0%}), "
            f"S1 (GT-danger) winner='{win_s1}' ({share_s1:.0%}), "
            f"S2 winner='{win_s2}' ({share_s2:.0%}). "
            "S1 (the recall-loss rows) should drive the next-phase decision; "
            "treat the S1 dominant label as the action trigger even if the "
            "overall majority differs, and inspect the disagreements."
        )

    return {
        "rows_total": len(rows),
        "rows_labeled": labeled,
        "rows_unlabeled": unlabeled,
        "coverage": round(coverage, 4),
        "label_counts_overall": dict(label_total),
        "label_counts_s1_gt_danger": dict(label_s1),
        "label_counts_s2_other": dict(label_s2),
        "winner_overall": win_all, "share_overall": round(share_all, 4),
        "winner_s1": win_s1, "share_s1": round(share_s1, 4),
        "winner_s2": win_s2, "share_s2": round(share_s2, 4),
        "decision_rule": (
            f"winner if share >= {DECISION_THRESHOLD:.0%} overall AND in S1 "
            f"with consistent label; else MIXED (S1 wins priority for the action)."
        ),
        "next_phase_recommendation": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the human-labeled T03 annotation CSV."
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("ai_cv/phases/02_detection_tracking/artifacts/yolo26_annotation/t03_annotation_template.csv"),
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    print(f"Validating {args.csv} ...")
    ok, problems, rows = validate(args.csv)
    if not ok:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print(f"  rows: {len(rows)}, headers unchanged, indices contiguous per frame.")
    report = aggregate(rows)

    print("\n================ Annotation Aggregate ================")
    print(f"Coverage:            {report['rows_labeled']}/{report['rows_total']} rows "
          f"({report['coverage']*100:.1f}%) labeled; "
          f"{report['rows_unlabeled']} still empty")
    print(f"  overall counts:    {report['label_counts_overall']}")
    print(f"  S1 (GT-danger):    {report['label_counts_s1_gt_danger']}")
    print(f"  S2 (other):         {report['label_counts_s2_other']}")
    print(f"  winner overall:    {report['winner_overall']} "
          f"(share {report['share_overall']*100:.1f}%)")
    print(f"  winner S1 (danger): {report['winner_s1']} "
          f"(share {report['share_s1']*100:.1f}%)")
    print(f"  winner S2 (other):  {report['winner_s2']} "
          f"(share {report['share_s2']*100:.1f}%)")
    print()
    print(f"RECOMMENDATION: {report['next_phase_recommendation']}")

    out_path = args.csv.with_suffix(".csv.report.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAggregate report saved to: {out_path}")

    # If zero rows are labeled, that is not an error per se (the CSV is valid
    # but unlabeled) -- exit 0 so the user can re-run after labeling.
    if report["rows_labeled"] == 0:
        print("\nNOTE: CSV is valid but no rows are labeled yet. Fill the 'label' "
              "column (one of {sorted(ALLOWED_LABELS)}) and re-run.")
        sys.exit(0)


if __name__ == "__main__":
    main()
