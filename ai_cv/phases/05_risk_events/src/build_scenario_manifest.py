"""Create telemetry-only scenario strata for V2 evaluation."""
from __future__ import annotations
import argparse, csv
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--starter-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn-accel-mps2", type=float, default=0.3)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.starter_root))
    from team_kit.dataset_loader import TripDataset
    rows=[]
    for trip_dir in sorted(args.practice_root.glob("T*-Sample")):
        for frame in TripDataset(trip_dir).iter_frames():
            rows.append({"trip_id":trip_dir.name,"frame_id":int(frame.frame_id),"scenario":"turn" if abs(frame.lateral_accel)>=args.turn_accel_mps2 else "straight","speed_mps":round(float(frame.speed_kmh)/3.6,3),"lateral_accel_mps2":round(float(frame.lateral_accel),3)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

if __name__ == "__main__": main()
