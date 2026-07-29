"""Create a small stratified CPA/path-relation labeling CSV from V2 evidence."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--per-trip",type=int,default=5)
    args=parser.parse_args(); rows=[]
    for trip_file in sorted(args.evidence_root.glob("T*-Sample.csv")):
        picked=0
        for row in csv.DictReader(trip_file.open(encoding="utf-8",newline="")):
            updates=json.loads(row.get("v2_shadow_updates_json","[]"))
            if not updates or picked>=args.per_trip: continue
            update=max(updates,key=lambda x:x.get("mahalanobis_squared",0.0))
            rows.append({"trip_id":trip_file.stem,"frame_id":row["frame_id"],"track_id":update["track_id"],"mahalanobis_squared":update["mahalanobis_squared"],"occupancy_probability":update.get("corridor_occupancy_probability",""),"path_relation":"","cpa_distance_m":"","occluded":"","notes":""}); picked+=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]) if rows else ["trip_id","frame_id"]);w.writeheader();w.writerows(rows)

if __name__=="__main__": main()
