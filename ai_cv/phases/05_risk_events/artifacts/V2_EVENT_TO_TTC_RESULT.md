# V2 event-to-TTC result: invalidated; rerun required

This historical run is invalid evidence. A July 30 source review found V2
candidate TTC initialized before final conservative-union selection, violating
its own contract to begin with raw union TTC. Fixed thresholds remain frozen;
a corrected one-run evaluation is required.

| Candidate | F1 | Critical TTC MAE | Composite |
|---|---:|---:|---:|
| V1 conservative union | 0.654 | 29.993 s | 42.8 |
| V2 event-to-TTC | 0.428 | 30.547 s | 35.8 |

The table is retained only for provenance, not comparison.

## Diagnosis

The old run cannot establish an FSM conclusion because its candidate TTC was
incorrect before FSM handling.

Do not retune IoU, occupancy, floor TTC, or FSM timing from these figures. V1
remains selected until a corrected pre-registered run is scored.
