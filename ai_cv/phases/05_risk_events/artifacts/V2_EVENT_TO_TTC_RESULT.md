# V2 event-to-TTC result: rejected

The pre-registered event-to-TTC experiment was run once on all six trips with
the fixed IoU `0.30`, occupancy `0.50`, finite floor `2.0 s`, and existing
`RiskStateMachine`.

| Candidate | F1 | Critical TTC MAE | Composite |
|---|---:|---:|---:|
| V1 conservative union | 0.654 | 29.993 s | 42.8 |
| V2 event-to-TTC | 0.428 | 30.547 s | 35.8 |

All pre-registered gates fail. Per-trip V2 F1 is T01 `0.000`, T02 `0.273`,
T03 `0.711`, T04 `0.651`, T05 `0.396`, T06 `0.539`.

## Diagnosis

The existing event FSM suppresses short genuine danger intervals. This happens
even where the low-occupancy gate did not suppress a track: raw/V2 danger frame
counts were T01 `38/7`, T02 `19/4`, T03 `33/16`, T04 `53/34`, T05 `75/56`, and
T06 `57/29`.

Therefore event hysteresis is suitable for deployment alerts but cannot be
converted directly to the organizer's framewise TTC submission metric. Do not
retune IoU, occupancy, floor TTC, or FSM timing on these six trips. V1 remains
the selected official candidate.
