# Phase 05 handoff — confidence, risk state, and events

Date: 2026-07-28  
Status: **READY_TO_START**

## Input candidate

Use the classical `track_p35_guarded` TTC pipeline. Do not load YOLO26 in the
Phase 05 runtime.

| Frozen metric | Value |
|---|---:|
| Macro danger-F1 | 0.5634 |
| Composite | 39.71 |
| Critical-TTC MAE | 44.806 s |
| T03 recall | 0.276 |
| T05 false positives | 45 |
| Compute P95 | 54.40 ms |

Phase 04B's semantic comparator was rejected because its best oracle F1 was
only 0.5745 and its full LOTO selection was infeasible.

## First implementation slice

1. Define a backend-neutral per-frame confidence/risk record containing frame
   ID, timestamp, selected track, raw and filtered TTC, confidence factors,
   risk state, degradation state, and reason codes.
2. Implement a deterministic risk-state machine with `NORMAL`, `ATTENTIVE`,
   `HIGH_RISK`, and `UNKNOWN`.
3. Add hysteresis and debounce without changing the frozen TTC estimator.
4. Unit-test threshold boundaries, low-confidence spikes, missing frames,
   irregular timestamps, trip reset, and event merge behavior.
5. Replay all six practice trips and report event counts, duration, duplicate
   events, missed danger runs, and false-event duration.

## Promotion rule

Phase 05 may improve alert stability, but it must not silently relabel the
underlying frame-level benchmark. Report both:

- frozen frame-level danger metrics; and
- event-level metrics after hysteresis/debounce.

The phase exits only when event JSON is deterministic and schema-valid, one
continuous hazard is not fragmented into repeated alerts, and low-confidence
single-frame TTC spikes do not create critical events.
