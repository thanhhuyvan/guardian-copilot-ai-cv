# Phase 05 handoff — confidence, risk state, and events

Date: 2026-07-28  
Status: **IN PROGRESS**

## Input candidate

Keep classical `track_p35_guarded` as the fallback. A detector-owned research
candidate is now also available; it must not replace the fallback until
confidence-based selection and live detector latency pass their gates.

| Frozen metric | Value |
|---|---:|
| Macro danger-F1 | 0.5634 |
| Composite | 39.71 |
| Critical-TTC MAE | 44.806 s |
| T03 recall | 0.276 |
| T05 false positives | 45 |
| Compute P95 | 54.40 ms |

The detector-owned candidate improved the official six-trip macro F1 to
`0.618` and composite to `42.4`, but T02 regressed from `0.765` to `0.375`.
See `artifacts/DETECTOR_OWNED_TTC_ABLATION.md`. This rules out a universal
replacement and motivates a causal confidence/fusion decision in this phase.

Phase 04B's semantic comparator was rejected because its best oracle F1 was
only 0.5745 and its full LOTO selection was infeasible.

## First implementation slice

1. Define a backend-neutral per-frame confidence/risk record containing frame
   ID, timestamp, selected track, raw and filtered TTC, confidence factors,
   proposal source, risk state, degradation state, and reason codes.
2. Implement a deterministic risk-state machine with `NORMAL`, `ATTENTIVE`,
   `HIGH_RISK`, and `UNKNOWN`.
3. Add hysteresis and debounce without changing the frozen TTC estimator.
4. Unit-test threshold boundaries, low-confidence spikes, missing frames,
   irregular timestamps, trip reset, and event merge behavior.
5. Replay all six practice trips and report event counts, duration, duplicate
   events, missed danger runs, and false-event duration.

## Mini-fold capacity result

An intentional overfit diagnostic was run before implementation:

| Window | Baseline F1 | In-sample F1 | Four-block F1 |
|---|---:|---:|---:|
| T03 frames 280–360 | 0.432 | 0.906 | 0.627 |
| T05 frames 430–580 | 0.247 | 1.000 | 0.426 |

Frame ID, timestamp, trip identity, and ground truth were excluded from model
inputs. The held-out predictions used four contiguous blocks.

Decision:

- T03 has useful causal signal in component count, height/width, depth, and
  closing speed. Implement a small physics-based ablation.
- T05's local features can memorize the episode and suppress many false
  positives, but blocked recall remains unchanged. Add object-centric
  depth/motion measurements instead of a larger classifier.

## Promotion rule

Phase 05 may improve alert stability, but it must not silently relabel the
underlying frame-level benchmark. Report both:

- frozen frame-level danger metrics; and
- event-level metrics after hysteresis/debounce.

The phase exits only when event JSON is deterministic and schema-valid, one
continuous hazard is not fragmented into repeated alerts, and low-confidence
single-frame TTC spikes do not create critical events.
