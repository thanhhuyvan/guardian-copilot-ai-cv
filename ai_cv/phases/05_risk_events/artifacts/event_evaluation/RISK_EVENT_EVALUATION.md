# Phase 05 risk-state and event evaluation

Date: 2026-07-28  
Input: parameter-free conservative-union TTC

## Fixed state policy

The selected `recommended` configuration was frozen before evaluation:

- enter `HIGH_RISK` after 3 consecutive danger frames;
- enter after 2 frames for critical TTC below 1.5 s;
- exit after 3 consecutive clear frames;
- enter `ATTENTIVE` after 2 warning-or-higher frames;
- exit `ATTENTIVE` after 4 clear frames;
- merge high-risk event gaps up to 4 frames.

Nominal TTC bands follow the shared contract. Every trip creates a fresh state
machine, and unreliable input produces `UNKNOWN` rather than `NORMAL`.

## Six-trip result

| Event/alert metric | Raw union | Recommended hysteresis | Change |
|---|---:|---:|---:|
| Macro alert-state F1 | 0.658 | **0.669** | +0.011 |
| Macro event recall | 0.778 | 0.778 | 0 |
| Macro event precision | 0.361 | **0.722** | +0.361 |
| Predicted events | 38 | **16** | -22 |
| Event fragmentation | 3 | **0** | -3 |
| False-event duration | 5.60 s | 5.40 s | -0.20 s |

Selected-policy detail:

| Trip | Alert F1 | Event recall | Event precision | False duration |
|---|---:|---:|---:|---:|
| T01 | 0.311 | 0.500 | 0.250 | 1.40 s |
| T02 | 0.765 | 1.000 | 1.000 | 0.15 s |
| T03 | 0.820 | 0.500 | 0.333 | 0.35 s |
| T04 | 0.764 | 1.000 | 1.000 | 0.80 s |
| T05 | 0.523 | 1.000 | 0.750 | 2.35 s |
| T06 | 0.835 | 0.667 | 1.000 | 0.35 s |

T03 event recall is unchanged and its alert-state recall rises from `0.759`
to `0.862`. T05 retains all three truth events and reduces false event count
from three to one, but its false duration remains `2.35 s`.

## Interpretation

Hysteresis solves flicker and fragmentation. It does not solve the sustained
T01/T05 false-positive episodes. Those runs persist long enough to pass any
safe debounce setting; suppressing them with a longer delay would also delay
or miss real T03/T05 hazards.

This confirms the earlier diagnosis: the remaining error requires new
environment evidence, corrected/expanded annotations, or a feature that
distinguishes path relevance. A larger adaptive router or dynamic CNN is not
supported by the current 124 disagreement examples.

## Contracts and determinism

- All generated events validate against `risk_event.v1.schema.json`.
- Repeating the complete evaluation produced identical SHA-256 hashes for all
  six event JSON files.
- One continuous high-risk run is not fragmented into repeated events.
- Single-frame danger spikes cannot create a high-risk event.
- `clip_path` remains `null` until clip export is connected.

## Latency

The state machine costs approximately `0.0014 ms/frame`. Reusing the existing
classical component extraction and tracking adds an independently measured
`5.30 ms` P95. Combining that with the live detector-owned `63.22 ms` P95 gives
a conservative planning estimate of **68.52 ms P95**, below the 75 ms target.
This sum is a budget estimate, not a replacement for a final integrated timing
run.

## Decision

Promote the deterministic state/event implementation for Phase 05 integration,
but do not claim that hysteresis fixes sustained perception errors. Preserve
raw classical, detector, and union TTC in evaluation outputs. The next accuracy
work requires additional independent data rather than more tuning on these six
trips.
