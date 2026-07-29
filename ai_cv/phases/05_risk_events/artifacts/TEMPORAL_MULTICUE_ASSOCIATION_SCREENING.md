# Temporal multi-cue association screening — identity coverage is not risk repair

## Fixed offline protocol

Using the corrected six-trip Phase 17 evidence, rank each classical-source
danger component against current YOLO stereo tracks by:

1. whether the YOLO-box centre lies inside the component;
2. whether its track ID is the top-ranked ID in the immediately prior frame;
3. normalized component-to-box centre distance; and then
4. IoU as a tie-breaker.

Component depth is deliberately absent: previous visual review showed that a
correct car can have an approximately 8.9 m component-vs-YOLO depth error.
This is a diagnostic ranking only. It changes no match contract, TTC, FSM,
submission, or reported organizer metric.

## Result

| Population | Classical dangers | Ranked candidate | Top candidate contained | Top ID continuous |
|---|---:|---:|---:|---:|
| All | 212 | 202 | 196 | 158 |
| True danger | 123 | 121 | 118 | 93 |
| False alert | 89 | 81 | 78 | 65 |
| T05 false alert | 45 | 45 | 45 | 42 |
| T05 true danger | 12 | 12 | 12 | 8 |

The ranker improves *candidate availability*: 202/212 classical danger frames
have a deterministic geometric candidate. It does **not** identify which
candidate explains the classical component. In the dominant T05 failure set,
the top candidate is contained in every false alert and persists across 42/45
consecutive false-alert frames. Continuity therefore carries the false alarm
forward rather than separating it from real danger.

## Decision

Do not insert this ranker into V2 TTC or run an official F1/MAE experiment.
That would only replace a known hard-IoU miss with an unvalidated—and in T05,
stably wrong—identity decision. The key upstream problem is not missing YOLO
object identity: most T05 false alerts already refer to a persistent visible
road user. Their depth/relative-motion TTC is wrong.

The next scientifically useful association step requires independent object
identity labels or image-motion measurements for the classical component. It
cannot be resolved from containment, box geometry, and YOLO track persistence
alone. V1 remains the official baseline.
