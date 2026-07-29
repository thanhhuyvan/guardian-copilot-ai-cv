# Contained-continuous path diagnostic

This experiment extends the fixed direct path gate only when a classical
danger component contains exactly one current YOLO box centre and that same
YOLO track continues from the immediately prior classical-danger frame. All
other frames fail open to frozen V1 TTC.

## Six-trip result

| Policy | Overall F1 | Critical TTC MAE | T05 F1 | T05 recall |
|---|---:|---:|---:|---:|
| Frozen V1 reproduction | 0.6543 | 29.993 s | 0.5091 | 0.8000 |
| Contained-continuous path diagnostic | 0.6569 | 29.994 s | 0.4906 | 0.7429 |

The candidate suppresses four T05 false-danger frames, but also suppresses
two true-danger frames. Its small overall F1 increase is therefore not a
valid win: it violates the pre-registered rule that T05 must not decline.

## Decision

**Rejected.** Unique containment plus one-frame continuity improves coverage
but does not make the direct path estimate safe enough to gate T05 events.
The association review showed strong apparent identity agreement, so the
remaining issue is not simply hard-IoU coverage: at least some true-danger
events receive a wrong path-state estimate. Do not tune the corridor or relax
continuity to rescue this candidate.

The next isolated target is the 13 associated, on-path/non-closing T05 false
alerts: test ego-compensated relative closing-rate reliability while leaving
path relation and association unchanged.
