# T05 false-danger partition

This audit reads the frozen Phase 17 T05 evidence and changes no TTC, event,
association, or detector output. A false-danger frame is a frozen union TTC
below 2 s with challenge truth at or above 2 s.

| Partition | Frames | Meaning |
|---|---:|---|
| Unassociated | 30 / 47 | The selected classical component has no current YOLO match at hard IoU >= 0.30. Path geometry cannot see these frames. |
| On-path, non-closing | 13 / 47 | A YOLO match and direct path geometry are available; the target is in the corridor but its raw classical depth-rate spuriously predicts TTC < 2 s. |
| Geometry unavailable | 4 / 47 | A match exists but valid direct host-path geometry is unavailable. |
| Off-path | 0 / 47 | No false-danger frame is currently suppressible by the hard-IoU path gate. |

## Consequence

The direct path gate correctly leaves T05 unchanged. The next work must stay
separate:

1. validate a multi-cue association proposal before allowing the 30
   unassociated frames to use semantic/path information; and
2. investigate ego-compensated relative closing-rate reliability for the 13
   associated, on-path frames.

Neither conclusion licenses a threshold sweep. The hard-IoU association is a
coverage boundary, while the on-path group is a state-estimation problem.
