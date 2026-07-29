# Cross-trip visual failure taxonomy — provisional

## Review set

Thirty blind 11-frame clips were rendered from the corrected Phase 17
evidence: five T01 and T05 known-failure examples, plus five organizer
critical-TTC anchors from each remaining trip. This is visual diagnosis only;
the organizer supplies frame-level minimum TTC, not object identity or
path/CPA ground truth.

## Observations

| Group | Visual pattern | Inference |
|---|---|---|
| T02/T03/T04/T06 critical anchors | Leading car or motorcycle grows along the visible host lane | Expected longitudinal closing case |
| T01 known failures | Urban turn/cross-street traffic and lateral pedestrians | Camera-depth closing is insufficient; ego-path relation matters |
| T05 frames 263/343 | Pedestrian beside the road corridor | A lateral/occupancy exclusion is needed |
| T05 frames 519/599 | Centered leading car remains visible | Object identity can be correct while TTC is false; relative speed is required |

## Key result

There is no single “T05 problem.” At least two mechanisms coexist:

1. **off-path road users** — needs ego-path/CPA reasoning; and
2. **in-path but non-closing lead vehicle** — needs reliable ego-compensated
   relative velocity, not another detector or association rule.

The rejected temporal-regression TTC experiment shows that smoothing existing
classical depth alone does not solve mechanism 2. The prior V2 path gate was
also not valid evidence for mechanism 1 because its association/occupancy
input was not independently validated.

## Constraint

Current challenge labels identify only the minimum TTC per frame. They do not
say which object caused it, whether a selected detection is on the host path,
or its CPA. Therefore neither a path gate nor object TTC can be claimed robust
from frame F1 alone. The 30-row review CSV remains the needed validation
instrument; ambiguous clips must stay unlabeled rather than be converted into
training or threshold data.

## Decision

Freeze V1 performance. Further meaningful research requires either:

1. manually verified object/path/CPA labels on the review pack, then a
   carefully validated ego-path relative-state model; or
2. an external/synthetic dataset with object-level trajectories and ego pose.

Do not run further detector, association, depth-smoothing, or F1-threshold
ablations on the six-trip data alone.
