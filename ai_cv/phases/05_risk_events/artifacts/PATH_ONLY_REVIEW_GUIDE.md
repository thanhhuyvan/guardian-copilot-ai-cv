# Path-only blind review guide

This review checks the geometric state, not the framewise TTC score.

## Material

Use the 11-frame clips and three-frame contact sheets in the review pack. Do
not open the shadow-offset CSV until all labels are complete. The label CSV
intentionally contains no offset, TTC prediction, F1 result, or occupancy
value.

## Per-row label

Set `path_relation` to exactly one value:

- `on_path`: the marked object plausibly occupies the host vehicle's immediate
  travel path.
- `adjacent`: the object is near the road but remains in a neighbouring lane,
  shoulder, or roadside region.
- `diverging`: the marked object is visibly moving away from, or crossing out
  of, the host path.

Set `occluded` to `yes`, `no`, or `unknown`. Leave `path_relation` blank when
the clip cannot support a judgement. Do not infer a label from the challenge
TTC target: this is a target-path review, not a danger-label review.

## Decision rule

The auditor will report every completed label and the corresponding direct
path offset. It will not optimise a corridor threshold. The review supports a
later risk-gate experiment only if visually `on_path` objects have offsets
that are consistently closer to zero than `adjacent`/`diverging` objects.
