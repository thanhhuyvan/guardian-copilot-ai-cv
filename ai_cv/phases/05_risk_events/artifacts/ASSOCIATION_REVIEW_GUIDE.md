# Blind association review

The magenta rectangle is the classical stereo component. The yellow rectangle
is the proposed YOLO stereo track selected by deterministic containment and
continuity cues. For each row, label `same_object` as `yes`, `no`, or
`uncertain` based only on the overlay and nearby clip.

Do not use TTC, false-alert/true-danger status, IoU, depth difference, or F1.
The review asks only whether the two boxes refer to the same physical road
user. A `yes` validates identity coverage; it does not imply that the road
user is dangerous or on the host path.

The blind sample contains both true-danger and false-alert cases from every
trip. It must be audited before multi-cue association may affect a risk gate.
