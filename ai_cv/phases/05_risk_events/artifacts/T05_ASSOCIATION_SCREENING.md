# T05 association screening — hard IoU falsified

## Protocol

Corrected Phase 17 evidence supplied all classical-source V1 danger frames
whose current classical-to-YOLO IoU was below `0.30`. This yielded 28 false
alerts and 13 true-danger candidate pairs across 40 rendered frames (one frame
contained a second, lower-ranked proposal). No runtime TTC, EKF, or risk gate
was changed.

## Provisional visual review

Visual inspection of the overlays found the top-ranked YOLO proposal is the
same visible road user as the broad classical component in all 40 frames. The
extra rank-two proposal in frame 484 is unrelated.

This is **provisional visual evidence**, not target-ID ground truth. It is
sufficient to falsify the idea that low IoU means different objects in this
T05 sequence; it is not sufficient to choose production association values.

## Key finding

Hard IoU is structurally wrong for this component representation. Classical
boxes cover the target plus road/background, while YOLO boxes cover the target
tightly. Correct same-object pairs have IoU only `0.076`–`0.283`.

Depth agreement is also unsafe as a hard association condition: visually
correct car pairs differ by up to about `8.9 m`, because classical component
depth is contaminated by its broad support.

## Consequence

Next shadow matcher should use semantic containment/relative position and
temporal persistence to propose identity. It must **not** require component
depth agreement. YOLO-box median disparity remains the appropriate object-depth
measurement after association.

Before a TTC experiment, validate this rule on stratified unmatched frames from
all six trips and report same-object precision separately for false alerts and
true-danger anchors. V1 remains unchanged.
