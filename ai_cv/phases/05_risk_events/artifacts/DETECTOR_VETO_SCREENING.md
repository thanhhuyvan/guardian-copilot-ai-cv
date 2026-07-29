# Detector-veto screening — rejected

## Offline result

For each classical TTC danger frame, compare detector-owned TTC without
changing any prediction. Across six trips, detector would veto 39/89 classical
false alerts but also 21/123 classical true-danger frames. That loss is too
large for a generic safety veto.

T05 is decisive: detector agrees with 40/45 classical false alerts and only
vetoes 5. Thus semantic object detection does not resolve the main T05 error.

## Inference

The T05 alert commonly concerns a real, detected road user. Both classical and
YOLO-box depth paths produce an early TTC. Association and detector presence
are not sufficient explanations; shared temporal relative-motion estimation is
the next causal hypothesis.

## Decision

Do not add a detector veto. Next experiment must assess a continuous filtered
relative state/TTC estimate, with finite TTC retained for MAE and no binary
reject/fallback behavior.
