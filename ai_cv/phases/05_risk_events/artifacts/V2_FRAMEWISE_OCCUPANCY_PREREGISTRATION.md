# V2 framewise occupancy ablation — pre-registration

## Motivation

The first V2 conversion was rejected: `RiskStateMachine` suppressed many
organizer-scored TTC frames, including true danger frames. This ablation tests
only whether that conversion layer—not any geometry parameter—caused the loss.

## Frozen policy

Start with the frozen V1 conservative-union TTC on every frame. Only when all
conditions already fixed in the rejected run hold—classical source, raw TTC
below `2.0 s`, classical-to-YOLO IoU at least `0.30`, accepted EKF update, and
corridor-occupancy probability below `0.50`—replace the submitted TTC by
`2.0 s`. Otherwise preserve raw TTC exactly.

There is **no FSM**, no new hysteresis, no altered IoU/occupancy threshold, and
no change to the TTC value for MAE except the same finite `2.0 s` suppression.
This is an organizer-metric diagnostic, not a deployable event policy.

## One-run protocol

Run all six untouched trips once with the fixed command flag
`--experimental-v2-event-framewise`; score with the organizer evaluator;
compare against the recorded same-run V1 score. Do not repeat, sweep, or tune
after observing the result.

## Interpretation

- If F1 improves or holds while critical TTC MAE does not regress, the prior
  loss was predominantly the event-to-frame conversion.
- If F1 or MAE still fails, the current association/occupancy coverage is the
  limiting factor; stop gate work and repair measurement/data association.
- This ablation cannot promote V2 to deployment regardless of score because it
  deliberately removes deployment temporal hysteresis.
