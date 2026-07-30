# Guardian V1/V2 research conclusion

## Decision

Freeze V1 as the selected baseline. Do not promote the V2 path, association,
or temporal-closing candidates. There is no further low-risk threshold or
filter experiment justified by the current six-trip, framewise-TTC dataset.

V1 result: **danger-F1 0.654**, **composite 42.8 / 100**, and certified
end-to-end compute **P95 65.91 ms** at batch 1 / 640 x 360.

## What was established

| Question | Evidence | Conclusion |
|---|---|---|
| Can direct ego-path geometry separate side traffic? | Provisional blind visual review: on-path offsets averaged 0.82 m vs 11.50 m for adjacent objects. | Yes, as a narrow T01 cue. |
| Does a fixed direct path gate improve the frozen run? | F1 0.6543 to 0.6564; two T01 false-danger frames removed; critical TTC MAE unchanged. | Small, safe diagnostic benefit only. |
| Is hard-IoU association a coverage limit? | 30 of 47 T05 false-danger frames have no direct current hard-IoU YOLO association in the Phase 17 partition. | Yes; downstream path logic cannot see them. |
| Can unique containment plus continuity solve that limit? | T05 F1 fell 0.5091 to 0.4906 and recall fell 0.8000 to 0.7429. | No; reject the candidate. |
| Are associated T05 false alerts caused by noisy depth-rate? | All 13 have positive robust YOLO-box closing; median 3.168 m/s and median fit MAD 0.0098 m. | No; temporal smoothing/EKF cannot safely remove them. |

## Central insight

The scorer supplies a **frame-level minimum TTC**, but it does not identify
the responsible visible object. The system observes several plausible road
users at once. For the unresolved T05 cases, the observed object has stable
range closure and may be on-path, while the frame label remains non-danger.
Without object-event correspondence, the data cannot distinguish:

1. an association to the wrong visible object;
2. a genuinely closing object whose future trajectory does not collide; or
3. a mismatch between a temporally stable safety event and the scorer's
   independent framewise TTC target.

Any additional threshold, corridor, association, or Kalman tuning would
choose between these explanations without labels. That would overfit six trips
rather than produce defensible generalization.

## Count reconciliation

Two counts appear in earlier reports because they answer different questions.
The **30/47** Phase 17 partition is the direct selected-classical-component
association count. The **28/47** value in the V2 framewise report is its
separate full V2-eligibility audit using that run's `v2_event_match_iou` and
occupancy evidence. They must not be presented as the same denominator or
interchanged in a final report.

## What is not concluded

- V1 is not a general collision-warning system for unrestricted turns,
  weather, or traffic.
- A lower F1 does not prove the stereo depth is bad: the T05 closing audit
  independently supports the measured depth trend.
- A small V2 F1 increase is not enough to promote a candidate when it loses
  known true-danger frames.

## Required next phase

Move from rule tuning to evidence acquisition and validation:

1. Label 20–30 representative tracks with object identity, on-path/adjacent/
   diverging relation, occlusion, and closest-point-of-approach (CPA).
2. Add a scenario-stratified external or synthetic holdout focused on turns,
   side traffic, and lead vehicles.
3. Use those labels to validate an object-level ego-compensated state model
   before allowing it to gate V1 events.
4. Keep V1 as the deployment baseline while this evidence is collected.

## Related evidence

- `PATH_ONLY_GATE_RESULT.md`
- `T05_FAILURE_PARTITION.md`
- `CONTAINED_TEMPORAL_PATH_RESULT.md`
- `RELATIVE_CLOSING_AUDIT_RESULT.md`
