# V2 pre-registered validation

## Fixed contracts

- V1 remains untouched and is the comparison candidate.
- EKF noise comes from shadow residual measurement, never F1 search.
- EKF innovation is not a binary TTC reject switch.
- Longitudinal TTC remains finite for MAE reporting whenever a valid track exists.
- Path occupancy is Gaussian probability from EKF lateral mean/covariance at
  CPA, not a learned or hand-ranked confidence score.
- Existing risk-event FSM supplies temporal hysteresis; no second debounce is added.

## Pass criteria

1. Track-level: labelled path relation/CPA checks must show that low occupancy
   corresponds to non-ego-path tracks.
2. Frame level: macro danger-F1 must not be below V1 (`.654`), critical TTC
   MAE must not exceed V1 (`29.993 s`), and T02/T05 must not regress.
3. Runtime: compute P95 remains at or below `75 ms`.

The organizer derives danger-F1 from submitted TTC, whereas the deployment
event FSM can consume path occupancy separately. Both outputs are reported;
path-event gains alone do not count as an organizer-score improvement.
