# Global experiment plan: from V1 baseline to validated V2

## Goal

Improve collision-risk reasoning only when an experiment demonstrates object-
level correctness, framewise score safety, and generalization beyond the six
practice trips. V1 remains frozen unless a candidate passes every gate.

## Current baseline

| Item | Frozen value |
|---|---:|
| Danger-F1 | 0.654 |
| Composite | 42.8 / 100 |
| Critical TTC MAE | 29.993 s |
| Compute latency P95 | 65.91 ms |

## Experiment sequence

| Phase | Question | Method | Required evidence | Exit condition |
|---|---|---|---|---|
| 0. Freeze | Is comparison reproducible? | Retain V1 code, score, latency, and evidence hashes. | Existing V1 report. | Completed. |
| 1. Correspondence labels | Which visible object owns an event? | Blind-label object identity, path relation, occlusion, target trajectory, and CPA. | 30–60 stratified tracks across all trips. | Sufficient inter-review agreement and scenario coverage. |
| 2. State validation | Is object state geometrically correct? | Shadow-only ego-compensated object state and CPA; no risk gating. | Track-level position/path/CPA error against Phase 1 labels. | State passes before any F1 test. |
| 3. Isolated mechanism tests | Which cue fixes each failure family? | Test path occupancy, association, and closing state one at a time. | Pre-registered labels plus six-trip F1/MAE. | No T01/T05 regression; MAE does not regress. |
| 4. External generalization | Does the mechanism work outside six trips? | Scenario-stratified external or synthetic turn/side-traffic holdout. | Object/event-compatible labels. | No material degradation versus labelled internal data. |
| 5. Selection | Is any candidate safer than V1? | Compare V1 and candidates using fixed gates. | Labels, holdout, F1/MAE, latency, determinism. | Lowest-latency all-gate pass wins. |
| 6. Deployment | Can it be shipped reproducibly? | ONNX/TensorRT only after accuracy selection; rerun latency certification. | P95, VRAM, parity, failure handling. | Deployment configuration frozen. |

## Phase 1 label contract

One row represents one visible object within a short frame window. The reviewer
must not see predicted TTC, F1, candidate name, or challenge target while
labelling.

Required fields:

1. `object_id_window`: persistent reviewer ID within the clip.
2. `event_owner`: `yes`, `no`, or `uncertain` — whether the object is the
   plausible owner of the safety-relevant interaction.
3. `path_relation`: `on_path`, `adjacent`, `crossing`, `diverging`, or
   `uncertain`.
4. `relative_motion`: `closing`, `steady`, `opening`, or `uncertain`.
5. `cpa_distance_m`: approximate closest approach, or `unknown`.
6. `occluded`: `yes`, `no`, or `unknown`.
7. `review_confidence`: `high`, `medium`, or `low`.

The labels validate state and correspondence; they must never be used as
training data or for threshold search.

## Fixed selection gates

A candidate can advance only if all apply:

- it improves or preserves the labelled object-state metric;
- its six-trip F1 does not reduce T01 or T05 below V1;
- critical TTC MAE does not exceed V1;
- it has no new NaN, missing-frame, or cross-trip state contamination;
- it passes the external/synthetic scenario holdout;
- its compute P95 remains at or below 75 ms (or is recertified if the selected
  deployment target changes).

## Stop rule

Stop method development after two independently validated object-level
mechanism classes fail on both labelled internal tracks and the external
holdout. At that point the limitation is sensors/ODD/labels, not untested
thresholds. Report the scoped V1 baseline rather than making a general claim.
