# TTC Method Research Strategy

## Goal

Select a TTC method using measurable evidence on the six practice trips while preserving a causal, deployment-neutral output contract.

## Candidate families

| ID | Method | Strength | Primary risk | Role |
|---|---|---|---|---|
| M0 | Fixed ROI + stereo median + finite difference | Reproducible baseline | Mixes road/background/objects | Reference only |
| M1 | Detector + target ROI stereo + per-track regression | Uses metric stereo and target identity | Detector/tracker/depth errors compound | Primary candidate |
| M2 | Detector + optical expansion/scale TTC | Can estimate TTC without metric depth | Sensitive to lateral motion and box jitter | Complement/fallback |
| M3 | Detector + monocular depth + tracking | Works if stereo degrades | Scale drift/domain shift | Fallback/ablation |
| M4 | Stereo/keyframe depth fusion + temporal filter | Uses provided depth signal | Must avoid leakage and over-reliance | Calibration/validation candidate |
| M5 | Learned temporal TTC regressor | Can model non-linear motion | Too little labeled training diversity | Stretch goal |

## Recommended research order

1. Reproduce M0 and freeze its metrics.
2. Audit whether depth keyframes are valid inputs, declare their manifest policy and quantify their quality.
3. Build M1 with simple tracking and robust target depth.
4. Add M2 as a second TTC cue and disagreement detector.
5. Evaluate M3 only if stereo is a dominant failure mode.
6. Consider M5 only after deterministic candidates are understood.

## Controlled evaluation

- Use the same six practice trips and official evaluator for every candidate.
- Report mean and per-trip composite, critical MAE, danger F1 and inverse-TTC MAE.
- Track false positives, missed danger frames, TTC jitter and detection delay.
- Record P50/P95 latency, throughput and hardware for every promoted implementation.
- Store a run manifest with code commit, model/config versions, dataset and processing mode.
- In `causal_online`, use neither future frames nor the future schedule in `events_log`.
- Change one major factor per experiment where possible.

## Promotion gate

A candidate advances only if it improves official composite or a documented critical failure mode without unacceptable worst-trip, false-alarm or latency regression.

## Initial decision

M1 is the default engineering direction because it directly fixes the baseline's largest conceptual error: estimating one scalar depth from a fixed image region rather than estimating motion for the collision-relevant target.
