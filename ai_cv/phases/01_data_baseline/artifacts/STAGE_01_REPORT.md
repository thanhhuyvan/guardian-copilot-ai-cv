# Stage 01 Final Report — Dataset and Baseline

**Status:** COMPLETE  
**Decision:** proceed to Stage 2A, classical object-centric TTC  
**Reference baseline:** 19.7/100 mean composite  
**Best lightweight experiment:** 32.7/100, research-only

## 1. Executive conclusion

Stage 1 established a trustworthy measurement floor and identified why the
organizer baseline fails. It did not prove that classical stereo is inadequate;
it proved that the specific combination of fixed ROI, mixed-scene median depth
and five-frame differentiation is inadequate.

The next step is therefore not to adopt deep learning by default. Stage 2A will
first test a classical object-centric pipeline:

```text
stereo confidence
  -> ground/road separation
  -> obstacle components or Stixels
  -> temporal association
  -> robust/Kalman 3D state
  -> ego-motion compensation
  -> collision corridor
  -> TTC + confidence/degraded state
```

Detection or semantic segmentation becomes a Stage 2B augmentation only when a
measured Stage 2A failure requires semantic class, better instance separation,
or adverse-scene robustness.

## 2. Dataset audit

- Audited 16 trips, 21,600 frames and 4,320 depth keyframes.
- Decoded all 64,800 left/right/driver images.
- Found no blocking structural errors and 11 documented warnings.
- Frame IDs are contiguous; timestamps are monotonic at 0.05 s.
- Stereo images are consistently 640 x 360; calibration uses `fx ≈ 320 px`
  and a 0.30 m baseline.
- Scored trips contain no finite TTC ground truth.

The main robustness case is `T01d`: 117 consecutive depth keyframes are all zero
from frame 1215 through 1795. Zero maps must be treated as unavailable, never as
zero-distance obstacles, and must not be interpolated over the long gap.

Practice GT is imbalanced. Only 283/3,600 frames have TTC below 3 s, and T04 plus
T06 contain 61.13% of them. Every later experiment must report per-trip and
worst-trip results, not only an aggregate.

## 3. Reproduced official baseline

The unmodified organizer pipeline is:

```text
SGBM -> fixed ROI -> median depth -> 5-frame OLS slope -> TTC
```

Strictly validated results on all six practice trips:

| Metric | Result |
|---|---:|
| Critical-zone MAE | 38.046 s |
| Inverse-TTC MAE | 0.2982 |
| Danger F1 | 0.220 |
| Mean composite | 19.7 / 100 |
| Worst trip | T03: 5.0 / 100 |
| Effective throughput | 18.28 FPS |

The README example of approximately 52.3 is not compatible with the supplied
GT/evaluator combination. Its T01 example claims 62 critical frames, while the
current evaluator finds 12. The frozen regression target is therefore 19.7, not
52.3.

The predictor produced 80 true positives, 436 false positives and 124 false
negatives at the 2-second danger threshold. It also returned `inf` on 106 of 283
frames in the 3-second critical zone.

## 4. Failure explanation

The evidence supports five root causes:

1. A single ROI mixes road, background and multiple obstacles.
2. Median depth suppresses small but dangerous targets.
3. Differentiating noisy depth over roughly 0.2 s creates impossible velocities.
4. There is no obstacle identity, ego-motion separation or collision-path test.
5. Weak disparity acceptance has no explicit uncertainty/degraded output.

Six exact FN/FP/TP cases are frozen in `BASELINE_FAILURE_CATALOG.md`; local
visualizations show the corresponding frames, disparity and temporal state.

## 5. Lightweight classical experiment

An 11-frame Theil-Sen trend, physical speed gate and alternate corridor were
tested without learned models:

| Variant | Composite | Worst trip | F1 | inv-TTC MAE |
|---|---:|---:|---:|---:|
| Official replay | 19.7 | 5.0 | 0.220 | 0.2982 |
| Robust median | 32.2 | 16.9 | 0.258 | 0.1896 |
| Robust corridor | 32.7 | 22.1 | 0.280 | 0.1922 |

This confirms that temporal robustness matters. It is not promoted as the new
baseline because:

- The same six trips were used for selection and evaluation.
- T04 regresses and T05 still has zero danger true positives.
- A hard 20 m/s closing-speed limit can reject legitimate motion.
- Eleven frames introduce about 0.5 s of history.
- Percentile and ROI choices remain scene heuristics.

The official 19.7 result remains the regression floor; 32.7 is the first target
that Stage 2A should beat.

## 6. Robustness and latency decision

Robustness is a first-class Stage 2 criterion even though it is not a separate
organizer score:

- degraded input must lower confidence instead of silently returning “safe”;
- `T01d` zero-depth dropout is a mandatory test;
- both FP and FN must be reviewed per trip/event;
- collision decisions must be causal in stream mode;
- post-trip smoothing must be labeled separately from causal results.

Latency was measured only as end-to-end wall time in Stage 1: 18.28 FPS versus a
20 FPS input stream. This is not a hard product SLA because the target product is
primarily out-car/post-trip. P50/P95/P99 instrumentation is deferred until
Stage 2 candidates exist, where it can guide a real quality/compute trade-off.

## 7. Stage 2A entry criteria and goals

Stage 2A starts with no deep-learning dependency. A candidate is promotable when
it:

- uses only current/past deployable inputs in causal mode;
- beats 32.7 mean composite or gives a clearly superior robustness trade-off;
- exceeds the 22.1 lightweight worst-trip score;
- restores non-zero danger true positives on T05;
- outputs obstacle-level depth/TTC confidence or an explicit degraded state;
- handles ground pixels, invalid disparity and the `T01d` dropout explicitly;
- reports quality together with throughput and percentile latency.

The experiment order is:

1. SGBM left-right consistency, confidence and invalid-pixel policy.
2. Ground-plane/V-disparity removal.
3. Disparity obstacle components or Stixel-lite representation.
4. Temporal association plus robust/Kalman state.
5. Ego telemetry compensation and collision corridor.
6. Optical expansion/flow fallback.
7. Only then compare semantic segmentation, object detection or learned depth
   against the strongest classical candidate.

## 8. Reproduction and verification

Full baseline run, normalization, strict validation and evaluation:

```powershell
python ai_cv/phases/01_data_baseline/src/run_baseline.py
```

Validate existing normalized predictions and rerun only evaluation:

```powershell
python ai_cv/phases/01_data_baseline/src/run_baseline.py --reuse-predictions
```

Run Stage 1 tests:

```powershell
python -m unittest discover `
  -s ai_cv/phases/01_data_baseline/tests `
  -p "test_*.py" -v
```

Run the frozen regression gate:

```powershell
python ai_cv/phases/01_data_baseline/verify/verify_phase01.py
```

At closure, all 18 tests pass and all six baseline CSVs pass strict validation.
The strict validator also rejects UTF-8 BOM because the organizer evaluator
misreads a BOM-prefixed `frame_id` header.

## 9. Deliberate deferrals

The following are not hidden unfinished work; they move to Stage 2 where they
become meaningful:

- exhaustive 10+ case catalog: six deeply diagnosed cases already cover the
  architectural failures;
- per-frame P50/P95/P99: requires candidate-level instrumentation;
- exhaustive pixel-level stereo calibration: sampled diagnostics show no gross
  registration error, while confidence/error behavior is a Stage 2A study;
- final detector/semantic choice: must be justified by a measured classical
  limitation rather than selected in advance.
