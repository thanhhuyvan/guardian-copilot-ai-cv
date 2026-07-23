# Dataset Audit Findings

## Audit decision

**Gate S1.1 passes.** The structural audit covered all 16 expected trips and found
no blocking data-integrity errors. The run inspected 21,600 frame records, loaded
all 4,320 depth keyframes, and decoded all 64,800 road/driver JPEG files.

**Gate S1.2 passes at Stage 1 scope.** Calibration values and stereo pair counts
are consistent. Sampled left/right disparity diagnostics showed no gross
registration error. Exhaustive stereo confidence and sub-pixel error analysis is
carried into Stage 2A rather than treated as proven.

## Confirmed properties

- Practice trips contain 600 frames and 120 depth keyframes each.
- Scored trips contain 1,800 frames and 360 depth keyframes each.
- Frame IDs are contiguous and timestamps are strictly monotonic at exactly
  0.05 seconds per frame.
- All 64,800 image files decode successfully. The recorded left road-image size
  is consistently 640 x 360 and the format is JPEG.
- Calibration is constant across all trips: `fx ~= 320 px`, `baseline = 0.3 m`.
- Scored trips contain no finite TTC ground truth, as required by redaction.
- Deployment telemetry exists on every frame. Target metadata and event
  annotations are also present in scored JSON, so they must not accidentally be
  used as unavailable model inputs unless the competition contract explicitly
  permits them.

## Practice ground-truth coverage

Across 3,600 practice frames:

| Measure | Frames | Share |
|---|---:|---:|
| Finite TTC ground truth | 602 | 16.72% |
| TTC < 3.0 s | 283 | 7.86% |
| TTC < 2.0 s | 204 | 5.67% |
| TTC < 1.5 s | 136 | 3.78% |

There are 13 danger episodes at both the 3-second and 2-second thresholds.
Coverage is imbalanced: T04-Sample and T06-Sample contribute 173 of the 283
frames below 3 seconds (61.13%). Evaluation and tuning must therefore report
per-trip results in addition to aggregate scores.

## Anomalies and required handling

### T01d depth dropout

T01d has 117 all-zero depth keyframes from frame 1215 through frame 1795 at the
5-frame keyframe cadence. This affects 32.5% of its depth keyframes and covers
approximately 60.75-89.75 seconds.

Required policy:

- Treat an all-zero depth map as unavailable, never as a valid zero-distance
  obstacle.
- Do not interpolate across this long gap.
- Fall back to a degraded monocular/stereo or no-prediction state and lower the
  confidence explicitly.
- Add this interval as a mandatory robustness test case.

### Depth sentinel values

Depth maps use values near 1000 m as invalid/far-plane sentinel data. The audit
uses `depth < 999 m` as the validity condition. Sentinel prevalence varies
substantially by trip, so depth statistics must be computed only on valid pixels.

### Scored metadata description mismatch

All ten scored trip descriptions say `30s`, while timestamps and metadata
duration both indicate 90 seconds. This is documentation-only metadata drift.
Use frame timestamps and the numeric duration field; do not truncate scored
trips based on the free-text description.

## Next actions

1. Visually verify stereo alignment on representative and difficult frames.
2. Define the causal input allowlist before baseline experiments.
3. Run and score the official baseline unchanged.
4. Add explicit degraded-mode tests for the T01d depth dropout and sentinel-heavy
   maps.

Raw evidence is in `dataset_audit.json`; compact tables are in the CSV artifacts.
