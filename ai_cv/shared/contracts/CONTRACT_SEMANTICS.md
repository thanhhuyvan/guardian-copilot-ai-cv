# Contract Semantics

## TTC and motion

- `timestamp` is seconds from trip start.
- `distance_m` is non-negative metric distance to the tracked target.
- Positive `closing_speed_mps` means the target distance is decreasing.
- For metric TTC (`distance / closing speed`), finite TTC requires positive closing speed.
- Direct image-space TTC methods may produce finite `ttc_sec` while leaving `distance_m` and `closing_speed_mps` null.
- TTC may be zero at collision/contact. JSON uses `null` when TTC is not finite/reliable; submission CSV serializes that state as `inf`.
- Frame `min_ttc_sec` is the minimum finite TTC among targets marked `in_collision_corridor=true`.

## Nominal risk thresholds

- `SAFE`: no finite TTC or TTC >= 3.0 s.
- `WARNING`: 2.0 s <= TTC < 3.0 s.
- `DANGER`: 1.5 s <= TTC < 2.0 s.
- `CRITICAL`: TTC < 1.5 s.
- `UNKNOWN`: sensor/input failure prevents a reliable state.
- These bands are state-machine triggers, not a required stateless equality between one frame's TTC and `risk_level`. Debounce, hysteresis and quality gates are finalized in Phase 05.

## Confidence and quality

- `detection_confidence` is the detector-provided score and may require calibration.
- `ttc_quality` is an uncalibrated engineering quality score combining detection, depth and tracking evidence.
- `perception_quality` aggregates frame-level evidence and is not a probability.
- `event_quality` aggregates quality across an event and is not a probability.
- `confidence_level` is a product-facing ordinal label. Its mapping is finalized and versioned in Phase 05; Stage 00 does not invent numeric cutoffs without calibration evidence.

## Status

- `valid`: normal processing; `degraded_reasons` is empty.
- `degraded`: output exists with reduced evidence; at least one reason is required.
- `unknown`: no reliable perception state; objects are empty, TTC is null, risk is UNKNOWN and quality is zero.

## Bounding boxes

- Format is pixel `[x1, y1, x2, y2]`.
- Semantic validation requires `0 <= x1 < x2 <= image_width` and `0 <= y1 < y2 <= image_height`.

## Class mapping

Dataset labels are intentionally mapped to coarse contract classes. `bike` is not assumed to mean motorcycle or bicycle until detector evidence refines it.
