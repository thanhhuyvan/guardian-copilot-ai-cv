# Fold performance versus driving environment

Date: 2026-07-28  
Source: original six-trip practice dataset and fresh guarded-baseline rerun

## Quantitative comparison

Image statistics use all 600 left-camera frames per trip. Darkness is the
fraction of grayscale pixels below intensity 40. Speed and lateral acceleration
come from the original trip metadata.

| Trip | F1 | Scenario | Mean luma | Dark pixels | Mean speed | P95 speed | Mean abs. lateral accel. | GT-danger frames | Targets/frame |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| T01 | 0.452 | Day urban pedestrian jaywalk | 117.7 | 12.0% | 25.3 km/h | 29.1 | 0.42 | 10 | 11.88 |
| T02 | 0.765 | Evening highway motorcycle cut-in | 11.7 | 91.6% | 23.3 km/h | 29.1 | 0.15 | 18 | 7.55 |
| T03 | 0.333 | Night highway, heavy rain, lead braking | 11.1 | 93.9% | 39.2 km/h | 87.0 | 0.43 | 29 | 0.96 |
| T04 | 0.763 | Day mixed road, stopped vehicle | 127.9 | 6.8% | 10.1 km/h | 29.0 | 0.40 | 52 | 3.00 |
| T05 | 0.261 | Bright rural, jaywalk and lead braking | 151.7 | 17.4% | 13.6 km/h | 29.1 | 0.02 | 35 | 1.61 |
| T06 | 0.807 | Moderate rain, cut-in and stopped vehicle | 110.5 | 22.2% | 18.4 km/h | 48.3 | 0.03 | 60 | 11.35 |

## What distinguishes successful folds

T02, T04, and T06 have different lighting and weather, so brightness alone
does not explain success. Their useful common properties are:

- a coherent target remains visible over a sustained danger interval;
- the target is large enough or centered enough for stable stereo support;
- relative motion changes more smoothly;
- danger evidence persists long enough for the tracker to confirm it.

T02 is the strongest counterexample to a darkness-only hypothesis. Its mean
luma is `11.7`, almost identical to T03's `11.1`, yet its F1 is `0.765`.
T02 has no rain, lower speed, lower lateral acceleration, and a simpler,
well-localized cut-in event.

## Why the weak folds fail

### T03 — perception and temporal depth failure

- Heavy rain `80%`, wetness `80%`, night, and reflective road surface.
- Mean speed `39.2 km/h`, P95 `87.0 km/h`.
- Mean danger-target closing speed `12.3 m/s`, roughly twice the other trips.
- Twenty-one false-negative frames, concentrated in frames `317–327`.

T03 needs confidence-aware stereo, object-centric disparity, and short temporal
recovery. A stronger event rule alone cannot recover measurements that become
`inf`.

### T05 — motion/risk interpretation failure

- Brightest and sharpest trip; no rain and almost no lateral acceleration.
- Forty-five false positives, including long runs `490–512` and `542–558`.
- Cars remain visually present while ground truth no longer marks a collision
  threat.

T05 is not an adverse-visibility problem. It needs corrected relative-speed
state, stale-track expiry, ego-motion compensation, and future-path risk.

### T01 — short lateral event and clutter

- Daylight, but approximately `11.88` targets per frame.
- Pedestrian danger occurs in short runs of seven and three frames.
- Mean danger-target lateral offset is `2.59 m`, much larger than other trips.

T01 needs lateral path probability and careful short-event handling.

## Method decision

Do not apply one global weather or confidence threshold.

Use failure-specific branches:

1. Low stereo confidence or adverse imagery: object-centric disparity plus
   temporal/optical fallback.
2. Good imagery but inconsistent closing motion: uncertainty-aware motion
   filter and stale-state reset.
3. Lateral or cut-in targets: predicted corridor-overlap probability.
4. Apply event hysteresis only after the measurement and motion layers.

This comparison supports a method change rather than further global threshold
tuning.
