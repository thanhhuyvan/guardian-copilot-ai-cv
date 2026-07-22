# Data Usage and Processing Profiles

## `causal_online`

- A frame prediction may use only video, telemetry and state available at or before the current timestamp.
- Future video frames and bidirectional smoothing are forbidden.
- The complete `events_log` schedule must not be loaded as a future prior. An event record may be consumed only after its timestamp has occurred and only if the runtime would genuinely expose it.
- This is the default profile for official TTC comparison and any real-time claim.

## `offline_post_trip`

- Full-trip context and explicitly named offline smoothing are allowed.
- Outputs must be labelled `offline_post_trip` in `run_manifest.v1`.
- Results must not be presented as real-time or directly compared with causal latency claims without clear separation.

## Depth keyframes

- Depth keyframes exist in both practice and scored data and are described by the local starter documentation as available input.
- Every run must declare one policy: none, validation-only, calibration, direct inference or interpolation.
- Before final submission, direct/interpolated use should be reconfirmed against organizer rules; validation-only experiments are always reported separately from inference features.

## Redacted and future information

- Ground-truth TTC, redacted 3D locations and future event knowledge are prohibited in causal predictions.
- `target_id` and coarse `target_class` may be used when present, but usage must be declared in `input_features`.

