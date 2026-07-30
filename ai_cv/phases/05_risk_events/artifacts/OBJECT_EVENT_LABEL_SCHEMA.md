# Object-event review schema

Use a reviewer-visible CSV with only clip location, object box/track ID, and
empty fields below. Do not include model score, TTC, association score, or the
challenge target.

| Field | Allowed values | Purpose |
|---|---|---|
| `object_id_window` | free short ID | Lets several frames refer to one object. |
| `event_owner` | `yes`, `no`, `uncertain` | Identifies the object plausibly responsible for a safety interaction. |
| `path_relation` | `on_path`, `adjacent`, `crossing`, `diverging`, `uncertain` | Separates path geometry from raw depth. |
| `relative_motion` | `closing`, `steady`, `opening`, `uncertain` | Provides a visual motion check independent of stereo TTC. |
| `cpa_distance_m` | positive number or `unknown` | Validates closest-point-of-approach estimates. |
| `occluded` | `yes`, `no`, `unknown` | Explains unreliable geometry/tracking. |
| `candidate_type` | `road_user`, `static_structure`, `shadow_reflection`, `stereo_artifact`, `mixed_unknown`, `unknown` | Classifies a selected classical candidate, especially when it has no semantic association. |
| `review_confidence` | `high`, `medium`, `low` | Separates useful labels from weak visual evidence. |
| `notes` | free text | Records ambiguity without inventing a class. |

Review a short temporal clip, not one frame. Leave a field `uncertain` or
`unknown` when the scene cannot support a judgement. An uncertain label is
valid evidence; guessing to make a table complete is not.
