# Mini-fold overfit capacity diagnostic

This is an intentional capacity test, not promotion evidence. Frame ID,
timestamp, trip identity, and ground truth were excluded from model inputs.
Blocked validation holds out four contiguous frame blocks.

| Window | Baseline F1 | In-sample F1 | Blocked F1 | Diagnosis |
|---|---:|---:|---:|---|
| T03-Sample `280-360` | 0.432 | 0.906 | 0.627 | promising signal; proceed to causal ablation |
| T05-Sample `430-580` | 0.247 | 1.000 | 0.426 | local signal, poor blocked generalization |

## Interpretation

The current signals can fit the local windows but do not generalize across contiguous held-out blocks. A larger classifier would likely overfit; implement physics-based object depth/motion features before another selection sweep.

A high in-sample score with weak blocked score means the recorded
signals contain local discriminatory information but do not generalize
across the episode. A weak in-sample score means the present features
lack enough information and the measurement method must change.

## Details

### T03-Sample

- Baseline: F1=0.432, P=1.000, R=0.276, TP/FP/FN=8/0/21
- In-sample tree: F1=0.906, P=1.000, R=0.828, TP/FP/FN=24/0/5
- Blocked tree: F1=0.627, P=0.727, R=0.552, TP/FP/FN=16/6/13
- In-sample split features: `candidate_count` (1), `max_closing_speed_mps` (1), `max_confidence` (1), `max_height_norm` (1), `max_width_norm` (1)
- Blocked split features: `candidate_count` (4), `max_height_norm` (4), `max_closing_speed_mps` (1), `min_depth_m` (1), `urgent_center_offset` (1), `urgent_depth_m` (1), `urgent_height_norm` (1)

### T05-Sample

- Baseline: F1=0.247, P=0.192, R=0.345, TP/FP/FN=10/42/19
- In-sample tree: F1=1.000, P=1.000, R=1.000, TP/FP/FN=29/0/0
- Blocked tree: F1=0.426, P=0.556, R=0.345, TP/FP/FN=10/8/19
- In-sample split features: `min_depth_m` (2), `candidate_count` (1), `max_closing_speed_mps` (1), `min_candidate_ttc` (1)
- Blocked split features: `min_depth_m` (6), `max_closing_speed_mps` (3), `min_candidate_ttc` (2), `max_track_hits` (1)

