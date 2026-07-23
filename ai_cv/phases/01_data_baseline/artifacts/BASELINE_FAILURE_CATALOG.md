# Baseline Failure Catalog

This compact catalog is the Stage 1 hand-off set. It contains six deeply
diagnosed cases across four trips. A larger gallery would add examples but would
not change the selected Stage 2 direction.

| ID | Outcome | Trip/frame | GT / prediction | Observed failure | Stage 2 requirement |
|---|---|---|---|---|---|
| F01 | FN | T01 #324 | 1.06 s / `inf` | Pedestrian signal is erased by mixed-ROI median | Separate obstacle support from road/background |
| F02 | FN | T05 #469 | 1.44 s / `inf` | Noisy depth trend estimates a receding target | Robust tracked state and innovation gating |
| F03 | FN | T06 #146 | 0.84 s / `inf` | Motorcycle depth reverses across five frames | Per-obstacle temporal association and uncertainty |
| F04 | FP | T03 #293 | `inf` / 0.17 s | Empty-road disparity jump becomes 34.98 m/s closing speed | Stereo confidence and physical-state consistency |
| F05 | FP | T05 #314 | `inf` / 0.29 s | Mixed pixels create 21.30 m/s false closing speed | Ground removal and collision-corridor filtering |
| F06 | TP | T04 #265 | 1.75 s / 1.71 s | Stable coherent depth trend succeeds | Preserve causal stereo signal when confidence is high |

Exact temporal values and the visual gallery are generated locally by:

```powershell
python ai_cv/phases/01_data_baseline/src/visualize_baseline_failures.py
```

The generated images remain git-ignored because they contain competition data.
