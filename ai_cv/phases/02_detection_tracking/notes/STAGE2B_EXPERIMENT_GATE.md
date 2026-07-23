# Stage 2B Instance-Aware Extractor Gate

This is the next experiment contract, not a decision to adopt deep learning.
Stage 2B must solve the measured Stage 2A identity failure end to end.

## Measured problem

- Track p35 improves danger F1 from `0.220` to `0.402`.
- It improves TP/FN from `80/124` to `135/69`.
- It still scores only `28.7` mean composite and `4.6` worst trip.
- T03 has 139 danger false positives.
- Selected component IDs fragment heavily on T02, T03 and T06.

The next extractor must improve object identity and reduce false threats without
giving back the recovered small-object recall.

## Candidates

1. **Lightweight object detector — primary**
   - Gives instance boxes, class and confidence directly.
   - Reuses stereo depth, causal tracking and collision corridor.
2. **Instance segmentation — optional challenger**
   - Better depth isolation at a higher compute/dependency cost.
3. **Semantic road/free-space segmentation — secondary ablation**
   - Targeted specifically at T03 ghost obstacles from road structure,
     reflections or slight decalibration.
   - The supplied *Stereo Vision for ADAS* chapter reports that semantic road
     evidence nearly eliminates analogous ghost Stixels.
   - It still does not provide persistent object identity by itself.

Exact pretrained models are selected only after checking license, supported
classes, artifact size and CPU/GPU runtime.

## Fixed comparison

Every candidate is compared with:

| Reference | Purpose |
|---|---|
| Stage 1 robust fixed ROI (`32.2`, worst `16.9`) | score and robustness |
| Stage 2A track p35 (`F1 0.402`, TP 135, FN 69) | object-threat recall |
| Official baseline (`19.7`, F1 `0.220`) | organizer reproducibility |

The geometry, TTC evaluator and trip split stay fixed during the extractor
comparison.

## Falsification order

1. Six known failure frames: boundary and depth-isolation visual check.
2. All labeled danger frames plus matched empty-road negatives.
3. A 72-frame stratified sample for latency and coverage.
4. Sub-pixel/range and small rectification-offset sensitivity checks.
5. Full 3,600-frame official TTC evaluation.

A candidate that fails an earlier gate is not tuned on the full set.

## Promotion criteria

A candidate is promoted only when all are true:

- danger F1 is at least the Stage 2A track-p35 value;
- TP is not lower by more than 5 and FN is not higher by more than 5;
- mean composite exceeds `32.2`;
- worst-trip composite exceeds `16.9`;
- T03 danger FP decreases materially;
- selected-track ID switching decreases on T02/T03/T06;
- confidence is range-aware and calibration degradation is detectable;
- latency distribution, model license and degraded behavior are reported;
- inference is causal and does not use evaluation labels.

The score thresholds are deliberately tied to existing evidence. If no
candidate passes, Stage 1 robust ROI remains the score reference and Stage 2A
geometry remains an auxiliary threat proposal path.

## Stop rules

- Do not grid-search morphology after instance boundaries are available.
- Do not compare detector mAP alone; the decision metric is end-to-end TTC.
- Do not promote semantic segmentation solely because it is visually cleaner.
- Do not optimize deployment latency before a candidate passes the 72-frame
  correctness gate.
- Do not download or commit pretrained weights into this repository.
