# Phase 05 paper research — fixing the remaining F1 errors

Date: 2026-07-28

## Measured Guardian failure

The current classical guarded pipeline has macro danger-F1 `0.5634`. The
corrected YOLO26 comparator reaches `0.5745` globally but fails the frozen LOTO
promotion gates.

The remaining errors are primarily:

- T03: noisy/merged stereo support destabilizes object depth, closing speed,
  and danger continuity.
- T05: real detected objects or close-range components produce false danger
  because binary corridor membership plus instantaneous TTC does not represent
  their future path probability.

## What relevant papers do

### 1. Object-centric stereo instead of whole-component depth

Stereo R-CNN treats the object region as a coherent unit and refines object
depth through left/right region photometric alignment. IDA-3D likewise makes
depth instance-aware and reweights stereo matching costs around the object.

Guardian implication: do not use every pixel in a merged connected component
equally. Estimate depth from a robust object-centric support region and retain
multiple disparity modes when foreground and background are mixed.

Sources:

- Peiliang Li et al., *Stereo R-CNN Based 3D Object Detection for Autonomous
  Driving*, CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Stereo_R-CNN_Based_3D_Object_Detection_for_Autonomous_CVPR_2019_paper.html
- Wanli Peng et al., *IDA-3D: Instance-Depth-Aware 3D Object Detection From
  Stereo Vision for Autonomous Driving*, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Peng_IDA-3D_Instance-Depth-Aware_3D_Object_Detection_From_Stereo_Vision_for_Autonomous_CVPR_2020_paper.html

### 2. Structured obstacle representation

The Stixel World fuses stereo depth, image evidence, and semantic probabilities
into compact vertical obstacle segments designed for road scenes.

Guardian implication: split a broad component into vertical columns or
disparity-consistent subcomponents before estimating object depth. This is a
better response to merged road/vehicle components than another global
component threshold.

Source:

- Marius Cordts et al., *The Stixel World: A Medium-Level Representation of
  Traffic Scenes*, 2017: https://arxiv.org/abs/1704.00280

### 3. Temporal depth and motion consistency

Joint Spatial-Temporal Optimization combines current object localization with
motion consistency and summarizes historical cues through marginalization.
CODD aligns prior and current estimates with motion and then fuses them to
improve online temporal depth consistency.

Guardian implication: maintain a per-track state for distance and relative
velocity. Weight each new stereo measurement by its confidence instead of
recomputing TTC from an equally trusted frame measurement. Preserve a
high-confidence track through a short low-confidence gap.

Sources:

- Peiliang Li et al., *Joint Spatial-Temporal Optimization for Stereo 3D Object
  Tracking*, CVPR 2020:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Joint_Spatial-Temporal_Optimization_for_Stereo_3D_Object_Tracking_CVPR_2020_paper.html
- Zhaoshuo Li et al., *Temporally Consistent Online Depth Estimation in Dynamic
  Scenes*, WACV 2023:
  https://openaccess.thecvf.com/content/WACV2023/html/Li_Temporally_Consistent_Online_Depth_Estimation_in_Dynamic_Scenes_WACV_2023_paper.html

### 4. Filter state before issuing collision warnings

Nested-Kalman forward-collision work first stabilizes relative distance and
speed, then stabilizes TTC to reduce noisy alerts.

Guardian implication: use a small constant-velocity Kalman or alpha-beta filter
over `[distance, relative_speed]`. Derive measurement noise from disparity
spread, LR residual, valid-pixel support, and association quality. Do not hide
uncertainty by smoothing TTC alone.

Source:

- Qun Lim et al., *Real-Time Forward Collision Warning System Using Nested
  Kalman Filter for Monocular Camera*, IEEE ROBIO 2018:
  https://repository.sutd.edu.sg/esploro/outputs/conferenceProceeding/Real-Time-Forward-Collision-Warning-system-using/9912707409846

### 5. Risk is path probability plus TTC

Lane-based probabilistic collision assessment estimates the probability that a
target occupies each future lane/path using lateral position and velocity,
then combines that with TTC between predicted trajectories.

Guardian implication: replace binary corridor membership at the event layer
with a lightweight corridor-occupancy probability. A real car beside or
leaving the ego path should not remain a high-risk candidate merely because a
large component overlaps the present corridor.

Source:

- Jaehwan Kim and Dongsuk Kum, *Collision Risk Assessment Algorithm via
  Lane-Based Probabilistic Motion Prediction of Surrounding Vehicles*, IEEE
  T-ITS 2018: https://doi.org/10.1109/TITS.2017.2768318

## Recommended Guardian experiment

Implement one interpretable, CPU-light ablation ladder:

1. **Object depth support**
   - Intersect the tracked component with an inner object/track ROI.
   - Cluster valid disparity into at most two modes.
   - Select the temporally consistent foreground mode.
   - Report median, MAD, valid fraction, and LR-consistency fraction.

2. **Uncertainty-aware track filter**
   - State: distance and relative speed.
   - Measurement variance increases with disparity MAD, low support, poor LR
     consistency, and weak association.
   - Predict through at most two missing/low-confidence frames.
   - Return `UNKNOWN`, not `SAFE`, when covariance exceeds a frozen limit.

3. **Probabilistic corridor risk**
   - Estimate current lateral overlap and its recent velocity.
   - Predict overlap probability over a short constant-velocity horizon.
   - Risk evidence requires both closing motion and sufficient future-path
     probability.

4. **Event hysteresis**
   - Enter high risk only after persistent evidence.
   - Exit at a different threshold or after a short clear run.
   - Merge short gaps for the same track.

## Validation

Use six-fold leave-one-trip-out validation. Freeze the search space before the
run and report frame-level and event-level metrics separately.

Hard safeguards:

- frame-level macro danger-F1 must not fall below `0.5634`;
- composite must remain at least `39.71`;
- critical-TTC MAE must not exceed `44.806 s`;
- T03 recall must remain at least `0.276`;
- compute P95 must remain below `75 ms`;
- no trip-specific frame exceptions;
- reset all state between trips.

Primary target for the first ablation: reduce T05 false-positive event duration
without decreasing T03 danger-event recall. A frame-level F1 of `0.60` remains
a research target, not a guaranteed outcome.

## Deferred approaches

- Full Stereo R-CNN, IDA-3D, CODD, or transformer risk prediction: too much
  training data and deployment complexity for the current six-trip dataset and
  6 GB GPU.
- Knowledge distillation: not indicated because the reviewed critical sample
  contains no genuine YOLO miss.
- Further YOLO threshold sweeps: ruled out by the measured oracle ceiling.
