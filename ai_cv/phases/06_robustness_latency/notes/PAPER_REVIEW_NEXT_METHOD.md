# Paper Review and Diagnosis — Next Method Decision

Date: 2026-07-28  
Scope: decide what to change after the completed Phase 06 deployment gate.

## Evidence from Guardian

The current candidate is deployable on clean input (full six-trip macro
danger-F1 `0.6579`, compute P95 `65.91 ms`) but has three different residual
failure mechanisms:

| Failure | Evidence | Root cause |
|---|---|---|
| T03 rain/night | Lost or unreliable disparity during rapid lead-vehicle braking | Adverse-image stereo measurement and temporal recovery |
| T05 bright rural | Long false alerts for real, non-collision-path vehicles | TTC uses instantaneous corridor/depth motion without future path relevance |
| T01 lateral pedestrian | Brief lateral danger interval in high clutter | Current fixed corridor does not anticipate object/ego path overlap |
| Medium sensor noise | Screening F1 falls `0.1133` from matched clean screen | Stereo correspondence/measurement reliability, not detector class semantics |

This rules out another global confidence/threshold sweep and also rules out
YOLO fine-tuning as the primary solution: the T05 diagnosis confirms that the
detector often identifies a real car correctly while the GT treats it as
non-dangerous.

## Reviewed literature and fit

### 1. Confidence-aware, temporal classical stereo — implement first

Park and Yoon, *Leveraging Stereo Matching With Learning-Based Confidence
Measures* (CVPR 2015), show that match confidence can modulate SGM costs under
unexpected outdoor difficulty. Gehrig et al., *Priors for Stereo Vision under
Adverse Weather Conditions* (ICCVW 2013), use temporal and scene priors to
reduce weather-induced disparity outliers and false positives.

**Guardian fit:** high. This works with the existing SGBM lane, needs no large
new runtime model, and directly addresses measured noise/T03 failure.

**Smallest safe implementation:**

1. Compute an object-level confidence from current left/right disparity
   agreement, valid support fraction, disparity spread, and track-age
   agreement.
2. For low confidence, emit reduced quality; use a short temporal depth prior
   only when the track association and ego-motion-compensated residual agree.
3. Otherwise emit `UNKNOWN`; never reuse stale TTC to manufacture a danger
   alert.
4. Evaluate noise and T03 separately, with leave-one-trip-out selection.

This is a measurement repair, not a new classifier. It should improve T03
recall/noise stability but cannot decide whether a genuine adjacent car is on
a collision path.

### 2. Domain-adaptive learned stereo — research lane, not immediate default

AdaStereo (Song et al., 2021) adapts stereo features at image and cost-volume
levels with no added inference layer. Tonioni et al., *Learning to Adapt for
Stereo* (CVPR 2019), uses unsupervised adaptation and masks unreliable pixels
with learned confidence. RobuSTereo (Wang et al., ICCV 2025) specifically
targets adverse-weather zero-shot stereo through stereo-consistent synthetic
weather and robust features.

**Guardian fit:** plausible for T03/noise only, but unproven for the current
RTX 3060/75-ms budget. It cannot correct T05's collision-path ambiguity.

**Required protocol before attempting it:**

- use external stereo data with adverse conditions (for example DrivingStereo)
  plus unlabeled target pairs for self-supervised adaptation;
- freeze one Guardian trip or an external labeled test set before adaptation;
- compare against the current SGBM path on disparity validity, T03 recall,
  full macro F1, VRAM, and P95;
- accept only if it improves held-out danger F1 without breaking the 75-ms
  deployment limit.

Do not adapt on all six Guardian trips and then report the same trips as
generalization evidence.

### 3. Future corridor-overlap risk — implement second

Neumann and Vedaldi, *Pedestrian and Ego-Vehicle Trajectory Prediction From
Monocular Camera* (CVPR 2021), explicitly separates pedestrian motion from
ego-motion and predicts future positions relative to the ego vehicle. Highway
risk work combines TTC with future trajectory distance rather than relying on
instantaneous TTC alone. Lane-based trajectory proposals (Wang et al., CVPR
2022) are effective but require map/lane inputs unavailable in this project.

**Guardian fit:** high if kept deterministic and geometry-based. It directly
targets T01/T05 rather than image appearance.

**Smallest safe implementation:**

1. Project each confirmed object bottom-centre to the calibrated ground plane.
2. Estimate ego forward path from speed and yaw/lateral motion metadata.
3. Propagate object ground-plane position over a short horizon using a
   confidence-bounded constant-velocity model.
4. Replace the binary fixed corridor test with a probability/bounded score for
   future ego-object corridor overlap.
5. Require both a reliable TTC and sufficient future-overlap score before a
   high-risk alert, while retaining the present conservative fallback for
   uncertain close objects.

This is deliberately not a deep trajectory network: the six trips are too
small to train one without leakage. It can be evaluated in leave-one-trip-out
fashion and gives interpretable T05 false-positive reasons.

## Rejected directions

| Direction | Why not now |
|---|---|
| More YOLO threshold/fine-tuning | T05 objects are often real and correctly classified; semantic fusion already failed full LOTO. |
| Global weather threshold | T02 is very dark but strong, while T05 is bright but weak; illumination is not the causal variable. |
| Large end-to-end dynamic CNN/router | The mini-fold diagnostic shows local fit but weak blocked generalization; it would likely memorize episodes. |
| Heavy occupancy/world model | Needs maps, substantial labeled trajectories, and compute beyond the current deployment scope. |

## Recommended next experiment order

1. **Object-level stereo confidence + temporal prior** — target T03/noise.
2. **Geometry-based future corridor overlap** — target T01/T05.
3. Run the same six-trip LOTO protocol and retain the clean Phase 06 latency
   benchmark as an independent deployment gate.
4. Only if step 1 has a held-out depth/recall benefit, open a separately
   versioned domain-adaptive learned-stereo experiment with external data.

The realistic target for steps 1–2 is a defensible macro F1 around `0.70–0.75`.
Do not represent `>0.80` as achievable without new representative labelled
trajectory/risk data and external validation.

## Primary sources

- Park & Yoon (CVPR 2015), [confidence-modulated stereo matching](https://openaccess.thecvf.com/content_cvpr_2015/html/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.html).
- Gehrig et al. (ICCVW 2013), [stereo priors in adverse weather](https://openaccess.thecvf.com/content_iccv_workshops_2013/W07/papers/Gehrig_Priors_for_Stereo_2013_ICCV_paper.pdf).
- Song et al. (2021), [AdaStereo](https://arxiv.org/abs/2112.04974).
- Tonioni et al. (CVPR 2019), [unsupervised stereo adaptation with confidence](https://openaccess.thecvf.com/content_CVPR_2019/html/Tonioni_Learning_to_Adapt_for_Stereo_CVPR_2019_paper.html).
- Wang et al. (ICCV 2025), [RobuSTereo](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_RobuSTereo_Robust_Zero-Shot_Stereo_Matching_under_Adverse_Weather_ICCV_2025_paper.pdf).
- Neumann & Vedaldi (CVPR 2021), [pedestrian/ego trajectory prediction](https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html).
- Wang et al. (CVPR 2022), [lane-based trajectory prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_LTP_Lane-Based_Trajectory_Prediction_for_Autonomous_Driving_CVPR_2022_paper.html).
