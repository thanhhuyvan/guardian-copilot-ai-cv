# YOLO26 implementation log

Plan: `docs/YOLO26_SEMANTIC_FUSION_PLAN.md`

Update this file after each work session. Keep entries short. Never paste
passwords, tokens, private URLs, model files, or full raw logs.

## Current state

- Status: CLOSED — research completed; YOLO26 candidate rejected, not promotable
- Branch: `research/phase-04b-yolo26-fusion`
- Phase 04B base: `19b6796` (docs: plan YOLO26 semantic fusion)
- Last completed step: T03 critical labeling, association correction, fixed
  27-policy rerun, final reject decision, and full Phase 02 regression suite
- Final evidence:
    1. All 23 critical T03 rows reviewed: 16 association failures, 7 stereo
       noise, 0 genuine detector misses, 0 unsure.
    2. Symmetric containment association correction raised global-best F1 from
       0.5634 to 0.5745 with no critical-TTC MAE regression.
    3. Promotion still fails: 5/6 LOTO folds feasible, partial LOTO F1 0.5286,
       T03 recall 0.241, T05 FP 45, oracle ceiling 0.5745 < 0.60.
    4. Per the frozen stop rule, no more broad tuning, fine-tuning, TensorRT,
       INT8, or official latency benchmarking is justified for this candidate.
    5. ONNX raw parity remains narrowly outside two gates but is
       fusion-equivalent; it is moot for a rejected candidate.
    6. Ultralytics licensing remains documented but is not a product blocker
       because the YOLO dependency is not promoted.
- Repository rule: commit Phase 04B source, tests, reports, and small
  reproducibility artifacts; keep `.venv_yolo26/`, models, and derived
  overlays gitignored.

## Latest metrics

```text
Branch:              research/phase-04b-yolo26-fusion
HEAD:                19b6796

Physical guard baseline (no semantics, confirmed correct):
  LOTO macro F1:     0.5634
  LOTO composite:    39.71
  Critical TTC MAE:  44.806 s
  T03 MAE:           62.231 s
  T05 FP:            45

Semantic fusion after association correction:
  Global best:       F1=0.5745, composite=40.234, MAE=44.806 s
  LOTO:              5/6 folds feasible; partial macro F1=0.5286
  LOTO composite:    37.950
  LOTO MAE:          46.953 s
  LOTO T03 recall:   0.241
  T05 FP:            45
  Oracle ceiling:    F1=0.5745 (still below 0.60)

ONNX parity (GPU rerun, both backends confirmed on CUDA):
  PyTorch device:    cuda:0
  ONNX providers:    [CUDAExecutionProvider, CPUExecutionProvider]
  Matched pairs:     87 / 72 frames
  Class agreement:   96.55%  [GATE FAIL — target >=99%]
  Median IoU:        0.9968  [PASS — target >=0.98]
  Mean conf diff:    0.0205  [GATE FAIL — target <=0.02]
  Conf diff dist:    p50=0.0072 p90=0.0383 p95=0.1156 p99=0.2512 max=0.279
                     17/87 pairs >0.02, 5/87 >0.10
  Class swaps:       3 total — ALL same-box competing-class (0 distinct-object)
                     000150 car@0.446 <-> truck@0.489 IoU=0.998
                     000350 truck@0.251 <-> car@0.498 IoU=0.997
                     000250 truck@0.299 <-> car@0.455 IoU=0.999
  Detection-set diff: 2 torch-only + 2 onnx-only unmatched, all conf 0.259-0.281 (boundary)
  Fusion-equivalent:  VERIFIED — car/truck swap produces identical soft-guard behavior
                      (only retained-class membership + confidence matter; both stay
                       above 0.25 support threshold, neither fires suppression)
  Root cause:         end-to-end ONNX export retains competing class hypotheses per box
                      and tiebreaks by insertion order; Ultralytics native postprocess
                      (PyTorch) tiebreaks differently. NOT double-NMS, NOT a parser bug.
  parity_valid:      false (raw-tensor gates), true at fusion level (per plan: "Compare
                     danger output, not only raw detector tensors")

Unit tests:          31 collected, 31 passing (test_yolo26_fusion.py)
Full Phase 02 suite:  138 passed + 8 subtests
  New: test_ensure_cuda_dlls_on_path_is_idempotent_and_safe
       test_car_truck_same_box_are_fusion_equivalent
       test_same_box_competing_class_swap_is_not_a_localization_error

GPU env (RESOLVED):
  RTX 3060 6GB, driver 572.16, CUDA 12.8
  venv_yolo26 torch: 2.11.0+cu128  [cuda available: True]
  venv_yolo26 ORT:   1.20.1 onnxruntime-gpu, providers: [CUDA, CPU, TensorRT]
  FIX: _ensure_cuda_dlls_on_path() prepends torch/lib to PATH before ORT session
       creation — ORT's CUDA EP needs cuDNN 9/CUDA 12 DLLs that torch bundles but
       does not put on PATH; without this ORT silently fell back to CPU.
```



## Session entries

### 2026-07-28 (cont. 4) — critical labeling, association repair, final closure

```text
STATUS: CLOSED — candidate rejected at accuracy gate
BRANCH: research/phase-04b-yolo26-fusion
HEAD:   19b6796 (working tree not committed)

LABEL REVIEW:
  23/23 critical T03 rows labeled
  association_failure=16, stereo_noise=7, genuine_miss=0, unsure=0
  Validator recommendation: fix component/detection association

ROOT CAUSE:
  Large merged stereo components fully contained correct YOLO car boxes, but
  failed the frozen one-way component-center rule.

FIX:
  Add symmetric containment match: detection center inside component,
  intersection/detection area >=0.50, vertical overlap >=0.50.

RESULT:
  baseline:     F1=0.5634, composite=39.712, MAE=44.806s, FP=89
  global best:  F1=0.5745, composite=40.234, MAE=44.806s, FP=81
  LOTO:         5/6 feasible, partial F1=0.5286, T03 recall=0.241, T05 FP=45
  oracle:       F1=0.5745
  target:       F1>=0.60 — FAIL

TESTS:
  test_yolo26_fusion.py: 31/31 passed
  full Phase 02 suite: 138 passed + 8 subtests

DECISION:
  Reject YOLO26 promotion. Retain classical guarded pipeline and advance to
  Phase 05 confidence/risk/events. Accuracy failure stops TensorRT/INT8 and
  official latency work for this candidate.

FILES:
  ai_cv/phases/02_detection_tracking/src/semantic_fusion.py
  ai_cv/phases/02_detection_tracking/tests/test_yolo26_fusion.py
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_annotation/t03_annotation_labeled.csv
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_annotation/t03_annotation_labeled.xlsx
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto_association_v2/
```

### 2026-07-28 (cont. 3) — true oracle correction + full regression suite

```text
STATUS: BLOCKED on human T03 labeling; reproducibility defect fixed
BRANCH: research/phase-04b-yolo26-fusion
HEAD:   19b6796 (no new commits)

FIX:
  sweep_yolo26_fusion.py previously called the best semantic-only per-trip
  selection (0.5577) an "oracle upper bound" even though it excluded the
  observed no-semantic baseline (0.5634). A bound below an available result is
  invalid. The oracle now selects from baseline/off + all 27 semantic configs
  per trip, without per-trip gates; macro gates are reported on the aggregate.

RESULT:
  baseline:        F1=0.5634, composite=39.71, MAE=44.81s
  global semantic: F1=0.5577, composite=39.77, MAE=47.69s, INFEASIBLE
  LOTO:            0/6 feasible folds
  true oracle:     F1=0.5686, composite=39.89, MAE=44.81s
  target:          F1>=0.60 — FAIL even under test-label-cherrypicked oracle

TESTS:
  test_yolo26_fusion.py: 28/28 passed
  full Phase 02 suite: 135 passed + 8 subtests
  annotation template validator: valid, 0/377 labels filled

FILES:
  src/sweep_yolo26_fusion.py — corrected oracle definition and aggregate gates
  tests/test_yolo26_fusion.py — regression test requiring baseline in oracle
  artifacts/yolo26_loto/sweep_selection_modes.json — regenerated
  artifacts/yolo26_loto/sweep_per_config_per_trip.csv — regenerated

NEXT:
  Human labels 23 S1 danger rows first, then remaining 354 rows. Run
  validate_t03_annotation.py on t03_annotation_labeled.csv. Do not choose
  fine-tuning/stereo/association work before the S1 distribution is known.
```

### 2026-07-28 (cont. 2) — annotation row<->box matcher fix + validator script

```text
STATUS: scaffold ready for human labeling (overlays + CSV + validator)
BRANCH: research/phase-04b-yolo26-fusion
HEAD:   19b6796 (no new commits)
STEP:   Fix the overlay-box <-> CSV-row mismatch the user flagged. The old
        overlay drew red boxes labelled "#1, #2, ..." but the CSV had no column
        matching that index, so a labeler could not reliably tie a row to its
        box. Add the matcher column, print the matching key on the box, add a
        validation/aggregate script, regenerate.

FIX (build_t03_annotation_scaffold.py):
  - Merged overlay-render and CSV-write into a single per-frame pass. The SAME
    enumerate(sup_by_frame[fid], start=1) now drives BOTH the on-image "#k"
    box label and the CSV row's `overlay_box_index`, so they match by
    construction, not by convention.
  - The red box label now prints BOTH keys: "#<k> tid=<track_id> d=..m
    TTC=..s GT=.." so a labeler can match a CSV row by either
    overlay_box_index or track_id.
  - Replaced the old `labeler_seq` column (a frame sequence number) with
    `overlay_box_index` (the actual on-image matcher). frame_id ordering is
    preserved in CSV write order so the file is still browsable frame-by-frame.

NEW SCRIPT (src/validate_t03_annotation.py):
  - Validates the labeled CSV:
      header unchanged (exact field order vs scaffold writer)
      overlay_box_index is a positive int, contiguous 1..N per frame, no dups
      label values exactly one of
        genuine_miss | stereo_noise | association_failure | unsure
      geometric/stereo scalar columns finite; TTC columns allow 'inf' (legit
        danger NULL per R-TTC-02/R-SUB-04 plan rules -- this was an off-by-one
        validator bug I caught while smoke-testing; fixed before any labeling).
  - Aggregates the label distribution split by S1 (GT-danger, 19 frames -> 23
    candidate rows = the recall loss) and S2 (221 frames -> 354 rows).
  - Decision rule: a winner requires share >= 40% BOTH overall AND in S1 with
    a consistent label -> DECISIVE; otherwise MIXED and S1 is declared the
    action trigger (because F1 loss is concentrated in S1).
  - Writes <csv>.report.json and prints the next-phase recommendation.

REGENERATION RESULT:
  Deleted the old scaffold dir and re-ran build_t03_annotation_scaffold.py:
    Sampled frames:    240  (S1 danger=19, S2 strided=221)
    Suppressed cands:  377/798 on sampled frames
    Overlays rendered: 240  (missing source imgs: 0)
  Verification of the new CSV:
    header HAS overlay_box_index, removed labeler_seq -> 21 cols.
    per-frame overlay_box_index contiguity: 0/240 non-contiguous, 0 dups.
    first row (frame 309, track 138, idx 1): depth 8.678m, cand TTC 0.325,
                                              GT TTC 1.924, is_gt_danger_frame=1.
  Validator on the unlabeled CSV: PASSES, exit 0 ("valid but no rows labeled
    yet"). Smoketest with fake labels confirmed both the DECISIVE threshold and
    the MIXED S1-priority branch behave correctly. Test artifacts cleaned.

LABELING PROCEDURE (what the human does next):
  1. Open t03_annotation_template.csv.
  2. Open the matching overlay by frame_id -> overlays/<frame_id>.jpg.
  3. Match the row by overlay_box_index (the '#k' on the red box) AND track_id
     (the 'tid=...' also printed on the box).
  4. Fill the `label` column with exactly one of
       genuine_miss | stereo_noise | association_failure | unsure
  5. Add a short `annotator_notes` when uncertain.
  6. Label the is_gt_danger_frame = 1 rows FIRST (these are the recall loss).
  7. Save as CSV WITHOUT changing headers (the validator rejects header drift).

FILES CHANGED:
  src/build_t03_annotation_scaffold.py   MODIFIED (single-pass render+emit,
                                          overlay_box_index col, '#k tid=...',
                                          updated print summary + report)
  src/validate_t03_annotation.py        NEW (validate + S1/S2 aggregate +
                                          recommendation; TTC-cols tolerate inf)
  artifacts/yolo26_annotation/*         REGENERATED (CSV 377 rows w/
                                          overlay_box_index, 240 overlays w/
                                          '#k tid=...' labels, report JSON)
  docs/YOLO26_IMPLEMENTATION_LOG.md     this session entry.

TESTS: 27/27 pass (test_yolo26_fusion.py) -- unaffected. The scaffold/validator
       scripts are reproducibility contracts (not unit-tested yet); running
       validate_t03_annotation.py on the labeled CSV is the acceptance test.

NEXT (waiting on human):
  - Human labels the 377 rows following the procedure above.
  - Human runs: python src/validate_t03_annotation.py --csv <labeled CSV>
  - Validator report recommends the next phase:
      STRONG genuine_miss      -> fine-tune YOLO26 (separate phase)
      STRONG stereo_noise       -> improve stereo confidence/fusion
      STRONG association_failure -> improve bbox/track matching
  - Only after the human confirms which bucket dominates do we plan an
    accuracy fix. Do NOT fine-tune beforehand.

ERROR: none. Validator ran cleanly on unlabeled CSV; smoketest passed; test
       artifacts cleaned; tests still 27/27.
```

### 2026-07-28 (cont.) — T03 annotation scaffold built (step 5-6)

```text
STATUS: BLOCKED on human labeling (scaffold ready)
BRANCH: research/phase-04b-yolo26-fusion
HEAD:   19b6796 (no new commits)
STEP:   Build the stratified T03 annotation scaffold so the accuracy ceiling
        failure can be CLASSIFIED (genuine_miss | stereo_noise |
        association_failure) before any fine-tuning decision. Per user plan
        step 5-6.

NEW SCRIPT: src/build_t03_annotation_scaffold.py
  - Runs soft-guard with the reproducible-sweep winner config
    SemanticConfig(0.20, 2, 5.0) [matches sweep_yolo26_fusion.py]
  - Stratified frame sampling (sampled FRAMES, not candidates, so a human
    labels in one pass):
      S1 = ALL GT-danger frames (< danger_ttc_s, default 2.0s) that carry
           >=1 suppressed candidate (the recall loss).  mined 19 frames in T03.
      S2 = remaining suppressed-only frames, uniform-strided to --max-frames
           (default 240).  strided to 221 frames.
      total = 240 sampled frames.
  - For each sampled frame: renders an overlay into overlays/<frame>.jpg
    showing the RED stereo-component box of every suppressed candidate
    (labelled with its index, depth, candidate TTC, GT TTC) plus every
    CYAN YOLO detection on that frame (class + confidence). Source image:
    Practice_Dataset/T03-Sample/kitti/image_2/*.jpg (600 frames available).
  - Writes t03_annotation_template.csv: one row per suppressed candidate on
    the sampled frames, pre-populated with frame_id, track_id, labeler_seq,
    depth_m, candidate_ttc, ground_truth_ttc, ground_confidence,
    closing_speed_mps, motion_residual_m, yolo_matched_class, semantic_score,
    misses, selected_* geometry, is_gt_danger_frame. The empty `label` and
    `annotator_notes` columns are what the human fills.

ARTIFACTS (artifacts/yolo26_annotation/):
  overlays/                          240 jpg  (rendered, missing 0)
  t03_annotation_template.csv        377 rows, 21 cols
  t03_annotation_scaffold_report.json   reproducibility record

RESULT (scaffold run output):
  Stratified sampling for T03-Sample:
    S1 (GT-danger + suppressed): 19 frames (all)
    S2 (suppressed-only, strided): 221 frames
    total sampled frames:           240
    total suppressed candidates on sampled frames: 377 / 798 total
  Coverage: the 240 sampled frames carry 377 of the 798 suppressed candidates
  this config produces on T03 (47%); S1 covers ALL 19 GT-danger+suppressed
  frames, which is where the recall loss concentrates (FN 21->29).
  Label template rows by stratum: S1 = 23 rows, S2 = 354 rows.

LABEL CODES (see JSON report 'label_codes'):
  genuine_miss      suppressed candidate IS a real road user YOLO failed to
                   detect -> fine-tune YOLO26 (separate phase, fold-specific
                   models, no leakage).
  stereo_noise      suppressed candidate is NOT a real object, just stereo/
                   depth noise -> improve stereo confidence / fusion.
  association_failure YOLO did detect the object but the component-detection
                   match failed (IoU/center) -> improve bbox/track matching.
  unsure            ambiguous; needs more context.

DECISION RULE (per user step 6):
  After labeling, the aggregate label distribution chooses the next phase:
    dominant genuine_miss      -> fine-tune YOLO26
    dominant stereo_noise      -> improve stereo confidence/fusion
    dominant association_failure -> improve bbox/track matching
  Do NOT fine-tune until the labeling confirms which bucket dominates.

FILES CHANGED THIS SESSION:
  src/build_t03_annotation_scaffold.py                         NEW
  artifacts/yolo26_annotation/{overlays,t03_annotation_template.csv,
                                t03_annotation_scaffold_report.json}  NEW
  docs/YOLO26_IMPLEMENTATION_LOG.md                            MODIFIED:
    - updated current-state blocker #1 and #3
    - this new session entry

TESTS: 27/27 pass (test_yolo26_fusion.py) — unchanged.
       The scaffold scripts are reproducibility contracts, not unit-tested
       yet; re-running build_t03_annotation_scaffold.py yields byte-identical
       overlays + template given the frozen source and config.

COMMANDS RUN:
  .venv\Scripts\python.exe ai_cv\phases\02_detection_tracking\src\build_t03_annotation_scaffold.py
  .venv\Scripts\python.exe -m pytest ...test_yolo26_fusion.py -q  # 27/27

NEXT:
  1. Human labels the 377 rows in t03_annotation_template.csv using the 240
     overlays as ground truth.
  2. Aggregate the label distribution; choose next phase per the decision rule.
  3. Only after labels are in: decide whether to fine-tune YOLO26 (genuine-miss
     dominant) vs fix stereo/association. Keep the frozen contract untouched.

ERROR: none.
```

### 2026-07-28 — Reproducible per-config-per-trip sweep + corrected F1 ceiling

```text
STATUS: BLOCKED (accuracy ceiling proven reproducibly)
BRANCH: research/phase-04b-yolo26-fusion
HEAD:   19b6796 (no new commits)
STEP:   Make the accuracy conclusion reproducible before any fine-tuning, per
        user instruction. Correct the wrong "0.43" macro figure from the
        22:00 entry.

DELIVERABLES:
  New script: src/sweep_yolo26_fusion.py
    - Runs baseline + 27 semantic configs x 6 trips = 168 evaluations
    - Writes sweep_per_config_per_trip.csv with columns:
        configuration, trip, tp, fp, fn, precision, recall, f1,
        composite, mae_critical, suppressed_candidates
    - Computes 4 selection modes from the CSV:
        baseline       physical guard only (regression reference)
        global_best    one config for all trips, picked by plan's selection rule
        loto           proper 6-fold leave-one-trip-out (train 5, eval 1)
        oracle         per-trip best choice among baseline/off + 27 semantic configs
                       (UPPER BOUND, not selectable)
  New artifact: artifacts/yolo26_loto/sweep_per_config_per_trip.csv (168 rows)
  New artifact: artifacts/yolo26_loto/sweep_selection_modes.json (full report)

RESULTS (all numbers reproducible by running sweep_yolo26_fusion.py):
  Sweep verified: 168 rows = 28 configs x 6 trips, counts match.
  Baseline matches frozen reference exactly on every trip:
    T01 F1=0.4516 FP=14, T02 0.7647 FP=3, T03 0.3333 FP=11,
    T04 0.7629 FP=8,  T05 0.2609 FP=45, T06 0.8070 FP=8
    baseline macro F1=0.5634  composite=39.71  MAE=44.81s  total FP=89

  Four selection modes (target: macro F1 >= 0.60 without MAE regression):
    baseline       macro F1 = 0.5634   (composite 39.71, MAE 44.81s, FP 89)
    global_best    macro F1 = 0.5577   (cfg s0.2_m2_d5.0; composite 39.77 passes;
                                        MAE 47.69s FAILS gate; INFEASIBLE)
    loto           6/6 folds INFEASIBLE  (no config met both gates on any fold)
    oracle         macro F1 = 0.5686   (UPPER BOUND; baseline/off included;
                                        composite 39.89, MAE 44.81s)

  CORRECTED F1 ceiling: true oracle upper bound = 0.5686.
  The earlier 0.5577 value excluded the no-semantic baseline and therefore could
  not be an upper bound because it was below an observed baseline of 0.5634.
  The corrected oracle chooses baseline/off or one of 27 semantic configs per
  trip using test labels. It is not selectable, but it validly bounds this search
  space and remains below the 0.60 target.

  Oracle per-trip detail (which choice wins each trip if we could cheat):
    T01 s0.2_m2_d4.0 F1=0.4828  (semantic improves)
    T02 baseline       F1=0.7647  (semantic ties; baseline wins stable order)
    T03 baseline       F1=0.3333  (all semantic configs regress)
    T04 baseline       F1=0.7629  (all semantic configs regress)
    T05 baseline       F1=0.2609  (all semantic configs are flat)
    T06 baseline       F1=0.8070  (semantic ties; baseline wins stable order)
  Only T01 improves under the oracle. The small macro gain to 0.5686 still does
  not reach 0.60 and cannot be selected without test-trip leakage.

KEY REPRODUCIBLE FACTS:
  - T05 is FLAT across all 27 configs: F1=0.2609, FP=45, identical. Nothing in
    the frozen contract can move T05. Mechanism (confirmed 22:00): close-range
    fallback bypasses suppression by design; GT labels the close-range objects
    safe. This is a corridor/GT issue, not a detector issue.
  - T03 can only degrade under semantics: every config suppresses too many
    candidates (loses 8 TP), F1 0.3333 -> 0.2941, MAE 62 -> 77s. This is a
    YOLO coverage problem on dark T03 scenes.
  - The corrected oracle ceiling (0.5686) is slightly ABOVE baseline (0.5634)
    because T01 can improve if selected with test-trip labels. No single global
    semantic config or LOTO fold is feasible, and 0.60 remains unreachable
    without either (a)
    better detector coverage on T03 (fine-tune) or (b) corridor geometry change
    for T05 (plan amendment required).

DECISION (per user's step 5-6 instruction):
  No global/LOTO config reaches F1 0.60 without MAE regression. Proceed to
  stratified T03 annotation (step 5-6) to classify whether the 750 suppressed
  T03 candidates are:
    - genuine YOLO misses    -> fine-tune YOLO26 (separate phase)
    - stereo noise           -> improve stereo confidence/fusion
    - association failures   -> improve bbox/track matching
  Do NOT fine-tune until the annotation confirms which class of failure this is.

FILES CHANGED THIS SESSION:
  src/sweep_yolo26_fusion.py                              NEW (240 lines)
  artifacts/yolo26_loto/sweep_per_config_per_trip.csv     NEW (168 rows)
  artifacts/yolo26_loto/sweep_selection_modes.json        NEW
  docs/YOLO26_IMPLEMENTATION_LOG.md                       MODIFIED:
    - corrected the wrong "0.43" figure (in-place, marked CORRECTION)
    - this new session entry

TESTS: 27/27 pass (test_yolo26_fusion.py) — unchanged.
       sweep_yolo26_fusion.py has no unit tests yet; reproducibility is the
       contract (re-running yields byte-identical CSV given the frozen source).

COMMANDS RUN:
  .venv\Scripts\python.exe ai_cv\phases\02_detection_tracking\src\sweep_yolo26_fusion.py
  .venv\Scripts\python.exe -m pytest ...test_yolo26_fusion.py -q  # 27/27

NEXT:
  Build stratified T03 annotation scaffold (step 5-6): sample 200-300 T03
  frames with the suppressed candidates overlaid, plus a per-candidate label
  template (genuine-miss / stereo-noise / association-failure). A human labels
  in one pass; the fine-tune-vs-stereo-fix-vs-association-fix decision becomes
  data-driven. Do not start fine-tuning before this is done.

ERROR: none. All gates failures are measurement results, reproducible, and
       correctly classified.
```

### 2026-07-27 22:00 — F1 lever investigation (per plan stop rule)

```text
STATUS: BLOCKED (confirmed — no F1 lever inside frozen contract)
BRANCH: research/phase-04b-yolo26-fusion
HEAD:   19b6796 (no new commits)
STEP:   Investigate whether macro F1 (0.5634) can reach 0.60 gate without
        violating the plan's stop rule (no threshold sweep / corridor change)

METHOD:
  Ran the FULL 27-config plan grid per-trip (not just per-fold) and recorded
  per-trip metrics, to find if ANY plan-compliant config moves the macro F1.

RESULT (per-trip, full 27-config swept):
  T01: baseline F1=0.4516/FP=14  -> best F1=0.4828/FP=12 (cfg s=0.2,m=2,d=4.0)
  T03: baseline F1=0.3333/FP=11  -> ALL 27 drive FP=0, F1=0.2941, LOSE 8 TP
                                   (this is the infeasibility mechanism)
  T05: baseline F1=0.2609/FP=45  -> ALL 27 give FP=45, F1=0.2609 EXACTLY
  T06: baseline F1=0.8070/FP=8   -> ALL 27 flat at F1=0.8070/FP=8

  CORRECTION (2026-07-28): the "0.43" line above was wrong — the arithmetic
  itself was off and the methodology was cherry-picking one fixed config per
  trip. The correct ceiling is the ORACLE upper bound (best feasible config
  PER trip, ignoring train/test selection). See the 2026-07-28 reproducible
  sweep entry below:
    oracle macro F1 = 0.5686  (corrected again: baseline/off is selectable)
    baseline    macro F1 = 0.5634
  => the true oracle is slightly above baseline but remains below 0.60. The
     earlier 0.5577 excluded baseline/off and was a semantic-only result, not a
     mathematical upper bound. Do not quote 0.43 or 0.5577 as the oracle.

ROOT CAUSE — T05 is provably unsolvable by soft-guard:
  T05 FP=45 across ALL 27 configs with identical F1. Mechanism confirmed by
  per-track instrumentation of cfg(s=0.25,m=3,d=5.0):
  - 34 total tracks, 7 get suppressed (but none of the 45 FPs)
  - 6 far(>5m)+unsuppressed tracks, max-misses dist {1:2, 2:3, 4:1}
  - Traced track_id=18: misses DID reach 3-4 (frame 211-212), but suppression
    blocked because latest_depth=4.0m <= 5.0m close-range fallback.
  - This IS the plan spec ("At depth <=5m, preserve current guarded TTC
    behavior"). NOT a soft-guard bug.
  - T05 track fragmentation: 65% of unique tracks <=5 frames, 26% single-frame.
    Even so, suppression correctly accumulates across candidates; the close-range
    fallback is what blocks it, by design.

ROOT CAUSE — T03 can only be made WORSE:
  Every one of the 27 configs suppresses 40+ T03 tracks (750/1355 candidates),
  driving FP 11->0 but LOSING 8 TP (FN 21->29). F1 0.3333->0.2941. MAE 62->77.
  This is a YOLO coverage problem on dark T03 scenes, not a soft-guard problem.

CONCLUSION — plan stop rule applied:
  The plan says: "Reject YOLO fusion if it fails F1 or T03 recall after the
  fixed 27-policy semantic search. Do not start another broad threshold sweep."
  Empirically confirmed: macro F1 cannot reach 0.60 via the 27 policies.
  T05 FPs are GT/corridor (not YOLO); T03 TP loss is YOLO coverage (not soft-guard).
  No legitimate F1 lever remains inside the frozen contract.

ACTION (correct, not a workaround):
  Do NOT tune further. The two paths the plan prescribes for THIS exact case:
    (A) "Detector misses true road users: collect box annotations." -> T03
        stratified annotation to confirm whether the 750 suppressed candidates
        are genuine road users (=> fine-tune, separate phase) or stereo noise
        (=> accept pretrained ceiling, document).
    (B) "Corridor geometry tightening experiment on T05." This is EXPLICITLY
        out of frozen-contract scope (plan: "Keep current physical guard"),
        so it requires a written plan amendment + your approval before any
        corridor parameter is touched.

  Recommend proceeding with (A): build the T03 stratified annotation scaffold
        (sample 12 frames/trip + suppressed-candidate overlay + per-object
        label template undecided real/nose/noise) so a human can label in
        one pass and the fine-tune decision becomes data-driven.

FILES CHANGED THIS SESSION: none (investigation only; all code from 21:05 entry
  remains untracked in the working tree).

TESTS: 27/27 pass (test_yolo26_fusion.py) — unchanged.
COMMANDS RUN:
  .venv\Scripts\python.exe -m pytest ...test_yolo26_fusion.py -q   # 27/27
  per-trip full-grid sweep + per-track suppression instrumentation
  (inline python -c diagnostics — not committed)

ERROR: none.
```

### 2026-07-27 21:05 — ONNX parity GPU rerun + root-cause diagnosis + CUDA DLL bootstrap fix

```text
STATUS: PARTIALLY UNBLOCKED
BRANCH: research/phase-04b-yolo26-fusion
HEAD:   19b6796 (no new commits — all changes untracked)
STEP:   ONNX parity rerun on GPU + diagnosis (per plan step 5) + backend fix + 3 tests

ENV RESOLVED:
  Earlier log recorded venv_yolo26 as CPU-only torch + ORT without CUDA.
  Current state:
    torch 2.11.0+cu128, cuda.is_available()=True, RTX 3060 detected
    onnxruntime-gpu 1.20.1, providers include CUDA + TensorRT
  BUT ORT CUDA EP failed to load: LoadLibrary error 126 — PyTorch bundles
  cuDNN 9/CUDA 12 DLLs in torch/lib but does NOT put that dir on PATH, so
  ORT (which loadlibrary's them) could not find them and silently fell back
  to CPUExecutionProvider.

FIX (yolo26_backends.py):
  Added _ensure_cuda_dlls_on_path() — prepends torch/lib to PATH before
  creating the InferenceSession when CUDA is requested. ORT CUDA EP now
  loads cleanly. Verified: session.get_providers() returns
  ['CUDAExecutionProvider', 'CPUExecutionProvider'].

ONNX PARITY RERUN (both backends on CUDA, 72 raw source frames):
  Matched pairs:        87
  Class agreement:      96.55%  [GATE FAIL — 3 car/truck swaps]
  Median IoU:           0.9968  [PASS]
  Mean conf diff:       0.0205  [GATE FAIL — over by 0.0005]
  Conf diff dist:       median 0.0072 (excellent), long tail
                         p95=0.1156, max=0.279
                         17/87 pairs >0.02, 5/87 >0.10

ROOT CAUSE (NOT a bug — do not patch away):
  All 3 class swaps are car<->truck on the SAME physical object (IoU 0.997-0.999)
  at low confidence near the 0.25 threshold:
    000150.jpg: PT car@0.446   <-> ONNX truck@0.489
    000350.jpg: PT truck@0.251 <-> ONNX car@0.498
    000250.jpg: PT truck@0.299 <-> ONNX car@0.455
  The end-to-end ONNX export (output shape [1,300,6]) retains COMPETING CLASS
  HYPOTHESES per box (e.g., one frame emits both car@0.4976 and truck@0.2463
  for the identical box) and tiebreaks by insertion order. Ultralytics'
  native postprocess (used by model.predict) tiebreaks differently.
  Debugging history: initial double-NMS hypothesis was DISPROVED — the
  export's internal NMS is class-aware on the top-300; the mismatch is
  upstream of any postprocess, at the export head itself.

FUSION EQUIVALENCE VERIFIED:
  Per plan: "Compare danger output, not only raw detector tensors."
  semantic_fusion.soft-guard uses only:
    (a) retained-class membership — car AND truck are both RETAINED_CLASSES
    (b) confidence feeding the EMA score
  No branching on class_id. Confirmed via direct simulation:
    same-box car@0.446 vs truck@0.489, 5 matched frames at depth 10m:
      car  score=0.4117, suppressed=[F,F,F,F,F]
      truck score=0.4512, suppressed=[F,F,F,F,F]
    Identical suppression behavior. EMA scores differ (0.41 vs 0.45) but
    both stay above the 0.25 support threshold — neither candidate is
    suppressed differently.
  Detection-set differences: 2 torch-only + 2 onnx-only unmatched, all at
  conf 0.259-0.281 (purely boundary cases near the 0.25 cutoff).

INTERPRETATION:
  Raw-tensor parity gates fail narrowly (96.55% vs 99%, 0.0205 vs 0.02) due
  to 3 boundary class-tiebreaks that are fusion-equivalent. The failures
  are cosmetic at the raw-tensor level and are an inherent property of the
  end-to-end ONNX export, not a fixable parser bug. Per the plan's own rule
  ("compare danger output, not only raw detector tensors"), the honest path
  is to document this, not to hack the postprocess to inflate agreement.

  The mean-conf-diff gate (0.0205 vs 0.02) is driven by the same 17 boundary
  pairs (median 0.0072 is well within gate). It reflects preprocessing
  divergence at the decision boundary, not systemic detector disagreement.

FILES CHANGED:
  src/yolo26_backends.py            -- _ensure_cuda_dlls_on_path(), provider
                                        capture on DetectionResult, call site
                                        in ONNXYolo26Detector.__init__
  src/export_yolo26_onnx.py         -- provider capture in report,
                                        confidence_diff_distribution
                                        (p50/p90/p95/p99/max/count_gt_0_02),
                                        class_swaps classification
                                        (same_box_competing_class vs
                                        distinct_object), --skip-export flag
  tests/test_yolo26_fusion.py        -- 3 new tests (27/27 pass):
                                        test_ensure_cuda_dlls_on_path_is_idempotent_and_safe
                                        test_car_truck_same_box_are_fusion_equivalent
                                        test_same_box_competing_class_swap_is_not_a_localization_error
  artifacts/yolo26_export/onnx_parity_report.json -- rewritten with full
                                        GPU diagnostics + swap classification

TESTS: 27/27 pass (test_yolo26_fusion.py)
  Run with: .venv\Scripts\python.exe -m pytest
            ai_cv/phases/02_detection_tracking/tests/test_yolo26_fusion.py -q

COMMANDS RUN:
  .venv_yolo26\Scripts\python.exe export_yolo26_onnx.py --skip-export  # GPU parity
  .venv\Scripts\python.exe -m pytest ... test_yolo26_fusion.py -q       # 27/27
  many inline python -c diagnostics for raw ONNX output inspection

LOTO STATUS: UNCHANGED — all 6 folds still infeasible (not rerun; T03 issue
  is upstream of detector backend choice).

NEXT REQUIRED STEPS (in order):
  1. Reviewer decision on ONNX parity: accept documented same-box tiebreaks
     as fusion-equivalent (per plan), OR require strict raw-tensor pass
     (would need end-to-end export head change / per-class NMS alignment).
  2. T03 stratified annotation — classify whether the 750 suppressed
     candidates are real road users YOLO missed or stereo noise. This is
     the fine-tune-vs-reject decision point (plan step: "Detector misses
     true road users: collect box annotations").
  3. If genuine misses: fine-tune YOLO26n on T03-style dark scenes
     (separate phase; 6 fold-specific models, no leakage).
  4. Resolve YOLO26 licensing (AGPL vs commercial) before any product merge.

ERROR: none — all scripts ran cleanly; gate failures are characterized
       measurement results, not runtime errors.
```

### 2026-07-27 16:20 — Bug fix pass + LOTO rerun + ONNX parity rerun

```text
STATUS: BLOCKED
BRANCH: research/phase-04b-yolo26-fusion  (was incorrectly recorded as main)
HEAD:   19b6796 (no new commits — all changes untracked)

BUGS FIXED:
  Fix 1: cross_validate_yolo26_fusion.py -- track state keyed by row idx, not track_id
          consecutive misses never accumulated; guard was effectively disabled
  Fix 2: cross_validate_yolo26_fusion.py -- width reconstructed from height instead of
          selected_width_norm; component bboxes were square, corrupting IoU matching
          added load_candidate_extras() to read track_id + width from candidate CSV
  Fix 3: yolo26_backends.py ONNXYolo26Detector -- output is xyxy not cxcywh
          NMS receives xywh-converted boxes as required by cv2.dnn.NMSBoxes
  Fix 4: export_yolo26_onnx.py -- zero matched detections = sys.exit(1) hard failure
          parity runs on raw kitti/image_2 frames, not annotated overlays
          parity_valid now reflects gate results, not just "comparisons exist"
  Fix 5: cross_validate_yolo26_fusion.py -- infeasible folds recorded explicitly;
          no silent fallback to default config
  Fix 6: .gitignore -- added .venv_yolo26/ (was untracked and unignored)

TESTS: 35 collected, 35 passing
  (test_yolo26_fusion + test_classical_tracking + test_classical_pipeline)
  New tests: test_three_consecutive_misses_accumulate,
             test_miss_resets_after_match,
             test_onnx_zero_match_is_not_silently_perfect

LOTO RERUN: ALL 6 FOLDS INFEASIBLE
  Every semantic config raises T03 MAE from 62.2s toward ~77.0s
  MAE constraint (mean_train_mae <= base_train_mae + 1e-4) fails on every fold
  Composite constraint passes for most configs
  Previous run F1=0.5634 was the physical guard baseline; bugs had disabled fusion

ONNX PARITY RERUN:
  Matched pairs: 86 (previous: 0 — was a hard parser bug, now fixed)
  Class agreement: 96.51%  [GATE FAIL — target >=99%]
  Median IoU:      0.9967   [PASS]
  Mean conf diff:  0.0205   [GATE FAIL — target <=0.02, over by 0.0005]
  parity_valid:    false
  Cause of failures: 3 car/truck label swaps at low confidence; conf diff tail present
  Assessment: cause unverified — must retest on GPU before any conclusion

GIT STATE:
  2 modified: classical_tracking.py, YOLO26_IMPLEMENTATION_LOG.md
  Untracked: all Phase 04B src/tests/artifacts, .venv_yolo26 (now gitignored)
  Safe to commit Phase 04B source files on this branch; do not push venv

NEXT REQUIRED STEPS (in order):
  1. Rebuild venv_yolo26 with CUDA 12.8 torch wheel + onnxruntime-gpu
  2. Rerun ONNX parity on GPU to verify whether gates fail or pass
  3. Annotate stratified T03 sample to confirm whether suppressed candidates
     are genuine road users or stereo noise (required before fine-tuning decision)
  4. If T03 annotations confirm genuine misses: fine-tune YOLO26n on T03-style scenes
  5. Resolve YOLO26 licensing

ERROR: none — all scripts ran to completion; gate failures are measurement results
```

```text
STATUS: PASSED (fixes verified) / BLOCKED (LOTO all infeasible)
STEP:   5-bug fix pass, 35 unit tests, LOTO rerun, GPU env check

BUGS FIXED:
  Fix 1: cross_validate_yolo26_fusion.py -- track state keyed by track_id not row idx
  Fix 2: cross_validate_yolo26_fusion.py -- width from selected_width_norm not height
          (added load_candidate_extras() to read track_id + width from candidate CSV)
  Fix 3: yolo26_backends.py ONNXYolo26Detector -- output is xyxy not cxcywh
          NMS now receives xywh-converted boxes as cv2.dnn.NMSBoxes requires
  Fix 4: export_yolo26_onnx.py -- zero matched detections = hard failure (sys.exit 1)
          parity now runs on raw kitti/image_2 frames, not annotated overlays
  Fix 5: cross_validate_yolo26_fusion.py -- infeasible folds recorded explicitly,
          no silent fallback to default config

TESTS: 35/35 pass
  New: test_three_consecutive_misses_accumulate (regression for track_id bug)
  New: test_miss_resets_after_match
  New: test_onnx_zero_match_is_not_silently_perfect

LOTO RERUN RESULT: ALL 6 FOLDS INFEASIBLE
  Reason: every semantic config raises T03 MAE from 62.2s to 77.0s (+14.8s)
  T03 suppressed 750/1355 candidates -- real danger tracks invisible to YOLO
  Composite constraint passes; MAE constraint fails on every fold
  Previous run's F1=0.5634 "improvement" was invalid -- bugs disabled the guard

GPU ENV:
  RTX 3060 6GB present (driver 572.16, CUDA 12.8) -- confirmed by nvidia-smi
  venv_yolo26 torch version: 2.13.0+cpu (CPU-ONLY BUILD -- incorrect)
  venv_yolo26 ORT version: 1.28.0, providers: [Azure, CPU] -- no CUDA provider
  Fix needed: reinstall torch with CUDA 12.8 wheel + onnxruntime-gpu

COMMANDS RUN:
  .venv\Scripts\python.exe -m pytest ... -v                    # 35/35 pass
  .venv\Scripts\python.exe cross_validate_yolo26_fusion.py ... # all 6 infeasible
  nvidia-smi                                                    # RTX 3060 confirmed
  .venv_yolo26\Scripts\python.exe -c "import torch; ..."       # CPU-only confirmed

FILES CHANGED:
  src/cross_validate_yolo26_fusion.py  -- fix 1,2,5
  src/yolo26_backends.py               -- fix 3
  src/export_yolo26_onnx.py            -- fix 4
  tests/test_yolo26_fusion.py          -- 3 new tests
  artifacts/yolo26_loto/loto_yolo26_fusion_report.json  -- updated (all infeasible)
  artifacts/yolo26_loto/LOTO_YOLO26_FUSION_SUMMARY.md   -- rewritten
  artifacts/yolo26_loto/T05_FP_DIAGNOSIS.md              -- updated note

NEXT:
  1. Rebuild venv_yolo26 with CUDA 12.8 torch wheel + onnxruntime-gpu
  2. Rerun ONNX parity test on raw source images (corrected parser)
  3. Fine-tune YOLO26n on T03-style dark frames (separate phase per plan)
  4. Resolve YOLO26 licensing before product merge

DO NOT: change corridor rules, loosen MAE constraint, or do another threshold sweep
```

```text
STATUS: PASSED
STEP:   T05 FP Diagnosis + classical_tracking.py live pipeline integration
COMMAND:
  .venv\Scripts\python.exe tmp\diagnose_t05_fp.py
  # then manual edits to classical_tracking.py + test_yolo26_fusion.py

RESULT:
  T05 FP ROOT CAUSE (111 FP candidates total):
    57% (63) - depth<=5m, close-range fallback active, soft-guard cannot fire
    15% (17) - depth>5m, YOLO matches component as 'car' conf=0.89-0.94 (real car, frames 481-555)
    23% (26) - depth>5m, YOLO present but no IoU match
     5%  (5) - depth>5m, no YOLO detection at all
  CONCLUSION: NOT a YOLO problem. FPs are real close-range objects and a real car
              correctly detected by YOLO. GT labels them safe (corridor/behavior issue).
              Fine-tuning YOLO will NOT help.

  PIPELINE INTEGRATION:
    classical_tracking.py: Added semantic_state to ComponentTrack,
      enable_semantic(), update_semantic(), is_semantically_suppressed() methods.
      ComponentTracker: semantic_score_threshold/max_misses/fallback_depth_m params,
      detections param to update(), reset() method.
      select_minimum_ttc: soft-guard check (bit-identical when semantic_state is None).
    7 new integration tests added to test_yolo26_fusion.py.
    ALL 33 TESTS PASS (33/33).
    None-backend bit-parity: verified by test_none_backend_semantic_state_is_null
      and test_select_minimum_ttc_none_backend_bit_parity.

FILES:
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto/T05_FP_DIAGNOSIS.md
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto/t05_fp_diagnosis.json
  ai_cv/phases/02_detection_tracking/src/classical_tracking.py
  ai_cv/phases/02_detection_tracking/tests/test_yolo26_fusion.py
  tmp/diagnose_t05_fp.py  (diagnostic script, not committed)

COMMIT: (none — working on main)

NEXT:
  - Step 5 (ONNX parity) and Step 6 (official 5-repeat benchmark) still pending GPU
  - TensorRT FP16 engine build pending target GPU
  - Promotion gates: Macro F1 (0.5634) still below 0.60 and T05 FP (45) unchanged
  - T05 FP is a GT/corridor issue, NOT fixable by YOLO fine-tuning
  - Recommended next action: corridor geometry tightening experiment on T05
    OR accept current LOTO result as best achievable with soft-guard + pretrained YOLO26n

ERROR: none
```

```text
STATUS: PASSED (partial — LOTO gates not all met, research phase complete)
STEP:   Phase 04B — YOLO26 Semantic Fusion research run
COMMAND:
  # LOTO cross-validation
  .venv\Scripts\python.exe ai_cv/phases/02_detection_tracking/src/cross_validate_yolo26_fusion.py \
    --source-root ai_cv/outputs/benchmarks/phase04_loto/source \
    --detections-dir ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/detections \
    --output-dir ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto

  # ONNX export and parity
  .venv_yolo26\Scripts\python.exe ai_cv/phases/02_detection_tracking/src/export_yolo26_onnx.py \
    --model-path yolo26n.pt --onnx-path yolo26n.onnx

RESULT:
  - LOTO: Macro F1=0.5634 (FAIL >=0.60), Composite=39.71 (PASS), MAE=44.806s (PASS),
          T05 FP=45 (FAIL <=20), T03 Recall=0.276 (PASS)
  - ONNX: All parity gates passed (class agreement 100%, IoU 1.0, conf diff 0.0)
  - Unit tests: 15/15 passed
  - Detector reference: 3,600 frames, 72 overlays, per-trip detection CSVs

FILES:
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_loto/loto_yolo26_fusion_report.json
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_export/onnx_parity_report.json
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/detector_reference_summary.json
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/detections/T01..T06-Sample.csv
  ai_cv/phases/02_detection_tracking/artifacts/yolo26_reference/overlays_72/ (72 jpg)
  yolo26n.onnx (D:\Python\, 9.5 MB — git-ignored)

COMMIT: (none — working on main, no commit made per plan)

NEXT:
  - Promotion BLOCKED: Macro F1 (0.5634) below gate (0.60) and T05 FP (45) above gate (20)
  - Per plan: "Reject YOLO fusion if it fails F1 or T03 recall after the fixed 27-policy search"
  - Next action options:
      1. Annotate T03/T05 frames and fine-tune (separate phase per plan)
      2. Fix fusion geometry — if association is failing detector sees objects but doesn't match
      3. Resolve licensing first, then decide whether to invest in fine-tuning
  - Do NOT start another broad threshold sweep

ERROR: none — all scripts ran cleanly
```

## Long-running process

```text
PID:
STARTED:
COMMAND:
STDOUT:
STDERR:
EXPECTED OUTPUT:
```

## Handoff instruction

Ask Codex:

```text
Read D:\Python\docs\YOLO26_IMPLEMENTATION_LOG.md and inspect only files,
commits, and short log tails named there. Continue from NEXT.
```
