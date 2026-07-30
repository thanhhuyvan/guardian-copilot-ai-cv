# Guardian V1 deployment release

## Decision

Deploy **V1 conservative union** for Challenge 1. V2 remains research-only:
its tested EKF, occupancy, association, temporal-depth, ground, and looming
variants did not safely exceed V1 on the available labelled practice data.

## Frozen configuration

- Input: decoded native 640 x 360 stereo pair and causal ego telemetry.
- Stereo: dual SGBM, two OpenCV threads, two matcher workers.
- Semantic detector: YOLO26n PyTorch/CUDA (`yolo26n.pt`).
- TTC policy: conservative union; detector-only danger can override the
  classical fallback.
- Event output: deterministic integrated risk-event FSM.
- Commit used for controlled output: `4e78447b0ba122b63263ab37468e7807007d6e23`.

## Controlled redacted deployment validation

The 10 redacted trips were processed five times with 100 warm-up frames per
trip. One truncated JPEG (`T08d`, frame 1615) was handled by the deterministic
fail-safe path: blank stereo input, tracker/FSM reset, explicit degraded-frame
record, and non-danger TTC output.

| Gate | Result | Status |
|---|---:|:---:|
| Compute P50 / P95 / P99 | 45.98 / 64.23 / 71.38 ms | PASS |
| P95 target | <= 75 ms | PASS |
| Peak RAM | 1,445 MB | Reported |
| Peak VRAM | 464 MB | PASS (< 5 GB) |
| Perception documents | 18,000 | Schema-valid |
| Risk events | 75 | Schema-valid |
| Challenge-1 TTC CSVs | 10 x 1,800 frames | Valid |

Redacted TTC truth is unavailable, so this is a deployment/format/robustness
validation, not a hidden-set F1 claim. The practice-set research result remains
F1 0.654, critical TTC MAE 29.993 s, and composite 42.8/100.

## Release files

The reproducible local submission package is ignored generated output:

- `ai_cv/outputs/phase31_v1_submission_controlled/predictions/guardian_v1/`
- `ai_cv/outputs/phase31_v1_submission_controlled/manifest.json`

## Reproduction

```powershell
D:\Python\.venv_yolo26\Scripts\python.exe `
  ai_cv\phases\05_risk_events\src\evaluate_detector_owned_ttc.py `
  --practice-root D:\Python\Hackathon_Dataset_Redacted `
  --starter-root D:\Python\Package_starterkit\package_starterkit `
  --trips T01d T02d T03d T04d T05d T06d T07d T08d T09d T10d `
  --detector-backend yolo26-pytorch --model-path D:\Python\yolo26n.pt `
  --integrated-union-events --warmup-frames 100 --repeats 5 `
  --opencv-threads 2 --stereo-workers 2 `
  --output-dir ai_cv\outputs\phase31_redacted_controlled_latency_rerun
```

Then use `src/package_challenge1_submission.py` to validate and package the
`conservative_union` CSVs.

## Known limits

- V1 is a scoped collision-risk baseline, not an unrestricted ADAS claim.
- T01 turns/side traffic and T05 false alerts remain known weak scenarios.
- V2 may resume only with independently reviewed object-event/CPA labels and
  an external or synthetic holdout.
- Formal submission and owner sign-off remain human actions.
