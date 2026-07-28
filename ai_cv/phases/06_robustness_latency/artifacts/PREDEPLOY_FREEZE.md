# Pre-deploy freeze

## Candidate

- Branch: `research/phase-08-confidence-temporal-stereo`
- Runtime: dual SGBM plus YOLO26n PyTorch, batch 1, `640x360`
- TTC policy: frozen conservative union with event hysteresis
- Disabled: confidence-temporal, low-ego gates, turn association gate, path-intersection gate
- Latency gate: compute P95 <= `75 ms`

## Certification command

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_predeploy_certification.ps1
```

The command runs all six trips with 100 warm-up frames and five repeats, then
uses the organizer's scorer on `conservative_union`. It writes the runtime
report and `official_evaluation.json` under `ai_cv/outputs/phase08_predeploy_final`.

## Known operating limits

- Accuracy is limited by T01 turns/side traffic and T05 classical stereo TTC.
- Experimental semantic association and path-intersection gates are not part
  of this candidate; both failed their held-out safeguards.
