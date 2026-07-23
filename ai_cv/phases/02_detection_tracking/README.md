# Phase 02 - Target Extraction and Tracking

**Status:** STAGE 2A COMPLETE — Stage 2B gate ready  
**Depends on:** Phase 01

## Mục tiêu

Tạo target-level observation và track ổn định cho road objects. Stage 2A ưu
tiên geometry/classical; Stage 2B chỉ bổ sung deep learning khi có failure được
đo rõ.

Thứ tự component:
[`notes/COMPONENT_SEQUENCE.md`](notes/COMPONENT_SEQUENCE.md).

## Component 1 — SGBM confidence

```powershell
python ai_cv/phases/02_detection_tracking/src/analyze_stereo_confidence.py

python ai_cv/phases/02_detection_tracking/src/experiment_lr_consistency.py
```

Kết luận:
[`artifacts/STEREO_CONFIDENCE_001.md`](artifacts/STEREO_CONFIDENCE_001.md).

Hard left-right mask không được promote: composite giảm `19.7 -> 17.8`. LR
residual được giữ làm confidence signal. Ground removal sau đó được đánh giá
trong vertical slice bên dưới.

## Stage 2A classical vertical slice

```powershell
python ai_cv/phases/02_detection_tracking/src/analyze_ground_obstacles.py

python ai_cv/phases/02_detection_tracking/src/experiment_classical_vertical_slice.py

python ai_cv/phases/02_detection_tracking/src/analyze_vertical_slice_results.py
```

Kết luận:
[`artifacts/CLASSICAL_VERTICAL_SLICE_001.md`](artifacts/CLASSICAL_VERTICAL_SLICE_001.md).

Pipeline object-centric tăng danger F1 `0.220 -> 0.402`, tăng TP `80 -> 135` và
giảm FN `124 -> 69`. Tuy nhiên component identity bị fragment mạnh; composite
`28.7` và worst-trip `4.6` chưa vượt robust fixed-ROI reference (`32.2`,
`16.9`). Vì vậy geometry/tracking được giữ, connected components không được
promote làm target identity cuối.

Stage 2B phải tuân theo promotion gate:
[`notes/STAGE2B_EXPERIMENT_GATE.md`](notes/STAGE2B_EXPERIMENT_GATE.md).

Vai trò và shortlist deep learning cho stereo/TTC:
[`artifacts/DEEP_STEREO_RESEARCH_001.md`](artifacts/DEEP_STEREO_RESEARCH_001.md).

## Câu hỏi nghiên cứu

- Confidence stereo nào tương quan với depth/TTC error?
- Instance-aware extractor có giảm identity fragmentation và T03 false positive?
- Object detector hay semantic mask tạo end-to-end TTC gain lớn hơn trên cùng
  compute budget?
- Geometry tracking chịu cut-in, occlusion và frame drop đến đâu sau khi có
  object identity ổn định hơn?

## Test cần có

- LR consistency đúng sign, correspondence và boundary.
- Ground removal không xóa obstacle support.
- Empty frame không crash.
- Track confirmation/expiry đúng config.
- Không tái sử dụng history của track hết hạn.

## Verification

- Mỗi component có visual, quantitative ablation và keep/reject decision.
- Báo cáo component coverage, track continuity và degraded behavior.
- Deep learning phải có comparison với classical candidate mạnh nhất.

## Exit criteria

- Geometry và causal TTC có implementation/test/report tái lập được.
- Classical identity limitation và Stage 2B promotion gate được ghi rõ.
