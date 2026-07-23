# Phase 01 - Dataset Audit and Baseline

**Status:** COMPLETE  
**Depends on:** Phase 00

Kết luận và quyết định chuyển phase:
[`artifacts/STAGE_01_REPORT.md`](artifacts/STAGE_01_REPORT.md).

## Mục tiêu

Hiểu dữ liệu bằng số liệu, tái lập baseline và tạo mốc metric đáng tin cậy.

Kế hoạch thực thi chi tiết: [`STAGE_01_EXECUTION_PLAN.md`](STAGE_01_EXECUTION_PLAN.md).

## Chạy dataset audit

```powershell
python ai_cv/phases/01_data_baseline/src/audit_dataset.py `
  --practice-root Practice_Dataset `
  --scored-root Hackathon_Dataset_Redacted `
  --output-dir ai_cv/phases/01_data_baseline/artifacts `
  --verify-images all
```

Script chỉ đọc dataset và chỉ ghi report nhỏ vào `artifacts/`.

## Chạy lại baseline end-to-end

```powershell
python ai_cv/phases/01_data_baseline/src/run_baseline.py
```

Command trên chạy đủ 6 practice trip, chuẩn hóa CSV về đúng cột Challenge 1,
strict-validate rồi mới gọi evaluator. Nếu prediction đã tồn tại:

```powershell
python ai_cv/phases/01_data_baseline/src/run_baseline.py --reuse-predictions
```

Chạy regression gate sau evaluation:

```powershell
python ai_cv/phases/01_data_baseline/verify/verify_phase01.py
```

## Visualize baseline failures

Sau khi đã sinh prediction CSV cho 6 practice trip:

```powershell
python ai_cv/phases/01_data_baseline/src/visualize_baseline_failures.py
```

Ảnh và CSV chẩn đoán được ghi vào
`ai_cv/outputs/reports/baseline_official/visualizations/` và không được commit.

## Thử lightweight improvements

```powershell
python ai_cv/phases/01_data_baseline/src/experiment_lightweight_improvements.py
```

Runner tính SGBM một lần mỗi frame, sau đó so sánh các policy temporal/ROI causal
trên cùng disparity và chấm bằng evaluator chính thức.

## Câu hỏi nghiên cứu

- Mỗi modality có đủ frame và đồng bộ không?
- Depth keyframe có đơn vị/range/chất lượng thế nào?
- TTC critical phân bố ở trip/event nào?
- Baseline thất bại ở tình huống nào?

## Test cần có

- Loader đọc đủ 16 trip.
- Frame count, timestamp và modality count đúng.
- Calibration có `fx`, baseline và resolution hợp lệ.
- Baseline sinh đúng số dòng và không NaN.
- Evaluator chạy trên đủ 6 practice trip.

## Verification

- Tạo dataset audit report.
- Lưu metric baseline theo trip và trung bình.
- Có failure-case gallery/video.

## Exit criteria

- [x] Baseline chạy lại bằng một command.
- [x] Sáu CSV pass strict validator trước evaluator.
- [x] Metric baseline `19.7` được đóng băng làm regression gate.
- [x] Có failure evidence, runtime và Stage 2 decision.
