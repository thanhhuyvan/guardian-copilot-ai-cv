# Phase 01 - Dataset Audit and Baseline

**Status:** RESEARCHING  
**Depends on:** Phase 00

## Mục tiêu

Hiểu dữ liệu bằng số liệu, tái lập baseline và tạo mốc metric đáng tin cậy.

Kế hoạch thực thi chi tiết: [`STAGE_01_EXECUTION_PLAN.md`](STAGE_01_EXECUTION_PLAN.md).

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

- Baseline chạy lại bằng một command.
- Metric baseline được dùng làm regression gate.
