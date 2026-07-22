# Phase 04 - TTC and Collision Corridor

**Status:** NOT_STARTED  
**Depends on:** Phase 03

## Mục tiêu

Tính TTC đúng target, chọn `min_ttc` hợp lệ và vượt baseline theo evaluator chính thức.

## Câu hỏi nghiên cứu

- Collision corridor xác định bằng lane, geometry hay heuristic nào?
- TTC smoothing nào giữ recall nhưng giảm false alarm?
- Threshold và confidence gate tối ưu thế nào trên 6 practice trip?

## Test cần có

- Non-closing target trả `inf`.
- Insufficient history không tạo TTC giả.
- Target ngoài corridor không thắng `min_ttc`.
- Negative/NaN TTC chuyển `inf`.
- Frame-level TTC là min của target hợp lệ.

## Verification

- Chạy evaluator trên đủ 6 practice trip.
- Báo MAE-critical, F1, inverse-TTC và composite từng trip.
- So sánh baseline và ablation.

## Exit criteria

- Composite trung bình vượt baseline hoặc cải thiện failure mode được phê duyệt.
- Worst-trip và false alarm nằm trong gate.

