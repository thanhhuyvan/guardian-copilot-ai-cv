# Phase 03 - Depth, Distance and Relative Motion

**Status:** NOT_STARTED  
**Depends on:** Phase 02

## Mục tiêu

Ước lượng distance và closing speed ổn định theo từng `track_id`.

## Câu hỏi nghiên cứu

- Stereo estimator nào phù hợp ảnh 640x360?
- Vùng nào trong bbox đại diện tốt cho object depth?
- Depth keyframe nên dùng trực tiếp, calibration hay validation?
- Kalman/robust regression/window nào ổn định nhất?

## Test cần có

- Disparity-to-depth đúng theo calibration.
- Invalid disparity không sinh finite depth giả.
- Background/outlier bị loại.
- Track mới không sinh closing speed chắc chắn.
- Timestamp irregular vẫn tính velocity đúng.

## Verification

- Depth error report trên keyframe.
- Distance continuity/jitter report.
- Closing-speed sanity plot theo representative tracks.

## Exit criteria

- Distance và closing speed đủ ổn định để TTC không bị spike hàng loạt.

