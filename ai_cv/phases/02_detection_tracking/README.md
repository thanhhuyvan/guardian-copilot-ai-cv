# Phase 02 - Object Detection and Tracking

**Status:** NOT_STARTED  
**Depends on:** Phase 01

## Mục tiêu

Tạo target-level observation và track ổn định cho road objects.

## Câu hỏi nghiên cứu

- Detector pretrained nào cân bằng accuracy/license/latency tốt nhất?
- Taxonomy nào map tốt với car/motorcycle/pedestrian/obstacle?
- Tracker chịu được cut-in, occlusion và frame drop đến đâu?

## Test cần có

- Class mapping deterministic.
- Empty frame không crash.
- Track confirmation/expiry đúng config.
- Không tái sử dụng history của track hết hạn.
- Annotated frame/video khớp output JSON.

## Verification

- So sánh tối thiểu hai detector/config nếu khả thi.
- Báo cáo detection coverage và track continuity.
- Video có bbox, class, confidence và track ID.

## Exit criteria

- Target chính giữ được track đủ dài để ước lượng motion.
- Failure mode và fallback được ghi rõ.

