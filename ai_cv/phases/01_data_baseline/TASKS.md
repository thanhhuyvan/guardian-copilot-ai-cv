# Tasks - Phase 01

- [x] Inventory toàn bộ trip và modality.
- [x] Validate frame/timestamp continuity.
- [x] Implement causal temporal policy and a future-invariance test.
- [x] Kiểm tra stereo pair count và calibration consistency.
- [x] Visual-check stereo/disparity alignment trên các sample đại diện.
- [x] Audit depth zero/sentinel/saturation và anomaly range của `T01d`.
- [x] Phân tích GT TTC và event trên practice set.
- [x] Chạy baseline cho `T01-Sample` đến `T06-Sample`.
- [x] Strict-validate prediction count/ID/timestamp/value trước evaluator.
- [x] Chạy evaluator và lưu JSON/CSV report.
- [x] Chốt 6 failure cases được chẩn đoán sâu, phủ 4 trip.
- [x] Viết baseline reproduction command.
- [x] Generate run manifest và validate source ordering.
- [x] Record wall time, FPS và hardware; chuyển percentile latency sang Stage 2.
- [x] Freeze baseline metric regression file.

## Deferred to Stage 2

- P50/P95/P99 per-frame instrumentation trên từng candidate.
- Mở rộng failure catalog khi method mới tạo failure mode mới.
- Exhaustive stereo-confidence/calibration study.
