# Tasks - Phase 01

- [x] Inventory toàn bộ trip và modality.
- [x] Validate frame/timestamp continuity.
- [ ] Implement causal frame/event access and a future-invariance test.
- [x] Kiểm tra stereo pair count và calibration consistency.
- [ ] Verify pixel-level stereo alignment trên các sample đại diện.
- [x] Audit depth zero/sentinel/saturation và anomaly range của `T01d`.
- [x] Phân tích GT TTC và event trên practice set.
- [ ] Chạy baseline cho `T01-Sample` đến `T06-Sample`.
- [ ] Strict-validate prediction count/ID/timestamp/value trước evaluator.
- [ ] Chạy evaluator và lưu JSON/CSV report.
- [ ] Chọn ít nhất 10 failure cases đại diện.
- [ ] Viết baseline reproduction command.
- [ ] Generate a run manifest and validate stream ordering/run IDs for the baseline.
- [ ] Record wall time, FPS, P50/P95 latency and hardware for the baseline.
- [ ] Freeze baseline metric regression file.
