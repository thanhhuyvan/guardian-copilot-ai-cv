# Verify - Phase 01

- [x] Audit report đủ 16 trip, frame/timestamp, image integrity, depth và calibration.
- [x] Stereo/disparity alignment được visual verify trên sample đại diện.
- [x] Baseline chạy bằng một command.
- [x] Sáu prediction CSV pass strict validator trước evaluator.
- [x] Causal future-invariance test và run manifest pass.
- [x] Có metric đủ 6 practice trip.
- [x] Có mean và worst-trip score.
- [x] Có failure-case catalog.
- [x] Có wall-time/FPS kèm hardware, chưa áp SLA cứng.

**Gate:** PASS. Xem `artifacts/STAGE_01_REPORT.md` để biết các hạng mục được
chuyển có chủ đích sang Stage 2.
