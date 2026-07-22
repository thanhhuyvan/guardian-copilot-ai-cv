# Tasks - Phase 00

- [x] Chốt out-car là product narrative chính; giữ core causal để tái sử dụng in-car.
- [x] Chốt core/auxiliary/stretch scope ở mức AI/CV.
- [x] Chốt accuracy-first trong research; target hardware/latency gate chuyển sang Phase 06.
- [x] Tạo `perception.v1.schema.json`.
- [x] Tạo `risk_event.v1.schema.json`.
- [x] Tạo payload examples cho valid/degraded/unknown.
- [x] Chốt threshold SAFE/WARNING/DANGER/CRITICAL.
- [x] Freeze file-based contract v1 cho AI/CV; transport integration review sau.
- [x] Ghi decision log và chuyển status sang DONE.

## Stage 00.1 — Contract hardening

- [x] Thêm run manifest cho model/config/commit/data/mode/hardware traceability.
- [x] Tách rõ `causal_online` và `offline_post_trip`; chặn future-frame và full future event schedule ở causal mode.
- [x] Map taxonomy dataset `vehicle/walker/bike` sang contract classes.
- [x] Cho phép TTC bằng 0 và chuẩn hóa TTC/risk/quality semantics.
- [x] Thêm semantic validator cho bbox, min TTC, severity, ordering và quality label.
- [x] Thêm negative tests và chạy trong CI.
- [x] Thêm runtime guardrails từ experiment đầu tiên; hard gate vẫn ở Phase 06.
