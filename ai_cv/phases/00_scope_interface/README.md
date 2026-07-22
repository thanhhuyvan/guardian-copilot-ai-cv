# Phase 00 - Scope and Interface Freeze

**Status:** DONE — hardened in repository v0.1.1  
**Depends on:** Không có

## Mục tiêu

Khóa phạm vi AI/CV, deployment direction và contract bàn giao trước khi viết pipeline.

## Câu hỏi nghiên cứu

- Product chính là in-car real-time hay out-car/post-trip?
- TTC là core; DMS/lane/risk embedding nằm ở mức nào?
- Hardware và dependency nào được phép?
- Integration cần dữ liệu frame-level hay event-level?

## Test cần có

- JSON Schema và semantic validator đều pass.
- Negative tests chặn missing/extra fields và quan hệ chéo không hợp lệ.
- Payload `valid`, `degraded`, `unknown`, risk event và run manifest đều hợp lệ.

## Verification

- Product direction được ghi thành một câu rõ ràng.
- `perception.v1` và `risk_event.v1` đã được CV owner freeze; integration sign-off là gate trước Phase 05.
- KPI, threshold và Definition of Done được chốt.

## Exit criteria

- Không còn quyết định lớn chưa có owner.
- Interface không phụ thuộc implementation cụ thể.

## Scope đã khóa

- Product narrative chính: out-car Fleet Collision Intelligence/post-trip analytics.
- AI/CV core: road-facing object detection, tracking, depth, closing speed và TTC.
- Pipeline mặc định causal để có thể tái sử dụng cho in-car nếu cần.
- Output chấm điểm: per-frame `predicted_ttc` CSV.
- Output tích hợp ban đầu: versioned JSON/JSONL file; transport service được để integration layer quyết định sau.
- DMS, Safety Kernel, CAN, HMI, dashboard và business recommendation ngoài core TTC.
- Target hardware không chặn nghiên cứu accuracy; mọi experiment được đo runtime sớm, còn latency/hardware hard gate ở Phase 06.
- Mỗi run khai báo `causal_online` hoặc `offline_post_trip` và có manifest truy vết.
