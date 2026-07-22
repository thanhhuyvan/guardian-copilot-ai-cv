# Phase 00 - Scope and Interface Freeze

**Status:** RESEARCHING  
**Depends on:** Không có

## Mục tiêu

Khóa phạm vi AI/CV, deployment direction và contract bàn giao trước khi viết pipeline.

## Câu hỏi nghiên cứu

- Product chính là in-car real-time hay out-car/post-trip?
- TTC là core; DMS/lane/risk embedding nằm ở mức nào?
- Hardware và dependency nào được phép?
- Integration cần dữ liệu frame-level hay event-level?

## Test cần có

- Schema example parse được.
- Required field không thiếu.
- Version field có mặt.
- Payload `valid`, `degraded`, `unknown` đều hợp lệ.

## Verification

- Product direction được ghi thành một câu rõ ràng.
- `perception.v1` và `risk_event.v1` được integration owner xác nhận.
- KPI, threshold và Definition of Done được chốt.

## Exit criteria

- Không còn quyết định lớn chưa có owner.
- Interface không phụ thuộc implementation cụ thể.
