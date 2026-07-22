# Phase 05 - Confidence, Risk State and Events

**Status:** NOT_STARTED  
**Depends on:** Phase 04

## Mục tiêu

Biến TTC stream thành output sản phẩm ổn định và có thể giải thích.

## Câu hỏi nghiên cứu

- Kết hợp detection/depth/track quality thành confidence thế nào?
- Hysteresis/debounce nào tránh event nhấp nháy?
- Pre/post clip buffer bao lâu là đủ ngữ cảnh?

## Test cần có

- Risk threshold boundary tests.
- Hysteresis không oscillate quanh ngưỡng.
- Merge gap gộp đúng event cùng track.
- Severity lấy theo minimum trusted TTC.
- Frame near-miss count tách biệt event count.

## Verification

- TTC stream log parse được.
- Event JSON đúng schema.
- Clip có đủ pre/post context.
- Per-trip summary khớp event list.

## Exit criteria

- Event output đủ dùng cho dashboard/report mà không cần đọc lại raw model state.

