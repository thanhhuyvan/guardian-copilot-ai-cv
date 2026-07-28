# Phase 05 - Confidence, Risk State and Events

**Status:** IN PROGRESS  
**Depends on:** Phase 04

Phase 05 starts from the classical guarded TTC candidate (`macro F1 0.5634`,
composite `39.71`, compute P95 `54.40 ms`) as its safe fallback. A new
detector-owned depth/TTC ablation reaches macro F1 `0.632`, composite `42.8`,
and live compute P95 `63.22 ms`, but is not yet the sole default because T01
and T02 regress. See
[`artifacts/DETECTOR_OWNED_TTC_ABLATION.md`](artifacts/DETECTOR_OWNED_TTC_ABLATION.md).

A leakage-controlled confidence-router experiment is documented in
[`artifacts/CONFIDENCE_ROUTER_LOTO.md`](artifacts/CONFIDENCE_ROUTER_LOTO.md).
The learned router was rejected (`F1 0.488`). A parameter-free conservative
union reaches `F1 0.658`, but remains an event-layer candidate because T01
false positives increase substantially.

The research basis and fixed first ablation are documented in
[`notes/PAPER_RESEARCH.md`](notes/PAPER_RESEARCH.md).

The intentional mini-fold capacity test is documented in
[`artifacts/minifold_overfit/MINIFOLD_CAPACITY_REPORT.md`](artifacts/minifold_overfit/MINIFOLD_CAPACITY_REPORT.md).
T03 shows usable blocked signal (`F1 0.627`), while T05 does not generalize
enough (`F1 0.426`); do not replace the risk logic with a larger classifier.

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
