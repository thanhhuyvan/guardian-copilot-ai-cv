# Cross-Phase Tests

- `unit`: reusable component tests.
- `integration`: detector -> tracker -> depth -> TTC -> event.
- `regression`: baseline/model metric and output stability.
- `robustness`: degraded input and fallback behavior.

Phase-specific tests ở trong `phases/<phase>/tests`. Chỉ chuyển test lên đây khi component đã được nhiều phase sử dụng.

