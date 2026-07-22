# Phase 07 - Submission and Handoff

**Status:** NOT_STARTED  
**Depends on:** Phase 06

## Mục tiêu

Sinh deliverable cuối tái lập được và bàn giao interface ổn định.

## Câu hỏi nghiên cứu

- Một command có thể sinh đủ 10 CSV không?
- Model/config/code version có truy vết được không?
- Integration có tái chạy và đọc output mà không sửa tay không?

## Test cần có

- Đủ 10 CSV, đúng tên và đúng 1.800 dòng/trip.
- Frame ID 0..1799, không duplicate/thiếu.
- Timestamp monotonic.
- TTC parse được hoặc là `inf`.
- Không có GT/leakage trong submission.
- Clean-environment smoke test.

## Verification

- Submission manifest/checksum.
- Reproduction command.
- Annotated demo video và reports.
- README/known limitations hoàn chỉnh.

## Exit criteria

- Raw dataset -> submission chạy tự động.
- CV Owner và integration owner ký nhận deliverable.

