# Conventions

## Naming

- Python module/file: `snake_case.py`.
- Config: `<component>.<variant>.yaml`.
- Experiment: `EXP-<phase>-<number>`; ví dụ `EXP-04-003`.
- Report: `<trip_or_scope>_<metric>_<model_version>.<ext>`.
- Model version: `<component>-v<major>.<minor>.<patch>`.

## Test levels

- Unit: một hàm/class, input nhỏ và deterministic.
- Integration: nhiều component nối với nhau.
- Regression: bảo vệ metric/output đã đạt.
- Robustness: input lỗi, nhiễu hoặc degraded.
- Verification: kiểm tra exit criteria bằng output/report thực tế.

## Artifact policy

- `artifacts/`: output nhỏ phục vụ phase.
- `outputs/`: deliverable tổng hợp.
- Dataset giữ nguyên tại thư mục gốc, không sao chép.
- Model weight lớn phải có nguồn, license, checksum và cách tải/tạo lại.

## Decision log

Mỗi quyết định lớn phải ghi:

- Ngày và người quyết định.
- Vấn đề.
- Các lựa chọn đã cân nhắc.
- Metric/bằng chứng.
- Quyết định.
- Hệ quả và điều kiện xem xét lại.

