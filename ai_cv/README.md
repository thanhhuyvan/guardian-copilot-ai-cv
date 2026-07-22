# AI/CV Research Workspace

Workspace này chia công việc AI/CV thành các phase độc lập, có test và verification riêng.

## Thứ tự nghiên cứu

1. `phases/00_scope_interface`
2. `phases/01_data_baseline`
3. `phases/02_detection_tracking`
4. `phases/03_depth_motion`
5. `phases/04_ttc_corridor`
6. `phases/05_risk_events`
7. `phases/06_robustness_latency`
8. `phases/07_submission_handoff`

Không bắt đầu phase sau khi exit criteria của phase trước chưa đạt, trừ experiment thăm dò không ảnh hưởng pipeline chính.

## Cấu trúc chuẩn của mỗi phase

```text
<phase>/
|- README.md     Mục tiêu, câu hỏi nghiên cứu, test và exit criteria
|- TASKS.md      Checklist công việc
|- src/          Code chỉ thuộc phase
|- tests/        Unit/integration tests của phase
|- verify/       Script/checklist nghiệm thu phase
|- artifacts/    Report, metric, biểu đồ và output nhỏ
`- notes/        Research notes và decision log
```

## Thư mục dùng chung

- `shared/contracts`: schema giữa CV và integration.
- `shared/configs`: config dùng chung, không hard-code trong code.
- `shared/utils`: utility đã ổn định và được nhiều phase sử dụng.
- `tests`: test toàn pipeline.
- `verification`: kiểm tra cấu trúc và nghiệm thu cuối.
- `outputs`: prediction, event, video, report và benchmark sinh ra.
- `models`: model artifact; không commit weight lớn nếu repo không cho phép.
- `notebooks`: notebook khám phá, không phải production pipeline.
- `docs`: architecture, experiment registry và tài liệu bàn giao.

## Quy ước trạng thái phase

- `NOT_STARTED`: chưa nghiên cứu.
- `RESEARCHING`: đang đọc/thử nghiệm.
- `IMPLEMENTING`: đang viết pipeline.
- `VERIFYING`: đang chạy test và metric.
- `DONE`: đạt toàn bộ exit criteria.
- `BLOCKED`: có blocker được ghi rõ trong `notes/decision_log.md`.

## Quy tắc chung

- Mỗi experiment phải có config, metric và kết luận giữ/loại.
- Code dùng cho submission không được chỉ tồn tại trong notebook.
- Artifact lớn và dataset không được copy vào workspace này.
- Mọi phase phải có test cho happy path, edge case và failure path.
- Verify phải tạo bằng chứng định lượng hoặc file report, không chỉ “chạy không lỗi”.
- Tham chiếu rule đầy đủ tại `../AI_CV_WORK_PLAN.md`.

## Kiểm tra cấu trúc

```powershell
powershell -ExecutionPolicy Bypass -File .\ai_cv\verification\check_structure.ps1
```

