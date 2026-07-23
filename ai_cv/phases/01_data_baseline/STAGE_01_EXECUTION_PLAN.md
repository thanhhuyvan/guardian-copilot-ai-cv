# Stage 01 Execution Plan — Dataset Audit and Baseline

> **Closure note (2026-07-23):** Stage 01 đã hoàn tất. Đây là execution plan
> ban đầu; kết quả thực tế, thay đổi gate và deliberate deferrals được ghi tại
> [`artifacts/STAGE_01_REPORT.md`](artifacts/STAGE_01_REPORT.md).

## 1. Mục tiêu

Stage 01 chưa tối ưu thuật toán TTC. Mục tiêu là tạo một nền đo đáng tin cậy trước khi
nghiên cứu detector, depth hoặc TTC mới:

1. Hiểu chính xác dataset có gì, thiếu gì và field nào bị redacted.
2. Phân loại input nào deployment-realistic, input nào chỉ dùng nghiên cứu/evaluation.
3. Tái lập nguyên bản organizer baseline và local evaluator trên 6 practice trips.
4. Đóng băng metric, runtime và failure cases làm mốc so sánh cho các phương pháp sau.

Stage 01 chỉ được đánh dấu `DONE` khi người khác có thể chạy lại baseline bằng một command
và thu được cùng kết quả trong tolerance đã ghi.

## 2. Những gì đã biết — và vẫn phải audit lại

| Nhóm | Practice set | Scored/redacted set | Kỳ vọng cần kiểm tra |
|---|---|---|---|
| Trip | `T01-Sample`…`T06-Sample` | `T01d`…`T10d` | Đủ 16 trip |
| Độ dài | Khoảng 600 frame/trip | Khoảng 1.800 frame/trip | 20 FPS, timestamp cách khoảng 0,05 s |
| Road camera | `image_2`, `image_3` | `image_2`, `image_3` | Stereo 640×360, frame pair đầy đủ |
| Camera calibration | Global và per-frame KITTI | Global và per-frame KITTI | `fx > 0`, stereo baseline khoảng 0,30 m |
| Depth | Keyframe `.npy` | Keyframe `.npy` | Shape/alignment/range/spacing chưa được giả định |
| KITTI labels | Full hơn | 3D location bị zero-out | Bbox/class có mặt không đồng nghĩa nên dùng lúc inference |
| Telemetry | Speed, longitudinal/lateral acceleration | Tương tự | Finite, đúng đơn vị và đồng bộ timestamp |
| TTC ground truth | Có | Bị redacted | Chỉ practice GT được dùng để tune/evaluate |
| Event log | Type/time và params đầy đủ hơn | Type/time, params redacted | Full schedule không được dùng trong causal prediction |
| Driver video/state | Có | Video có, GT state bị redacted | Chỉ inventory; ngoài core TTC của Stage 01 |

Con số trong bảng là kỳ vọng từ starter kit, không phải kết quả audit. Report phải ghi số
thực tế và anomaly thay vì tự động ép dataset khớp tài liệu.

Read-only preflight hiện quan sát thấy 6 practice trip có 600 frame/120 depth keyframe,
10 scored trip có 1.800 frame/360 depth keyframe, timestamp step 0,05 s, `fx=320` và
baseline 0,30 m. Left/right/driver/calib/label counts khớp frame count, ID bắt đầu từ 0,
ảnh là 640×360×3 `uint8`; finite-GT counts của 6 practice trip lần lượt là
`57, 35, 61, 121, 66, 262`, còn scored set không có finite TTC GT. Đây chỉ là sanity
snapshot; audit script vẫn phải tự xác minh và ghi bằng chứng thay vì hard-code các số này.

Preflight cũng đã thấy hai anomaly cần trở thành regression case, không được “clean” âm thầm:

- Depth là `float32` 360×640 nhưng có sentinel khoảng `1000 m`; finite không đồng nghĩa valid.
- `T01d` có 117/360 depth keyframe toàn zero từ frame 1215 đến 1795, trong khi road images
  cùng đoạn vẫn bình thường. Metadata description của trip này nói “30s” nhưng dữ liệu thực
  tế dài khoảng 90 s.

Hai dataset root phải được truyền riêng. Không giả định một `HackathonDataset` parent chung
vì workspace hiện tách `Practice_Dataset` và `Hackathon_Dataset_Redacted`.

## 3. Data-use policy cho Stage 01

| Loại | Ví dụ | Stage 01 được dùng thế nào |
|---|---|---|
| Deployable core input | Road images trái/phải, calibration, telemetry hiện tại/quá khứ | Được dùng cho causal baseline |
| Organizer auxiliary input | Depth keyframe, JSON target ID/class, event/event-active metadata, KITTI bbox/class | Audit/ablation riêng; phải khai báo manifest, không mặc định là final product input |
| Practice ground truth | `min_ttc`, target motion/TTC, collision cone, risk, aggregate | Chỉ evaluation, distribution và failure analysis; tuyệt đối không đưa vào predictor |
| Future information | Future frame, future event schedule, two-sided interpolation | Cấm trong `causal_online` |
| Out of core | Driver-facing inference/DMS | Chỉ kiểm tra file tồn tại; không nghiên cứu model ở Stage 01 |

Để giữ câu chuyện vision-based có khả năng triển khai, final TTC candidate không nên phụ
thuộc `label_2`, `target_id`, `target_class` hay full `events_log` dù các field đó có thể
được organizer cung cấp. Nếu thử chúng, kết quả phải mang nhãn `oracle` hoặc
`competition_specific`, không so lẫn với causal/deployable baseline.

Trong scored JSON, target ID/class và event type/time vẫn tồn tại nhưng target motion, TTC
và JSON bbox đã bị redacted; KITTI `label_2` là nguồn annotation riêng. `events_active` và
target metadata mặc định chỉ dùng phân tích cho đến khi có decision rõ ràng.

## 4. Work breakdown và gate

### S1.1 — Inventory và schema/redaction audit

**Câu hỏi**

- Mỗi trip có bao nhiêu frame và bao nhiêu file cho từng modality?
- Frame ID đầu/cuối, file naming và JSON frame list có khớp không?
- Practice và scored khác nhau chính xác ở key/giá trị nào?
- Có file hỏng, file thừa, frame thiếu hoặc modality lệch không?

**Công việc**

- Inventory 16 trip: JSON, left/right image, driver image, depth, calib và label.
- So sánh schema/key availability giữa practice và scored, không log giá trị nhạy cảm.
- Kiểm tra redacted TTC/3D/risk/driver fields đúng như starter documentation.
- Đối chiếu metadata description/duration với frame count và timestamp thực tế.
- Gắn mỗi field vào một loại trong data-use policy ở trên.

**Artifacts**

- `artifacts/dataset_inventory.csv`: một dòng/trip với modality counts.
- `artifacts/field_availability.csv`: field × practice/scored × allowed use.
- `artifacts/dataset_audit.md`: anomaly, mức ảnh hưởng và quyết định xử lý.

**Gate S1.1**

- Đủ 16 trip và mọi count mismatch đều được giải thích.
- Không có practice GT nào nằm trong danh sách inference features.
- Không sửa hoặc copy dataset gốc vào repo.

### S1.2 — Temporal, stereo, calibration và depth audit

**Câu hỏi**

- Frame/timestamp có liên tục, tăng đơn điệu và gần 20 FPS không?
- Stereo pair có cùng shape và cùng frame ID không?
- Calibration có nhất quán với ảnh và baseline 30 cm không?
- Depth keyframe có thực sự align với left image và phân bố cách bao nhiêu frame?

**Công việc**

- Kiểm tra duplicate/gap/out-of-order frame ID và timestamp delta.
- Decode sample đầu/giữa/cuối và toàn bộ file header để phát hiện ảnh hỏng.
- Kiểm tra `K_left`, `P2/P3`, `fx`, baseline, resolution và disparity sign.
- Với depth: shape, dtype, finite ratio, zero ratio, sentinel `1000 m`, min/median/P95/max,
  saturation và keyframe spacing; flag riêng đoạn zero-depth của `T01d`.
- So sánh stereo depth với depth keyframe trên ROI mẫu; đây là quality audit, không tune.

**Artifacts**

- `artifacts/temporal_calibration_audit.md`.
- `artifacts/depth_keyframe_summary.csv`.
- Gallery ảnh chỉ lưu local dưới `ai_cv/outputs/reports/phase01/`; không commit ảnh dataset.

**Gate S1.2**

- Không còn anomaly calibration/timestamp chưa có quyết định.
- Chọn và ghi một `depth_keyframe_policy` ban đầu: ưu tiên `validation_only` cho baseline;
  direct/interpolated use chưa được promote nếu chưa có zero/sentinel handling và BTC confirmation.

### S1.3 — Ground-truth distribution trên practice set

**Câu hỏi**

- Bao nhiêu frame có TTC hữu hạn?
- Phân bố `CRITICAL < 1,5`, `DANGER < 2`, `WARNING < 3`, `SAFE/inf` theo trip?
- Object/event nào gây nguy hiểm và practice set có mất cân bằng không?
- Trip nào khó nhất và không được phép chỉ báo cáo mean?

**Công việc**

- Thống kê TTC bins, class, collision-cone targets, ego speed và event type.
- Tính duration của từng danger segment, không chỉ đếm frame.
- Xác định data coverage gap: class/event/lighting nào không có hoặc quá ít.
- Chỉ đọc GT trong analytics/evaluator, tách module khỏi predictor.

**Artifacts**

- `artifacts/practice_gt_distribution.csv`.
- `artifacts/practice_gt_analysis.md`.

**Gate S1.3**

- Có thống kê per-trip và aggregate.
- Xác nhận/giải thích preflight imbalance: trên 3.600 practice frame hiện thấy khoảng
  283 frame `<3 s`, 204 frame `<2 s`, 136 frame `<1,5 s`.
- Biết rõ metric nào dễ bị một trip hoặc class chi phối.

### S1.4 — Reproduce organizer baseline nguyên bản

**Baseline cần hiểu**

```text
left/right image
  -> StereoSGBM disparity
  -> fixed central ROI
  -> median valid metric depth
  -> 5-frame linear regression of depth
  -> closing speed
  -> TTC = depth / closing speed, otherwise inf
```

Các tham số reference không được tune trong lần chạy chuẩn: ROI `35–65%` chiều ngang,
`50–85%` chiều dọc; depth `1,5–80 m`; temporal window `5`; minimum closing speed
`0,3 m/s`; SGBM `numDisparities=96`, `blockSize=11`.

Không sửa ba file starter `baseline_ttc_predictor.py`, `dataset_loader.py` và
`evaluation.py`. Phase 01 viết wrapper riêng, ghi SHA256 của cả ba file và pin Python,
OpenCV, NumPy vì SGBM có thể thay đổi theo dependency version.

**Trình tự chạy**

1. Smoke test `T01-Sample`; wrapper ghi normalized CSV chỉ gồm các cột Challenge 1 cần thiết.
2. Chạy strict prediction validation trước evaluator.
3. Chạy lại `T01-Sample` lần hai; prediction phải deterministic trong cùng environment.
4. Chạy đủ `T01-Sample`…`T06-Sample`, tạo predictor mới/reset giữa các trip.
5. Chạy organizer-provided local evaluator trên chính normalized CSV đã round/write ra disk,
   không chấm array chưa serialize trong memory.
6. Đo wall time, FPS, per-frame P50/P95 và ghi hardware/software.
7. Chạy prefix/future-invariance test: thay đổi dữ liệu sau thời điểm `t` không được đổi prediction trước hoặc tại `t`.
8. Sinh `run_manifest.v1` với commit, checksum baseline/evaluator/config và input profile.

`evaluation.py` không phải strict submission validator: nó có thể chỉ score frame có mặt,
skip row lỗi và để duplicate frame ID ghi đè. Do đó CSV phải pass các invariant trước khi
được chấm:

- Đúng chính xác source frame count: 600 practice hoặc 1.800 scored theo trip thực tế.
- Frame ID đầy đủ, unique, tăng dần và không có missing/extra frame.
- Timestamp khớp source frame trong tolerance đã định.
- TTC không âm/NaN; chỉ chấp nhận số hữu hạn hoặc literal `inf`.
- Final submission chỉ giữ cột challenge cần thiết; `ground_truth_ttc` local không được xuất.

Local composite là regression reference của team; không gọi nó là leaderboard/final score
cho đến khi BTC xác nhận evaluator parity.

README baseline score chỉ là minh họa, không phải golden metric. Một preflight không kiểm
soát trên `T01-Sample` đã cho composite `30,6`, MAE-critical `58,595` và F1 `0,125`, khác
con số minh họa `52,3`. Vì vậy regression chỉ được freeze từ controlled run có Python,
OpenCV, NumPy, OS/CPU và checksum source được ghi đầy đủ. Preflight này cũng đo được khoảng
`20,65 FPS`, compute P50/P95 `22,1/28,9 ms`; đó là observation, không phải latency target.

**Metric bắt buộc**

- Evaluator: critical MAE (`GT TTC < 3 s`), inverse-TTC MAE, danger precision/recall/F1
  (`TTC < 2 s`), FPR và composite score.
- Report cả per-trip, macro mean và worst trip.
- Diagnostic: finite-prediction ratio, `inf` ratio trong danger frames, TTC jitter,
  longest missed-danger segment và detection delay.
- Runtime: trip wall time, FPS, P50/P95/P99 latency và hardware; chưa áp SLA cứng.

**Artifacts**

- Generated CSV: `ai_cv/outputs/predictions/phase01_baseline/` — ignored, không commit.
- Generated evaluator/runtime output: `ai_cv/outputs/reports/phase01/` và
  `ai_cv/outputs/benchmarks/phase01/` — ignored.
- Commit bản nhỏ: `artifacts/baseline_metrics.json`, `artifacts/baseline_runtime.md`,
  `artifacts/run_manifest.json`, `artifacts/baseline_regression.json`.

**Gate S1.4**

- Sáu CSV pass strict count/ID/timestamp/value validation rồi evaluator mới được chạy.
- Hai lần chạy cùng config cho cùng prediction trong tolerance.
- Causal test và manifest pass.
- Metric/runtime đủ sáu trip, không chỉ có số aggregate.
- Không random-split frame; mọi phân tích/generalization split theo trip hoặc event để tránh
  temporal leakage.

Regression tolerance khởi đầu theo độ chính xác evaluator: MAE `±0,001`, inverse-TTC
`±0,0001`, F1 `±0,001`, composite `±0,1`. Chỉ nới tolerance khi có bằng chứng numeric
variation giữa environment, không nới để che regression.

### S1.5 — Failure analysis

Baseline có các failure hypotheses cần xác minh, không sửa ngay trong experiment reference:

- Fixed ROI trộn road/background với target và không biết collision corridor.
- Median depth không nhất thiết đại diện target nguy hiểm nhất.
- SGBM yếu ở vùng texture thấp, occlusion, ánh sáng khó hoặc disparity sai.
- `1,5 m` minimum depth và disparity range có thể làm mất tình huống rất gần.
- Invalid-depth frame giữ history cũ; gap/frame drop có thể tạo stale closing speed.
- Không có object association, ego-motion compensation hoặc lateral-motion reasoning.
- Exception per frame bị chuyển thành `inf`, có thể che lỗi hệ thống thành “safe”.

**Case selection tối thiểu**

- Danger false negative/prolonged `inf`.
- False positive TTC thấp khi GT safe.
- Critical frame có absolute error lớn.
- TTC spike/jitter hoặc delayed detection quanh event boundary.
- Ít nhất 10 case, phủ tối thiểu 4 trip và các event hiện có nếu đủ sample:
  `lead_brake`, `pedestrian_jaywalk`, `motorcycle_cut_in`, `stopped_vehicle_ahead`.

**Artifacts**

- `artifacts/baseline_failure_catalog.md`: trip/frame range, pred/GT, metric impact,
  hypothesized cause và method candidate sẽ xử lý.
- Local-only gallery/video dưới `ai_cv/outputs/reports/phase01/`.

**Gate S1.5**

- Mọi đề xuất Stage 02–04 phải liên kết tới ít nhất một failure case đo được.
- Không chọn model mới chỉ vì popularity hoặc leaderboard claim.

### S1.6 — Freeze reference và nghiệm thu Stage 01

**Công việc**

- Chốt command tái lập từ raw local dataset.
- Ghi dependency versions, commit, config/checksum và hardware.
- Tạo regression fixture với tolerance hợp lý; không hard-code elapsed time tuyệt đối.
- Chạy verification checklist và cập nhật experiment registry với kết luận giữ/loại.

**Gate cuối**

- Dataset audit đủ 16 trip và không còn anomaly nghiêm trọng chưa có owner.
- Baseline đủ 6 practice trip, deterministic và causal theo test.
- Có per-trip/mean/worst-trip metrics, runtime và failure catalog.
- Có run manifest và regression reference.
- Không có dataset, image, depth, video hoặc generated prediction bị commit.
- Scored set chỉ dùng integrity/smoke output sau khi method/config đã freeze; không dùng để
  tuning không nhãn.

## 5. Thứ tự thực thi đề xuất

```text
S1.1 Inventory/redaction
        |
        v
S1.2 Temporal/calibration/depth
        |
        +------> S1.3 Practice GT distribution
        |                  |
        v                  v
S1.4 Baseline reproduction + evaluator + runtime
        |
        v
S1.5 Failure catalog
        |
        v
S1.6 Regression freeze and Stage 01 verification
```

Không chạy tuning/model research trước khi S1.4 có reference metrics. Depth audit và GT
distribution có thể chạy song song sau khi inventory/redaction gate đã pass.

## 6. Target commands cần được implement

Các command dưới đây là interface mục tiêu; chúng chưa được coi là tồn tại cho đến khi có
script và test tương ứng:

```powershell
python ai_cv/phases/01_data_baseline/src/audit_dataset.py `
  --practice-root Practice_Dataset `
  --scored-root Hackathon_Dataset_Redacted

python ai_cv/phases/01_data_baseline/src/run_baseline.py `
  --practice-root Practice_Dataset `
  --output-root ai_cv/outputs/predictions/phase01_baseline

python ai_cv/phases/01_data_baseline/verify/validate_predictions.py `
  --predictions-root ai_cv/outputs/predictions/phase01_baseline `
  --data-root Practice_Dataset

python ai_cv/phases/01_data_baseline/verify/verify_phase01.py
```

Stage 01 implementation bắt đầu bằng `audit_dataset.py`, không bắt đầu bằng detector hoặc
tuning SGBM.
