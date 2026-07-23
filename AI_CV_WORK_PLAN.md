# GuardianCo-Pilot - AI/CV Work Plan

## 1. Mục đích tài liệu

Tài liệu này là kế hoạch thực thi và working contract cho phần AI/CV của GuardianCo-Pilot.
Mục tiêu là xây dựng một Perception Engine có thể:

1. Đọc road-facing stereo video và telemetry có sẵn.
2. Phát hiện, theo dõi và phân loại đối tượng nguy hiểm.
3. Ước lượng khoảng cách, closing speed và Time-To-Collision (TTC).
4. Xuất TTC theo frame để chấm điểm chính thức.
5. Gom các frame nguy hiểm thành collision-risk event có thể giải thích.
6. Cung cấp dữ liệu ổn định cho dashboard, report, clip evidence và các module tích hợp.

Tài liệu chỉ bao phủ AI/CV. Safety Kernel, CAN control, HMI, LLM, cloud và business analytics nằm ngoài phạm vi mặc định.

---

## 2. Phạm vi

### 2.1. Core scope - bắt buộc

- Data audit và validation.
- Road-object detection: car, motorcycle, pedestrian và obstacle nếu model hỗ trợ.
- Multi-object tracking và `track_id` ổn định.
- Stereo/depth estimation.
- Khoảng cách dọc và closing speed.
- TTC theo object và `min_ttc` theo frame.
- Collision corridor để loại object không nằm trên hướng va chạm.
- Temporal filtering và confidence.
- Risk level và event aggregation.
- CSV submission cho `T01d` đến `T10d`.
- JSON/CSV stream log cho integration.
- Annotated video.
- Evaluation, robustness test và latency benchmark.

### 2.2. Auxiliary scope - làm khi core đã ổn định

- Driver Monitoring: alert, drowsy, yawning, distracted, microsleep.
- PERCLOS/eye-state hoặc driver-state smoothing.
- Lane/road corridor nâng cao.
- Fusion TTC với ego speed và event prior.
- ONNX export và quantization.

### 2.3. Stretch goals

- Shared multi-task backbone.
- Learned trajectory prediction.
- Learned Risk Embedding.
- Road-surface/friction estimation.
- Adaptive inference scheduling.

### 2.4. Ngoài phạm vi mặc định

- Quyết định phanh hoặc điều khiển xe.
- Safety Kernel và rule điều khiển CAN.
- ECU/middleware.
- Android Cockpit/HMI.
- Voice explanation/LLM.
- Fleet dashboard hoàn chỉnh.
- UBI pricing, coaching recommendation và route planning.
- Vehicle Health/Battery SOH.

---

## 3. Dữ liệu và ràng buộc thực tế

### 3.1. Practice set

- `Practice_Dataset/T01-Sample` đến `T06-Sample`.
- 600 frame/trip, 20 FPS, khoảng 30 giây.
- Có ground truth TTC và các field đầy đủ.
- Dùng cho development, calibration, ablation và local evaluation.

### 3.2. Scored set

- `Hackathon_Dataset_Redacted/T01d` đến `T10d`.
- 1.800 frame/trip, 20 FPS, khoảng 90 giây.
- TTC, distance, closing speed và driver state đã bị redact.
- Dùng để sinh submission cuối cùng.

### 3.3. Input hợp lệ

- `kitti/image_2`: camera trái, 640x360.
- `kitti/image_3`: camera phải, 640x360.
- `kitti/depth`: depth keyframe, khoảng mỗi 5 frame.
- `kitti/calib`: calibration từng frame.
- `driver`: driver-facing video.
- `ego.speed_kmh`.
- `ego.longitudinal_accel` và `ego.lateral_accel`.
- `target_id`, `target_class` nếu có trong JSON.
- `events_log.type` và `events_log.t`; ở `causal_online` chỉ được dùng record đã xảy ra, không được nạp lịch event tương lai.

### 3.4. Input không được giả định tồn tại

- Brake-pedal state trực tiếp.
- Steering angle trực tiếp.
- GPS/geolocation ở scored set.
- Ground-truth bounding box sử dụng được ở mọi trip.
- Ground-truth TTC hoặc 3D location ở scored set.

---

## 4. Kiến trúc mục tiêu

```text
Stereo frames + telemetry
           |
           v
Object Detector ------> class, bbox, confidence
           |
           v
Multi-Object Tracker --> track_id, trajectory, track quality
           |
           +-------------------------------+
           |                               |
           v                               v
Stereo/Depth Estimator              Collision Corridor
           |                               |
           +---------------+---------------+
                           v
             Distance + Relative Motion
                           |
                           v
               TTC + Confidence Fusion
                           |
                           v
                Temporal Risk State
                           |
              +------------+-------------+
              |                          |
              v                          v
      Frame TTC Stream            Event Aggregator
              |                          |
              v                          v
       Submission CSV        JSON/event clip/annotated video
```

### Nguyên tắc kiến trúc

- TTC phải được tính theo target, không lấy median toàn ảnh làm giải pháp cuối.
- Chỉ target trong collision corridor mới được ưu tiên cho `min_ttc`.
- Tracking và temporal context là bắt buộc để tính closing speed ổn định.
- Quality score phải phản ánh cả detection, depth và tracking quality; không gọi là xác suất khi chưa calibration.
- Product output được sinh từ cùng một inference result với CSV submission.
- Không sử dụng ground truth bị redact, frame tương lai hoặc full future `events_log` khi đánh giá `causal_online`.
- Mỗi run phải có `run_manifest.v1` ghi commit, model/config, dataset, processing mode, input, depth policy và hardware.

---

## 5. Kế hoạch công việc

## Phase 0 - Scope và interface freeze

### Công việc

- Chốt in-car real-time hay out-car/post-trip là câu chuyện chính.
- Chốt core challenge là TTC.
- Chốt schema Perception Stream.
- Chốt dependency policy và trường runtime cần đo; hardware/SLA chính xác được defer sang Phase 06.
- Chốt cách backend nhận stream/event.

### Deliverable

- Schema version `perception.v1`.
- Config threshold ban đầu.
- Danh sách KPI và Definition of Done.

### Exit criteria

- Không còn field hoặc interface quan trọng chưa rõ.

## Phase 1 - Dataset audit và baseline reproduction

### Công việc

- Validate đủ frame và modality của 16 trip.
- Kiểm tra calibration, stereo alignment và depth keyframe.
- Phân tích TTC/event distribution của 6 practice trip.
- Chạy lại baseline trên từng practice trip.
- Lưu report làm mốc so sánh.

### Deliverable

- Dataset audit report.
- Baseline prediction CSV.
- Baseline metric theo từng trip.
- Danh sách failure case có hình/video minh họa.

### Exit criteria

- Pipeline baseline chạy tái lập.
- Metric baseline được ghi lại và không phụ thuộc thao tác thủ công.

## Phase 2 - Target extraction và tracking

### Công việc

- Stage 2A: phân tích stereo confidence, ground removal và obstacle
  components/Stixel-lite.
- Track geometry components causally và đánh giá continuity/occlusion.
- Chỉ mở Stage 2B detector/semantic khi Stage 2A có failure đo được chưa giải
  quyết.
- Khi mở Stage 2B: kiểm tra license, class mapping, latency và giá trị tăng thêm
  so với classical candidate.

### Deliverable

- Classical target extractor và confidence/degraded state.
- Track schema và continuity report.
- Keep/reject decision cho từng component.
- Detector/semantic comparison chỉ khi Stage 2B được kích hoạt.

### Exit criteria

- Target chính được giữ track đủ lâu để tính vận tốc tương đối.
- Không crash khi frame không có detection.

## Phase 3 - Depth, distance và closing speed

### Công việc

- Tính disparity/depth trong ROI từng target.
- Dùng depth keyframe theo policy khai báo trong manifest; direct inference/interpolation cần BTC xác nhận trước final submission.
- Loại road/background/outlier trong bbox.
- Xây robust distance estimator.
- Xây closing-speed estimator theo track.
- Thử median, trimmed median, percentile và robust regression/Kalman.

### Deliverable

- `distance_m` theo track.
- `closing_speed_mps` theo track.
- Depth validation report.

### Exit criteria

- Distance không nhảy vô lý giữa frame liên tiếp.
- Track mới hoặc thiếu history không sinh TTC giả.

## Phase 4 - TTC và collision corridor

### Công việc

- Xác định object nằm trong collision corridor.
- Tính TTC theo object.
- Chọn `min_ttc` hợp lệ theo frame.
- Tối ưu threshold trên practice set.
- So sánh với baseline bằng evaluator chính thức.

### Deliverable

- TTC module.
- CSV cho 6 practice trip.
- Evaluation report và ablation table.

### Exit criteria

- Composite trung bình vượt baseline.
- Không đánh đổi recall bằng số lượng false alarm không kiểm soát.

## Phase 5 - Confidence, risk state và event aggregation

### Công việc

- Xây confidence tổng hợp.
- Xây state machine `SAFE/WARNING/DANGER/CRITICAL/UNKNOWN`.
- Thêm hysteresis và debounce.
- Gom frame liên tục thành event.
- Sinh pre/post-event clip.

### Deliverable

- TTC stream log.
- Collision Risk Event List.
- Clip evidence.
- Per-trip summary JSON.

### Exit criteria

- Một nguy cơ liên tục không bị tách thành nhiều event giả.
- Một TTC spike đơn lẻ không tự động tạo critical event nếu confidence thấp.

## Phase 6 - Robustness và latency

### Công việc

- Test blur, brightness, noise, frame drop và stereo failure.
- Test theo từng event/object/trip.
- Đo P50/P95 latency, throughput, RAM/VRAM.
- Đo TTC jitter và recovery time.
- Xây degraded-mode behavior.

### Deliverable

- Robustness report.
- Latency report.
- Failure/fallback matrix.

### Exit criteria

- Không crash ở bất kỳ test lỗi đầu vào nào.
- Output luôn có trạng thái rõ: valid, degraded hoặc unknown.

## Phase 7 - Submission và handoff

### Công việc

- Sinh `T01d.csv` đến `T10d.csv`.
- Validate 1.800 dòng/trip, frame ID, timestamp và numeric format.
- Sinh annotated demo video.
- Viết README và lệnh chạy từ đầu.
- Freeze config/model version/checksum.
- Bàn giao schema và example payload cho integration.

### Deliverable

- 10 CSV submission.
- Source code tái lập.
- Model artifact/config.
- Demo video.
- Technical report và known limitations.

### Exit criteria

- Pipeline chạy từ raw dataset ra đúng submission mà không sửa tay.

---

## 6. Rule xử lý bắt buộc

Các rule dưới đây phải được triển khai hoặc có test xác nhận rõ ràng.

### A. Input và validation rules

**R-IN-01 - Frame identity**  
Mỗi prediction phải gắn đúng `trip_id`, `frame_id`, `timestamp`. Không được ghép theo thứ tự file nếu frame ID có thể đọc trực tiếp.

**R-IN-02 - Monotonic time**  
Timestamp phải tăng. Nếu timestamp lặp/giảm, temporal history của affected track phải reset hoặc chuyển degraded.

**R-IN-03 - Missing image**  
Nếu thiếu/không đọc được một camera, không crash. Output frame phải mang `status=degraded` hoặc `unknown`.

**R-IN-04 - Stereo mismatch**  
Nếu trái/phải khác kích thước, lệch frame hoặc quality quá thấp, không dùng stereo TTC như kết quả tin cậy.

**R-IN-05 - Calibration required**  
Không tính metric depth bằng stereo nếu thiếu `fx` hoặc baseline. Phải log lỗi rõ ràng.

**R-IN-06 - Trip boundary**  
Mọi tracker, Kalman filter và temporal history phải reset giữa hai trip.

**R-IN-07 - No future leakage**  
Chế độ `causal_online` không được dùng frame tương lai hoặc full future `events_log` để dự đoán frame hiện tại. Smoothing hai chiều chỉ được dùng ở `offline_post_trip` và phải ghi rõ trong manifest.

### B. Detection và tracking rules

**R-DET-01 - Class normalization**  
Mọi class phải qua mapping version hóa. Dataset `vehicle/walker/bike` map mặc định sang `vehicle/pedestrian/two_wheeler`; chỉ refine `bike` thành motorcycle/bicycle khi detector có bằng chứng.

**R-DET-02 - Confidence gate**  
Detection dưới threshold không được tạo critical event trực tiếp.

**R-DET-03 - Track confirmation**  
Track mới cần đủ số hit tối thiểu trước khi được coi là stable, trừ tình huống TTC cực thấp và confidence rất cao.

**R-DET-04 - Track expiry**  
Track mất quá `max_age` phải kết thúc; không tiếp tục ngoại suy vô hạn.

**R-DET-05 - ID switch protection**  
Khi class, bbox hoặc motion thay đổi bất thường, không nối history cũ một cách mù quáng.

**R-DET-06 - No-object frame**  
Nếu không có target hợp lệ trong corridor, internal/CSV dùng `inf`, còn JSON contract dùng `null`; đây không phải lỗi hệ thống.

### C. Depth và distance rules

**R-DEP-01 - Positive disparity**  
Chỉ disparity dương, finite và qua quality gate mới được đổi sang depth.

**R-DEP-02 - Target-specific depth**  
Depth phải lấy trong ROI/mask của target; không dùng median toàn central frame làm kết quả cuối.

**R-DEP-03 - Background rejection**  
Ưu tiên vùng trung tâm/thấp hoặc mask foreground; loại pixel road, sky và background nếu có thể.

**R-DEP-04 - Minimum support**  
Nếu số pixel depth hợp lệ dưới ngưỡng, đánh dấu depth low-confidence thay vì ép ra số.

**R-DEP-05 - Outlier rejection**  
Loại giá trị ngoài physical range và spike không phù hợp trajectory.

**R-DEP-06 - Keyframe fusion**  
Starter kit cung cấp depth `.npy`; mọi cách dùng phải được khai báo trong manifest. Direct/interpolated use cho final submission cần được BTC xác nhận.

**R-DEP-07 - Distance continuity**  
Khoảng cách thay đổi vượt giới hạn vật lý giữa hai frame phải bị giảm confidence hoặc reset estimator.

### D. Closing speed và TTC rules

**R-TTC-01 - Definition**  
Với target đang tiến gần: `TTC = distance / closing_speed`.

**R-TTC-02 - Non-closing target**  
Với phương pháp range-rate, nếu `closing_speed <= epsilon`, internal TTC là `inf` và JSON là `null`. Quy tắc này không áp cho phương pháp TTC trực tiếp từ image scale/flow.

**R-TTC-03 - Insufficient history**  
Track chưa đủ temporal history phải trả internal `inf`, JSON `null`, hoặc quality thấp; TTC trực tiếp một-frame chỉ hợp lệ khi phương pháp và uncertainty được ghi rõ.

**R-TTC-04 - Per-target TTC**  
TTC được tính riêng theo `track_id` để tránh trộn distance của hai vật thể.

**R-TTC-05 - Frame-level TTC**  
`predicted_ttc` của frame là TTC nhỏ nhất trong các target hợp lệ thuộc collision corridor.

**R-TTC-06 - Physical bounds**  
TTC âm, NaN hoặc không hợp lệ phải chuyển thành internal/CSV `inf` và JSON `null`. TTC hữu hạn có thể clip ở mức cấu hình để tránh số bất thường.

**R-TTC-07 - Critical-distance handling**  
Không được tự động loại mọi depth dưới 1,5 m; đây là vùng nguy hiểm nhất và cần fallback riêng.

**R-TTC-08 - Temporal stability**  
Smoothing không được làm trễ danger detection quá mức cấu hình. Raw và filtered TTC nên được log riêng khi debug.

**R-TTC-09 - Corridor priority**  
Target ngoài đường đi không được thắng `min_ttc` chỉ vì đứng gần camera.

**R-TTC-10 - Risk thresholds**  
Mặc định:

- `SAFE`: TTC >= 3,0 s hoặc không có TTC hữu hạn (`inf` internal/CSV, `null` JSON).
- `WARNING`: 2,0 s <= TTC < 3,0 s.
- `DANGER`: 1,5 s <= TTC < 2,0 s.
- `CRITICAL`: TTC < 1,5 s.
- `UNKNOWN`: không đủ dữ liệu hoặc sensor degraded.

### E. Confidence rules

**R-CONF-01 - Composite confidence**  
Confidence TTC phải kết hợp ít nhất detection confidence, depth quality và track quality.

**R-CONF-02 - Missing component**  
Thiếu một thành phần quan trọng phải giảm confidence; không giữ confidence cao mặc định.

**R-CONF-03 - Confidence is not probability by default**  
Nếu chưa calibration, phải gọi đây là quality score thay vì xác suất đúng tuyệt đối.

**R-CONF-04 - Critical action gate**  
TTC thấp vẫn giữ physical severity tương ứng; quality thấp phải được ghi `confidence_level=LOW` và không được tự động promote thành intervention/confirmed event. Quyết định phanh thuộc Safety Kernel, không thuộc CV.

### F. Event aggregation rules

**R-EVT-01 - Event start**  
Event bắt đầu khi risk vượt threshold trong đủ số frame liên tục hoặc có một critical observation với confidence rất cao.

**R-EVT-02 - Event end**  
Event chỉ kết thúc khi TTC trở về vùng safe trong thời gian hold cấu hình.

**R-EVT-03 - Hysteresis**  
Dùng ngưỡng bắt đầu và kết thúc khác nhau để tránh trạng thái nhấp nháy.

**R-EVT-04 - Merge gap**  
Hai đoạn nguy hiểm cùng `track_id` cách nhau dưới `merge_gap` được gộp thành một event.

**R-EVT-05 - Severity**  
Severity của event lấy theo TTC nhỏ nhất đáng tin cậy trong event, không lấy frame cuối.

**R-EVT-06 - Evidence window**  
Clip phải gồm pre-event và post-event buffer để người xem hiểu ngữ cảnh.

**R-EVT-07 - Frame count vs event count**  
Phải phân biệt số frame near-miss của evaluator với số event thực tế của product.

**R-EVT-08 - Event identity**  
Event cần `run_id`, `event_id`, trip, start/end time, min TTC, object type, track ID, severity, `event_quality` và `confidence_level`.

### G. Robustness và fallback rules

**R-ROB-01 - Camera failure**  
Mất một camera: chuyển monocular/degraded nếu có; mất cả hai: `UNKNOWN`, không tự tạo TTC.

**R-ROB-02 - Frame drop**  
Dùng timestamp thật khi ước lượng vận tốc; không giả định mọi khoảng frame luôn là 50 ms.

**R-ROB-03 - Blur/darkness**  
Quality thấp phải giảm confidence và được log.

**R-ROB-04 - Tracker failure**  
Không dùng history của track đã hết hạn cho detection mới.

**R-ROB-05 - Model exception**  
Lỗi một frame không làm dừng trip; phải log và xuất fallback hợp lệ.

**R-ROB-06 - Determinism**  
Cùng model/config/input phải sinh cùng output submission trong sai số xác định.

**R-ROB-07 - Worst-trip visibility**  
Report phải hiển thị cả mean và worst-trip metric, không chỉ điểm trung bình.

### H. Performance rules

**R-PERF-01 - Measure, do not estimate**  
Latency công bố phải là số đo trên phần cứng được ghi rõ.

Runtime thô được đo từ experiment đầu tiên; Phase 06 là nơi áp dụng hard gate và robustness matrix đầy đủ.

**R-PERF-02 - Separate stages**  
Đo riêng decode/preprocess, detection, depth, tracking/TTC và serialization.

**R-PERF-03 - Percentiles**  
Báo cáo P50, P95 và throughput; không chỉ báo latency trung bình.

**R-PERF-04 - In-car target**  
Nếu chọn in-car: throughput mục tiêu >=20 FPS và perception P95 mục tiêu <=50 ms trên target hardware.

**R-PERF-05 - Out-car target**  
Nếu chọn post-trip: báo cáo processing-time/trip và real-time factor thay vì tuyên bố real-time không có benchmark.

### I. Submission rules

**R-SUB-01 - File naming**  
`predictions/<team_name>/<trip_id>.csv`.

**R-SUB-02 - Required columns**  
Tối thiểu: `frame_id,timestamp,predicted_ttc`.

**R-SUB-03 - Row completeness**  
Mỗi scored trip phải có đúng 1.800 frame từ 0 đến 1.799, không duplicate hoặc thiếu.

**R-SUB-04 - Infinity format**  
Dùng `inf` khi không có TTC hữu hạn.

**R-SUB-05 - No manual editing**  
CSV cuối phải được pipeline sinh tự động.

**R-SUB-06 - Extra product fields**  
Confidence/object type có thể nằm trong stream log riêng; không phụ thuộc evaluator sẽ chấm các cột này.

**R-SUB-07 - Version traceability**  
Mỗi submission phải lưu model version, config checksum, code commit hoặc build ID và thời gian sinh.

---

## 7. Metric và tiêu chí lựa chọn model

### 7.1. Metric chính thức Challenge 1

```text
Composite = 40% * MAE-critical score
          + 30% * danger F1 score
          + 30% * inverse-TTC score
```

- Critical MAE: ground-truth TTC < 3 giây.
- Danger F1: TTC < 2 giây.
- Near-miss cho trip score: TTC < 1,5 giây.
- Điểm cuối macro-average đều theo trip.

### 7.2. Metric nội bộ

- MAE-critical và inverse-TTC MAE.
- Precision, recall, F1, false-positive rate.
- Worst-trip composite.
- TTC jitter và số spike giả.
- Danger detection delay.
- Depth MAE trên keyframe.
- Track continuity/ID switch.
- Tỷ lệ frame `UNKNOWN/degraded`.
- Event-level precision/recall.
- P50/P95 latency, FPS, RAM và VRAM.

### 7.3. Model promotion gate

Một experiment chỉ được promote nếu:

1. Tăng composite trung bình hoặc giải quyết failure mode quan trọng.
2. Không làm worst-trip giảm quá ngưỡng chấp nhận.
3. Không tăng false alarm ngoài giới hạn.
4. Không phá vỡ interface hoặc reproducibility.
5. Latency/memory vẫn phù hợp deployment target.

---

## 8. Schema bàn giao

Canonical schemas, semantics and examples nằm tại [`ai_cv/shared/contracts`](ai_cv/shared/contracts/README.md).
Không copy payload vào tài liệu này để tránh hai nguồn contract bị lệch nhau.

---

## 9. Vai trò làm việc

### CV Owner - người dùng

- Chốt scope, product direction và deployment target.
- Là đầu mối với team integration/product.
- Chốt trade-off lớn về model, dependency, accuracy và latency.
- Cung cấp hoặc hỗ trợ chạy GPU/hardware ngoài workspace khi cần.
- Hiểu, trình bày và bảo vệ technical decisions/kết quả.
- Phê duyệt submission cuối.

### AI/CV Engineer - Codex

- Phân tích dataset/evaluator/proposal.
- Thiết kế và triển khai pipeline AI/CV.
- Tạo experiment, test, metric report và failure analysis.
- Sinh CSV, JSON, annotated video và tài liệu kỹ thuật.
- Kiểm tra reproducibility, robustness và interface.
- Báo rõ assumption, limitation và trade-off.
- Không tự ý mở rộng sang module ngoài CV.

---

## 10. Definition of Done

Phần AI/CV hoàn thành khi:

- [ ] Dataset audit không còn lỗi blocker.
- [ ] Baseline được tái lập và lưu metric.
- [ ] Core detector/tracker/depth/TTC chạy end-to-end.
- [ ] Composite practice set vượt baseline hoặc có bằng chứng cải thiện failure mode chính.
- [ ] Có report mean và từng trip.
- [ ] Có TTC stream với confidence/object type.
- [ ] Có event aggregation và clip evidence.
- [ ] Có annotated demo video.
- [ ] Có robustness/fallback tests.
- [ ] Có latency benchmark phù hợp deployment direction.
- [ ] Có đủ 10 CSV scored trip, mỗi file 1.800 dòng.
- [ ] CSV được sinh tự động từ raw dataset.
- [ ] Có README, config, dependency và lệnh chạy.
- [ ] Có model/config version traceability.
- [ ] Schema bàn giao được integration team xác nhận.
- [ ] Known limitations được ghi rõ.

---

## 11. Thứ tự ưu tiên khi thiếu thời gian

1. TTC submission correctness.
2. TTC metric trên practice set.
3. Detection/tracking đúng target trong collision corridor.
4. Robust distance và closing speed.
5. Temporal stability và event aggregation.
6. Annotated video và integration payload.
7. Robustness/latency report.
8. DMS auxiliary.
9. ONNX/INT8.
10. Risk Embedding, road surface và các stretch goal.

Nguyên tắc cuối cùng: không hy sinh một pipeline TTC đo được, tái lập được và có output dùng được để đổi lấy một kiến trúc rộng nhưng không thể chứng minh bằng metric hoặc demo.
