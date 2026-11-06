# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

---

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:**  
   `class_id` = 468 ; `class_name` = "cab" ; `rank` = 1 ; `score` = 0.510915 ; `taxonomy_name` = "ImageNet-1K"

2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**  
   Record này nói về **cả bức ảnh**. Tác vụ phân loại ảnh (Classification) gán một nhãn tổng thể đại diện cho toàn bộ nội dung bức ảnh, không cung cấp tọa độ (bounding box) để định vị riêng cho từng vật thể.

3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**  
   Danh sách tên lớp đi kèm **taxonomy/checkpoint** cố định (ở đây là bộ dữ liệu ImageNet-1K). Mô hình chỉ dự đoán xác suất dựa trên tập các lớp đã được định nghĩa sẵn khi huấn luyện.

4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**  
   Cần ghi đầy đủ để tránh sự mơ hồ khi đối chiếu. `class_id` là mã số chuẩn trong lập trình, `class_name` giúp con người đọc hiểu, và `taxonomy_name` xác định rõ ngữ cảnh định nghĩa lớp (tránh nhầm lẫn khi trùng tên lớp giữa các bộ dữ liệu khác nhau).

5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**  
   Guideline cần quy định rõ **tiêu chí ưu tiên** (Priority Rule): ví dụ chọn chủ đề chính chiếm diện tích lớn nhất ở trung tâm, hoặc ưu tiên chọn theo mục tiêu nghiệp vụ cụ thể (như ưu tiên nhãn phương tiện giao thông chính).

6. **Vì sao không được lấy `score` của model làm ground truth?**  
   Vì `score` chỉ thể hiện độ tin cậy mang tính xác suất của mô hình toán học tại thời điểm dự đoán (mô hình vẫn có thể đoán sai với score cao), còn Ground Truth là giá trị thực tế khách quan do con người xác minh theo chuẩn guideline.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen` (hoặc record mẫu từ bộ detection `traffic`/`kitchen`).

1. **Chép một record:**  
   `class_name` = "bus" ; `score` = 0.912558 ; `bbox_xyxy` = `[93.17, 187.95, 223.01, 320.91]` ; `bbox_width` = 129.84 ; `bbox_height` = 132.96

2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**  
   Hộp có góc trên-trái nằm tại tọa độ pixel $x = 93.17, y = 187.95$, góc dưới-phải nằm tại tọa độ pixel $x = 223.01, y = 320.91$. Chiều rộng của hộp là $129.84\text{ px}$ và chiều cao là $132.96\text{ px}$.

3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**  
   - Ngưỡng `0.20`: Có **17** vật thể (giữ lại nhiều hộp hơn, bao gồm cả các vật thể nhỏ/mờ).
   - Ngưỡng `0.60`: Có **6** vật thể (chỉ giữ lại các hộp mô hình cực kỳ tự tin).

4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**  
   Reviewer phải tốn thời gian lọc và xóa bỏ các hộp dự đoán sai (False Positives), xử lý các hộp bị trùng lặp cho cùng một vật thể, và điều chỉnh lại nhãn bị đoán sai do nhiễu ở ngưỡng thấp.

5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):**  
   *Quy tắc:* Khung chữ nhật (bounding box) phải bao sát viền ngoài cùng của vật thể nhìn thấy, không được cắt mất phần thuộc vật thể và hạn chế tối đa việc chứa thừa khoảng trống nền (không quá 5 pixel nhiễu nền).

6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**  
   - *Hiện tượng:* Người làm bếp đứng sau bàn chế biến chỉ nhìn thấy một phần thân/chân (`person`), hoặc chiếc bát (`bowl`) bị đè một phần.
   - *Quyết định:* Dùng **guideline**.
   - *Vì sao:* Việc vật thể bị che khuất (occlusion) hoặc cắt mép (truncation) là tình huống rất phổ biến trong thực tế. Guideline cần đưa ra quy tắc chuẩn chung (ví dụ: "chỉ gán nhãn khi phần nhìn thấy của vật thể lớn hơn 20%") để annotator tự áp dụng nhất quán thay vì phải báo cáo escalation liên tục.

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen` (hoặc record mẫu từ bộ segmentation `traffic`/`kitchen`).

1. **Chép một record:**  
   `instance_id` = "traffic-001" ; `class_name` = "bus" ; `score` = 0.925745 ; `polygon_point_count` = 120 ; `3–5 điểm đầu của polygon_xy` = `[[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], [142.0, 192.0]]`

2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**  
   Polygon cung cấp đường viền chính xác từng pixel bám theo hình dạng thực tế của đối tượng, giúp phân biệt rõ ranh giới của vật thể với vùng nền hoặc các đối tượng xung quanh thay vì chỉ là một khung hình chữ nhật bao quanh.

3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**  
   `instance_id` dùng để phân biệt các cá thể đối tượng riêng biệt trong cùng một bức ảnh (ví dụ: `traffic-001` và `traffic-002`). Nó không phải `class ID` vì nó mang tính định danh từng vật thể chứ không đại diện cho tên lớp, và không phải `tracking ID` vì nó chỉ có giá trị nội bộ trong bức ảnh hiện tại chứ không liên kết/theo dõi vật thể qua các frame video tiếp theo.

4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo:**  
   *Quy tắc:* Các đỉnh của đa giác (polygon) phải chấm bám sát sát ranh giới tự nhiên của đối tượng, tăng mật độ điểm ở các đoạn cong phức tạp và giữ phẳng ở các cạnh thẳng, không để lấn sang vùng nền hoặc vật thể khác.

5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**  
   - *Hiện tượng:* Chiếc ly chạm vào chai nhựa hoặc ranh giới giữa áo người làm bếp và khung cảnh hậu cảnh mờ đục.
   - *Quyết định:* Dùng **escalation**.
   - *Vì sao:* Khi các vật thể dính liền, che khuất phức tạp hoặc biên quá mờ không thể xác định rõ bằng mắt thường và guideline chưa bao quát case này, annotator cần leo thang (escalation) xin ý kiến Lab Coach/Reviewer để chốt phương án gán nhãn chuẩn, tránh tự bịa gây sai lệch dữ liệu.

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn cấp ảnh duy nhất (`class_id` / `class_name`). | Ảnh đông đúc nhiều phương tiện (`cab`, `bus`, `minibus`), khó chọn 1 nhãn tổng thể. | Quan sát và chọn nhãn đại diện chính duy nhất theo ưu tiên trong guideline. | Kiểm tra nhãn được chọn có đúng tiêu chuẩn ưu tiên trong guideline không. |
| Phát hiện vật thể | Danh sách các hộp `[x_min, y_min, x_max, y_max]` kèm `class_name`. | Bỏ sót các vật thể nhỏ ở xa; khung bboxes đè lên nhau gây nhầm lẫn nhãn. | Vẽ khung bao sát viền vật thể, đảm bảo gán đúng tên lớp. | Kiểm tra độ ôm sát của khung hình chữ nhật và đảm bảo không bỏ sót vật thể. |
| Instance segmentation | Chuỗi điểm `polygon_xy` kèm `instance_id`, `bbox` và `class_name`. | Đường viền phức tạp bị lem ra nền; khó tách biệt ranh giới hai vật đè lên nhau. | Chấm các điểm polygon bám sát từng pixel viền ngoài của đối tượng. | Kiểm tra độ chi tiết của polygon mask, đảm bảo không bị chờm ra nền hay lấn đối tượng khác. |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)**  
   Chỉ sử dụng ảnh công khai được cố định sẵn trong notebook bài lab; tuyệt đối không upload ảnh cá nhân, thông tin nhạy cảm và không ghi họ tên/MSSV vào nội dung báo cáo hay file output.

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**  
   Tôi sẽ dừng công việc ngay lập tức và báo cáo cho Lab Coach hoặc Giảng viên quản lý bài lab.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS