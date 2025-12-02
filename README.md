# README — Giải thích `tomato_leaf.ipynb`

## TÓM TẮT TỔNG QUAN

### 1. MỤC ĐÍCH BÀI TOÁN

**Vấn đề thực tiễn**: Trong nông nghiệp hiện đại, việc phát hiện sớm và chính xác các bệnh trên cây trồng (đặc biệt là cà chua) là cực kỳ quan trọng để:
- Giảm thiệt hại kinh tế cho nông dân
- Giảm lượng thuốc trừ sâu (bảo vệ môi trường và sức khỏe)
- Tăng năng suất và chất lượng nông sản

**Bài toán Machine Learning**: Xây dựng hệ thống **Object Detection** (phát hiện đối tượng) tự động nhận diện và phân loại các bệnh trên lá cà chua thông qua ảnh chụp. Hệ thống cần:
- Xác định **vị trí** (bounding box) vùng bệnh trên lá
- Phân loại **loại bệnh** (10 lớp: 9 bệnh + 1 lá khỏe mạnh)
- Đánh giá hiệu năng của 2 thuật toán SOTA: **YOLOv8s** (real-time) vs **Faster R-CNN** (high accuracy)

### 2. MỤC ĐÍCH CỦA CODE

Notebook `tomato_leaf.ipynb` thực hiện **pipeline hoàn chỉnh** cho bài toán Computer Vision từ A-Z:

**Giai đoạn 1: Chuẩn bị dữ liệu (Data Preparation)**
- Tổ chức lại dataset theo chuẩn YOLO (images/ và labels/)
- Chia dataset: 70% train, 15% val, 15% test (stratified split)
- Phân tích phân bố lớp (class imbalance analysis)
- Tạo file cấu hình `tomato.yaml` cho YOLO

**Giai đoạn 2: Huấn luyện YOLOv8s (YOLO Training)**
- Load pretrained weights `yolov8s.pt` (transfer learning)
- Fine-tune trên tomato dataset (15 epochs, 256px, batch 16)
- Đánh giá trên test set (mAP@50, Precision, Recall, F1)
- Visualize training curves (loss, mAP) và confusion matrix

**Giai đoạn 3: Huấn luyện Faster R-CNN (Baseline Comparison)**
- Tạo custom PyTorch Dataset cho detection task
- Load pretrained Faster R-CNN ResNet-50-FPN (COCO weights)
- Fine-tune classifier head cho 11 classes (10 diseases + background)
- Huấn luyện 10 epochs với SGD optimizer

**Giai đoạn 4: So sánh và Đánh giá (Evaluation & Comparison)**
- Tính toán metrics chuẩn: mAP@50, Precision, Recall, F1-Score
- Vẽ biểu đồ so sánh trực quan giữa 2 models
- Visualize kết quả dự đoán trên ảnh thực tế
- Phân tích trade-off giữa tốc độ và độ chính xác

### 3. PHỐI HỢP GIỮA CODE VÀ TEMPLATE BÁO CÁO PDF

File PDF `ỨNG DỤNG DEEP LEARNING TRONG PHÁT HIỆN BỆNH TRÊN LÁ CÀ CHUA.pdf` là **template báo cáo học thuật** phục vụ mục đích:
- Trình bày lý thuyết nền tảng (Introduction, Literature Review)
- Mô tả phương pháp luận (Methodology)
- Báo cáo kết quả thực nghiệm (Results & Discussion)

**Cách phối hợp Code ↔ PDF Template**:

| Phần trong PDF | Nội dung từ Code | Cách lấy |
|----------------|------------------|----------|
| **1. Introduction** | Mô tả bài toán, dataset (10 classes) | Từ `CLASS_NAMES` và phân tích `analyze_distribution()` |
| **2. Dataset** | Số lượng ảnh train/val/test, phân bố lớp, imbalance ratio | Output của `prepare_data()` và `analyze_distribution()` |
| **3. Methodology** | Kiến trúc YOLOv8 & Faster R-CNN, hyperparameters | Code blocks huấn luyện (epochs, imgsz, batch, optimizer) |
| **4. Experiments** | Training curves (loss, mAP theo epoch) | `results.csv` từ YOLO và `rcnn_loss_history` |
| **5. Results** | Bảng so sánh mAP@50, Precision, Recall, F1 của 2 models | Biến `yolo_map50`, `rcnn_map50`, `yolo_f1`, `rcnn_f1` |
| **6. Visualization** | Confusion matrix, ảnh dự đoán có bounding boxes | `confusion_matrix.png` và output từ cell 10 |
| **7. Discussion** | Phân tích ưu/nhược điểm YOLOv8 vs Faster R-CNN | So sánh tốc độ (FPS), accuracy, use cases |

**Workflow chuẩn**:
1. **Chạy code** → Thu thập tất cả kết quả (metrics, biểu đồ, ảnh)
2. **Chụp/Lưu outputs** → Copy vào các section tương ứng trong PDF
3. **Viết phân tích** → Giải thích ý nghĩa của kết quả trong phần Discussion
4. **Tổng kết** → Conclusion + Future Work

**Lưu ý quan trọng**:
- Code sinh ra **raw data** (số liệu, hình ảnh)
- PDF cần **diễn giải** raw data thành insights có ý nghĩa
- Báo cáo tốt = Code chạy đúng + Phân tích sâu sắc

---

## HƯỚNG DẪN SỬ DỤNG

Mục tiêu: giải thích chi tiết từng ô (cell) của notebook `tomato_leaf.ipynb` để bạn hiểu luồng làm việc, từng hàm, và phần thuật toán mà không cần chạy lại mã. Nội dung viết bằng tiếng Việt, kèm chú giải về các khái niệm như YOLO label format, IoU, mAP, Precision/Recall/F1, Faster R-CNN, và các lưu ý khi chạy trên Colab hoặc local.

**File**: `d:\colab\tomato_leaf.ipynb`

**Yêu cầu môi trường**:
- Chạy tốt trên Google Colab (đã mount Google Drive trong notebook).
- Thư viện chính: `ultralytics` (YOLOv8), `torch`, `torchvision`, `torchmetrics`, `scikit-learn`, `opencv-python`, `matplotlib`, `pyyaml`.
- Nếu chạy local Windows, cài Python 3.8+ và GPU CUDA (nếu muốn tăng tốc). Ví dụ lệnh cài đặt Colab (đã có trong notebook):

```powershell
!pip install ultralytics torchmetrics
!pip install -U scikit-learn
```

**Cấu trúc tổng quan của notebook**
- Chuẩn bị: mount Drive, import thư viện, set `device`.
- Chuẩn hóa dataset (copy images & labels sang cấu trúc YOLO), chia train/val/test.
- Viết file cấu hình YOLO (`tomato.yaml`).
- Phân tích phân bố lớp dựa trên file nhãn YOLO, vẽ biểu đồ.
- Huấn luyện YOLOv8s (sử dụng `ultralytics.YOLO`).
- Tạo dataset cho Faster R-CNN (PyTorch `Dataset`), huấn luyện Faster R-CNN.
- Đánh giá và so sánh hai mô hình (mAP@50, Precision, Recall, F1), trực quan hóa kết quả.

**Giải thích từng ô / hàm / khối mã**

**1) Cài đặt & import (ô đầu)**
- Mục đích: cài gói cần thiết và import các thư viện.
- Lưu ý: `!pip` chỉ chạy trong môi trường notebook (Colab). Trên Windows PowerShell, dùng `pip install` bình thường.

**2) Mount Google Drive & chọn device**
- Code mount Drive để truy cập dataset lưu trên Drive.
- `device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` chọn GPU nếu có.

**3) Biến cấu hình và danh sách lớp**
- `ROOT_DIR` là nơi chứa `images/` và `labels/` (nhãn YOLO *.txt).
- `WORK_DIR` là nơi sao chép và lưu cấu trúc mới `images/train, images/val, images/test` và `labels/...`.
- `CLASS_NAMES`: danh sách tên lớp (phải khớp với chỉ số trong nhãn YOLO).

**4) Hàm `prepare_data()`**
- Mục tiêu: tạo cặp (image, label) hợp lệ, chia ngẫu nhiên thành train/val/test theo tỷ lệ 70/15/15 và sao chép vào `WORK_DIR`.
- Cơ chế:
  - Duyệt tất cả `.jpg` trong `RAW_IMAGES`, kiểm tra file label tương ứng `.txt` trong `RAW_LABELS`.
  - Dùng `train_test_split` để chia: trước chia test 15%, còn lại chia train/val ~15% cho val.
  - Nếu `WORK_DIR` đã tồn tại và số lượng file khớp, hàm sẽ bỏ qua sao chép để tránh ghi đè tốn thời gian.
- Lưu ý: giữ `random_state` cố định để tái lập kết quả.
- Output: thư mục `WORK_DIR` với cấu trúc phù hợp YOLO.

**5) Hàm `write_yolo_yaml()`**
- Mục tiêu: tạo file config `tomato.yaml` để YOLOv8 biết đường dẫn dữ liệu, số lớp và tên lớp.
- Nội dung YAML gồm `path`, `train`, `val`, `test`, `nc`, `names`.
- Lưu ý: `nc` phải bằng số lớp (ở đây 10), `names` tương ứng với chỉ số nhãn.

**6) Hàm `plot_two_charts(class_counts, class_names, split_name, save_path=None)`**
- Mục tiêu: vẽ 2 biểu đồ (bar đứng và bar ngang) cho phân bố nhãn.
- Input:
  - `class_counts`: Counter mapping class_id -> count
  - `class_names`: danh sách tên lớp
  - `split_name`: chuỗi để hiển thị tiêu đề (Training/Validation/Test)
  - `save_path` (tùy chọn)
- Lưu ý: mã tính `pad` để đặt text nhãn cách cột.

**7) Hàm `analyze_distribution(labels_dir, split_name)`**
- Mục tiêu: đọc tất cả file label `.txt` trong `labels_dir`, thống kê:
  - `class_counts` tổng số annotation mỗi lớp,
  - `images_with_multiple_classes` số ảnh có nhiều hơn 1 lớp,
  - `total_annotations` tổng annotations.
- Cách đọc file YOLO label: mỗi dòng `class_id cx cy w h` với tọa độ normalized (0..1).
- Sau khi tính toán in ra bảng tóm tắt và gọi `plot_two_charts` để vẽ rồi lưu ảnh.
- Lưu ý: tỉ lệ mất cân bằng được tính bằng `max_c / min_c` trên các lớp có >0 annotation.

--- Thuật toán & định dạng nhãn YOLO (giải thích chi tiết)
- Mỗi dòng file nhãn YOLO: `class_id cx cy w h`:
  - `class_id`: số nguyên bắt đầu từ 0.
  - `(cx, cy)`: tâm hộp giới hạn (relative to image width/height, normalized 0..1).
  - `(w, h)`: chiều rộng và chiều cao hộp (normalized).
- Để chuyển sang bbox dạng `[x1, y1, x2, y2]` theo kích thước ảnh:
  - x1 = (cx - w/2) * W, y1 = (cy - h/2) * H, x2 = (cx + w/2) * W, y2 = (cy + h/2) * H.
- Lưu ý quan trọng: nhãn YOLO sử dụng class_id bắt đầu từ 0. Khi dùng Faster R-CNN, code đã cộng thêm `+1` vì PyTorch detection models mong nhãn bắt đầu từ 1 (0 dành cho background).

**8) Khối huấn luyện YOLOv8s**
- Dùng `ultralytics.YOLO` (ví dụ `YOLO('yolov8s.pt')`) để tải weights sẵn có và gọi `.train(...)`.
- Tham số chính:
  - `data=yaml_path`: file cấu hình dataset
  - `epochs`, `imgsz`, `batch`, `project`, `name`.
- Mô hình YOLOv8 sẽ tự sinh các biểu đồ training và lưu runs trong `runs/detect/...` (confusion matrix, results.csv).

**9) Khối đánh giá YOLO**
- Sau huấn luyện, gọi `model_yolo.val(split='test')` để đánh giá trên tập test.
- Kết quả chứa nhiều chỉ số trong `metrics_yolo.box` như `map50`, `map` (map50-95), `mp` (precision), `mr` (recall).
- F1 được tính thủ công từ precision & recall: F1 = 2 * (P*R)/(P+R).

**10) Khối đọc `results.csv` và vẽ trực quan**
- `results.csv` chứa loss & metric theo epoch do YOLO xuất ra.
- Notebook tìm folder `runs/detect/*` mới nhất, lấy `results.csv` và `confusion_matrix_normalized.png` để hiển thị.
- Vẽ loss curves và mAP curves nếu các cột tồn tại.

**11) Lớp dataset cho Faster R-CNN: `TomatoRCNNDataset`**
- Mục tiêu: tạo `torch.utils.data.Dataset` trả về `img_tensor` và `target` dạng dict như PyTorch detection API mong đợi.
- `__init__(self, img_dir, label_dir, width, height, transforms=None)` lưu đường dẫn và danh sách ảnh.
- `__getitem__(self, idx)`:
  - Đọc ảnh bằng `cv2`, chuyển sang RGB, resize về `(width, height)`.
  - Dùng `T.ToTensor()` để chuyển sang tensor có shape `[C,H,W]` và normalized 0..1.
  - Đọc file label `.txt`, chuyển mỗi dòng YOLO -> `[x1,y1,x2,y2]` trên kích thước ảnh resized.
  - `labels` được cộng `+1` để reserve 0 cho background.
  - Nếu ảnh không có object, `boxes` là `torch.zeros((0,4))` và `labels` là `torch.zeros((0,))`.
  - `target` chứa `"boxes"`, `"labels"`, `"image_id"`.
- `collate_fn` dùng để DataLoader ghép batch trả về tuple of lists theo requirement của detection models.

**12) Hàm `get_rcnn_model(num_classes)`**
- Tải `fasterrcnn_resnet50_fpn(pretrained=True)` từ `torchvision.models.detection`.
- Lấy `in_features` từ `roi_heads.box_predictor.cls_score.in_features` và thay predictor bằng `FastRCNNPredictor(in_features, num_classes)` để phù hợp số lớp dataset.
- `num_classes` ở đây = 11 (10 disease + 1 background).

**13) Khối huấn luyện Faster R-CNN**
- Chuẩn bị `train_loader_rcnn` và `val_loader_rcnn`.
- Dùng optimizer SGD (lr=0.005, momentum=0.9, weight_decay=0.0005).
- Vòng lặp huấn luyện cơ bản:
  - `model_rcnn.train()`
  - Với mỗi batch, chuyển images & targets sang `device`.
  - `loss_dict = model_rcnn(images, targets)` trả về dict các mất mát (classification loss, box regression, etc.).
  - `losses = sum(loss_dict.values())` rồi backward & step.
- Ghi lại `rcnn_loss_history` để vẽ loss curve sau này.

**14) Hàm `calculate_f1_rcnn(loader, model, device, conf_threshold=0.5, iou_threshold=0.5)`**
- Mục tiêu: tính Precision/Recall/F1 cho Faster R-CNN bằng phép khớp đơn giản giữa bbox dự đoán và GT.
- Cách hoạt động:
  - Dự đoán `outputs = model(images)`; với mỗi ảnh, lấy `pred_boxes`, `pred_scores`, `pred_labels`.
  - Lọc các dự đoán theo ngưỡng confidence `conf_threshold`.
  - Nếu không có GT box: mọi dự đoán là FP.
  - Nếu không có dự đoán: mọi GT box là FN.
  - Tính IoU giữa `pred_boxes` và `gt_boxes` sử dụng `torchvision.ops.box_iou`.
  - Với mỗi pred box, tìm GT có IoU lớn nhất; nếu IoU >= `iou_threshold` và nhãn trùng và GT chưa được matched thì TP++, else FP++.
  - Sau kiểm tra hết predictions, FN += số GT chưa matched.
- Cuối cùng tính precision = TP/(TP+FP), recall = TP/(TP+FN), F1 theo công thức chuẩn.
- Lưu ý: đây là cách đánh giá thủ công đơn giản, có thể khác so với cách tính mAP đầy đủ (mAP xử lý IoU thresholds và AP per class).

**15) Đo lường mAP với `torchmetrics.detection.MeanAveragePrecision`**
- `MeanAveragePrecision(iou_type="bbox")` cung cấp cách tính mAP chuẩn.
- Cần format `preds` và `targets` theo spec của thư viện (boxes, scores, labels).

**16) Khối so sánh & vẽ biểu đồ**
- So sánh `yolo_map50` vs `rcnn_map50`, `yolo_f1` vs `rcnn_f1`.
- Vẽ 3 biểu đồ (map, f1, loss curve Faster R-CNN).

**17) Khối trực quan kết quả (Visualization Final Block)**
- Lấy 1 ảnh từ `WORK_DIR/images/test` chọn ngẫu nhiên.
- Dự đoán bằng YOLO (`model_yolo.predict(sample_img_path, imgsz=256)`), hiển thị ảnh do YOLO vẽ.
- Dự đoán bằng Faster R-CNN: đọc ảnh, resize 256x256, ToTensor, model_rcnn(img_tensor).
- Vẽ các hộp có score >0.5 lên ảnh và hiển thị 2 ảnh song song.
- Fix trong notebook: xử lý vị trí text khi box sát mép trên (đặt text xuống dưới nếu y1 < 15).

**Các khái niệm thuật toán quan trọng (tóm tắt ngắn)**
- IoU (Intersection over Union): tỉ lệ giao/ hợp giữa bbox dự đoán và bbox thực. Dùng làm tiêu chí khớp.
- Precision: TP / (TP + FP) — tỷ lệ dự đoán đúng trên tổng dự đoán.
- Recall: TP / (TP + FN) — tỷ lệ dự đoán đúng trên tổng GT.
- F1-score: 2 * (P*R)/(P+R) — trung bình điều hòa Precision và Recall.
- mAP@50: mean Average Precision với ngưỡng IoU=0.5. mAP@50-95 là trung bình mAP với IoU từ 0.5 đến 0.95.

---

## NỀN TẢNG LÝ THUYẾT CHI TIẾT CỦA 2 THUẬT TOÁN

### 1. YOLOv8 (You Only Look Once version 8)

#### 1.1. Giới thiệu tổng quan
YOLO là họ thuật toán Object Detection (phát hiện đối tượng) ra đời từ năm 2015 bởi Joseph Redmon. Khác với các phương pháp cổ điển chia làm nhiều giai đoạn (như R-CNN), YOLO xử lý bài toán phát hiện đối tượng như một **bài toán hồi quy duy nhất** (single regression problem), dự đoán trực tiếp bounding boxes và xác suất lớp trong một lần truyền qua mạng (one-shot).

YOLOv8 là phiên bản mới nhất (2023) do Ultralytics phát triển, kế thừa kiến trúc YOLOv5 nhưng được tối ưu về độ chính xác, tốc độ và khả năng triển khai.

#### 1.2. Kiến trúc tổng quan YOLOv8

YOLOv8 bao gồm 3 thành phần chính:

**a) Backbone (Xương sống - trích xuất đặc trưng)**
- Sử dụng kiến trúc **CSPDarknet** (Cross Stage Partial Darknet) với các khối C2f (CSPLayer with 2 convolutions).
- Mục đích: trích xuất đặc trưng đa tỷ lệ (multi-scale features) từ ảnh đầu vào.
- Qua nhiều lớp convolution, pooling, ảnh được giảm kích thước dần và tạo ra feature maps ở các độ phân giải khác nhau (ví dụ: 80×80, 40×40, 20×20 cho ảnh 640×640).
- **Cơ chế hoạt động**: 
  - Input: ảnh RGB shape `[B, 3, H, W]`
  - Output: nhiều feature maps ở các scale khác nhau, mã hóa thông tin từ chi tiết nhỏ (texture) đến ngữ cảnh lớn (object-level).

**b) Neck (Cổ - kết hợp đặc trưng đa tỷ lệ)**
- Sử dụng cấu trúc **PAN (Path Aggregation Network)** và **FPN (Feature Pyramid Network)**.
- Mục đích: trộn thông tin từ các tầng feature khác nhau để cải thiện khả năng phát hiện đối tượng ở nhiều kích thước.
- **PAN**: truyền thông tin từ bottom-up (từ tầng cao xuống tầng thấp) để bổ sung ngữ cảnh.
- **FPN**: truyền thông tin từ top-down (từ tầng thấp lên tầng cao) để tăng độ phân giải chi tiết.
- Kết quả: tạo ra 3 đầu ra feature maps (thường gọi là P3, P4, P5) tương ứng với 3 kích thước đối tượng: nhỏ, trung bình, lớn.

**c) Head (Đầu - dự đoán kết quả)**
- YOLOv8 sử dụng **Decoupled Head** (tách rời) và **Anchor-free**.
- Thay vì dùng anchor boxes cố định như YOLOv5, YOLOv8 dự đoán trực tiếp tâm đối tượng và kích thước box.
- Mỗi cell trong feature map dự đoán:
  - **Bounding box**: 4 giá trị `(x, y, w, h)` (tâm và kích thước).
  - **Objectness score**: xác suất có đối tượng trong cell đó.
  - **Class probabilities**: vector xác suất cho mỗi lớp (10 lớp trong trường hợp này).
- **Anchor-free**: không cần định nghĩa trước anchor boxes, giúp mô hình linh hoạt và dễ huấn luyện hơn.

#### 1.3. Cơ chế hoạt động chi tiết

**Bước 1: Chia ảnh thành lưới (Grid)**
- Ảnh đầu vào được chia thành lưới S×S (ví dụ 20×20).
- Mỗi ô lưới (cell) chịu trách nhiệm phát hiện đối tượng có tâm nằm trong ô đó.

**Bước 2: Dự đoán Bounding Boxes**
- Với mỗi cell, mạng dự đoán nhiều bounding boxes (thường 3 boxes/cell tương ứng 3 scale).
- Mỗi box được mô tả bởi:
  - `(tx, ty)`: offset từ góc trên-trái của cell → tâm box.
  - `(tw, th)`: log-space width/height.
  - Công thức tính tọa độ thực:

$$
\begin{align}
b_x &= \sigma(t_x) + c_x \quad \text{(với $c_x$: cột của cell)} \\
b_y &= \sigma(t_y) + c_y \quad \text{(với $c_y$: hàng của cell)} \\
b_w &= e^{t_w} \\
b_h &= e^{t_h}
\end{align}
$$

**Bước 3: Tính Objectness và Class Probability**
- **Objectness**: xác suất từ 0→1 rằng cell đó chứa đối tượng.
- **Class probability**: với 10 lớp bệnh, mạng xuất vector 10 chiều qua softmax.

**Bước 4: Áp dụng Non-Maximum Suppression (NMS)**
- Sau khi có hàng ngàn boxes dự đoán, NMS loại bỏ các boxes trùng lặp:
  - Sắp xếp boxes theo objectness × class_prob.
  - Giữ box có score cao nhất, loại bỏ các boxes khác có IoU > ngưỡng (thường 0.45).
- Kết quả: chỉ giữ lại các boxes tốt nhất, không overlap quá nhiều.

#### 1.4. Hàm mất mát (Loss Function)

YOLOv8 sử dụng 3 thành phần loss:

**a) Classification Loss (CLS Loss)**
- Đo sai lệch giữa phân bố lớp dự đoán và nhãn thực.
- Dùng **Binary Cross Entropy** (BCE) hoặc **Focal Loss** để xử lý class imbalance.

**b) Localization Loss (Box Loss)**
- Đo độ chính xác vị trí bounding box.
- YOLOv8 dùng **CIoU Loss** (Complete IoU):
  - CIoU = IoU loss + Distance penalty + Aspect ratio penalty
  - Mục tiêu: boxes dự đoán không chỉ overlap tốt mà còn có tâm gần nhau và tỷ lệ khung hình đúng.

**c) Objectness Loss (OBJ Loss)**
- Đo xác suất có/không có đối tượng trong cell.
- Chỉ áp dụng cho các cell có ground truth object.

**Tổng Loss:**

$$
\mathcal{L}_{\text{total}} = \lambda_1 \cdot \mathcal{L}_{\text{cls}} + \lambda_2 \cdot \mathcal{L}_{\text{box}} + \lambda_3 \cdot \mathcal{L}_{\text{obj}}
$$

(với $\lambda$ là các hệ số cân bằng, tự động điều chỉnh theo dataset)

#### 1.5. Ưu điểm và hạn chế

**Ưu điểm**:
- **Tốc độ cao**: xử lý 1 ảnh chỉ mất 2-5ms trên GPU (đạt 200+ FPS), phù hợp realtime.
- **Dễ huấn luyện**: anchor-free, ít hyperparameter cần tune.
- **Linh hoạt**: hỗ trợ nhiều task (detection, segmentation, classification, pose estimation).
- **Tích hợp tốt**: Ultralytics cung cấp API đơn giản, dễ export sang ONNX, TensorRT, CoreML.

**Hạn chế**:
- **Khó phát hiện object nhỏ sát nhau**: do mỗi cell chỉ dự đoán số box giới hạn.
- **Kém chính xác hơn 2-stage detectors** trong một số trường hợp phức tạp (nhưng gap đã giảm đáng kể ở YOLOv8).

---

### 2. Faster R-CNN (Region-based Convolutional Neural Network)

#### 2.1. Giới thiệu tổng quan

Faster R-CNN là thuật toán Object Detection thuộc họ **2-stage detectors**, ra đời năm 2015 bởi Shaoqing Ren, Kaiming He và nhóm Microsoft Research.

Khác với YOLO (1-stage), Faster R-CNN chia bài toán thành 2 bước rõ ràng:
1. **Stage 1**: Tạo các vùng đề xuất (Region Proposals) có khả năng chứa đối tượng.
2. **Stage 2**: Phân loại và tinh chỉnh (refine) các proposals đó.

Kiến trúc này cho phép đạt độ chính xác cao nhưng tốc độ chậm hơn YOLO.

#### 2.2. Kiến trúc chi tiết Faster R-CNN

Faster R-CNN gồm 4 thành phần chính:

**a) Backbone Network (CNN trích xuất đặc trưng)**
- Thường dùng **ResNet-50** hoặc **ResNet-101** với **FPN** (Feature Pyramid Network).
- Input: ảnh RGB `[B, 3, H, W]`.
- Output: feature maps `[B, C, H', W']` (ví dụ C=2048 channels, H'=H/16, W'=W/16).
- **ResNet-50**: mạng tích chập 50 lớp sử dụng **residual connections** (skip connections) để tránh vanishing gradient, học được đặc trưng sâu.
- **FPN**: tạo kim tự tháp đặc trưng ở nhiều tỷ lệ (P2, P3, P4, P5, P6) để phát hiện objects ở nhiều kích thước.

**b) Region Proposal Network (RPN)**
- Đây là **trái tim của Faster R-CNN**, thay thế Selective Search (phương pháp cổ điển chậm).
- **Mục đích**: quét feature map và đề xuất ~1000-2000 vùng có khả năng chứa đối tượng.

**Cơ chế hoạt động RPN**:
1. **Anchor Boxes**: tại mỗi vị trí trên feature map, đặt k anchor boxes (thường k=9) với 3 tỷ lệ (1:1, 1:2, 2:1) và 3 kích thước (128², 256², 512² pixels).
2. **Sliding Window**: một cửa sổ trượt 3×3 conv quét toàn bộ feature map.
3. **Dự đoán 2 nhánh**:
   - **Objectness score**: 2 giá trị (object / non-object) cho mỗi anchor → dùng softmax.
   - **Box regression**: 4 giá trị (Δx, Δy, Δw, Δh) để điều chỉnh anchor thành proposal chính xác hơn.
4. **Lọc proposals**: 
   - Loại bỏ boxes ngoài biên ảnh.
   - Áp dụng NMS với IoU threshold (0.7) để giảm overlap.
   - Giữ top-N proposals có objectness cao nhất (N~2000 khi train, N~1000 khi test).

**Công thức Box Regression:**

$$
\begin{align}
x_{\text{pred}} &= x_{\text{anchor}} + \Delta x \times w_{\text{anchor}} \\
y_{\text{pred}} &= y_{\text{anchor}} + \Delta y \times h_{\text{anchor}} \\
w_{\text{pred}} &= w_{\text{anchor}} \times e^{\Delta w} \\
h_{\text{pred}} &= h_{\text{anchor}} \times e^{\Delta h}
\end{align}
$$

**c) RoI Pooling / RoI Align**
- **Vấn đề**: Proposals có kích thước khác nhau, nhưng FC layer cần input cố định.
- **Giải pháp**: 
  - **RoI Pooling** (cũ): chia mỗi proposal thành lưới cố định 7×7, max pooling mỗi ô.
  - **RoI Align** (mới, chính xác hơn): dùng bilinear interpolation để tránh quantization errors.
- Output: feature vector cố định cho mỗi proposal (ví dụ 7×7×C).

**d) Fast R-CNN Head (Phân loại & Refine Box)**
- Mỗi RoI feature đi qua 2 nhánh:

**Nhánh 1: Classification**
- Fully Connected layers + Softmax.
- Output: xác suất cho (num_classes + 1) lớp (bao gồm background class).
- Loss: Cross Entropy Loss.

**Nhánh 2: Bounding Box Regression**
- FC layers dự đoán 4 giá trị refine (Δx, Δy, Δw, Δh) để điều chỉnh proposal thành final box.
- Loss: Smooth L1 Loss.

#### 2.3. Luồng xử lý end-to-end

**Training Phase**:
1. Ảnh → Backbone → Feature maps.
2. RPN nhận feature maps:
   - Tạo anchors tại mỗi vị trí.
   - Dự đoán objectness & box deltas.
   - Tính RPN loss (cls loss + box regression loss).
3. Proposals được lọc và match với Ground Truth:
   - IoU > 0.5 → positive samples.
   - IoU < 0.3 → negative samples.
4. RoI Align lấy features cho mỗi proposal.
5. Fast R-CNN Head:
   - Dự đoán class và refine box.
   - Tính Fast R-CNN loss (cls loss + box regression loss).
6. **Multi-task Loss:**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RPN\_cls}} + \lambda_1 \cdot \mathcal{L}_{\text{RPN\_box}} + \mathcal{L}_{\text{RCNN\_cls}} + \lambda_2 \cdot \mathcal{L}_{\text{RCNN\_box}}
$$

**Inference Phase**:
1. Ảnh → Backbone → Feature maps.
2. RPN tạo proposals (~1000).
3. RoI Align + Fast R-CNN Head dự đoán class & box cho mỗi proposal.
4. Áp dụng NMS (IoU threshold ~0.5) để loại bỏ overlap.
5. Output: final detections với class labels và boxes.

#### 2.4. Chi tiết các Loss Functions

**a) RPN Classification Loss**
- Binary Cross Entropy giữa objectness dự đoán và label (object/non-object).
- Chỉ tính cho các anchors được assign (matched với GT hoặc background).

**b) RPN Box Regression Loss**
- Smooth L1 Loss giữa predicted box deltas và target deltas.
- Chỉ tính cho positive anchors (có GT object).

**c) Fast R-CNN Classification Loss**
- Cross Entropy cho (num_classes + 1).
- Background class giúp model học phân biệt object/non-object rõ ràng.

**d) Fast R-CNN Box Regression Loss**
- Smooth L1 Loss cho 4 giá trị (Δx, Δy, Δw, Δh).
- Chỉ tính cho proposals có class ≠ background.

**Smooth L1 Loss Formula:**

$$
\text{SmoothL1}(x) = \begin{cases}
0.5 \times x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}
$$

(ít nhạy cảm với outliers hơn L2 loss)

#### 2.5. Ưu điểm và hạn chế

**Ưu điểm**:
- **Độ chính xác cao**: 2-stage design cho phép refine proposals kỹ lưỡng.
- **Phát hiện tốt object nhỏ và overlap**: RPN tạo nhiều proposals đa dạng.
- **Backbone mạnh mẽ**: ResNet-50-FPN học được đặc trưng phong phú.
- **Ổn định**: ít bị miss detection so với 1-stage detectors.

**Hạn chế**:
- **Chậm**: ~100-200ms/ảnh (so với YOLO 2-5ms). Không phù hợp realtime.
- **Phức tạp**: nhiều thành phần cần tune (anchor sizes, IoU thresholds, NMS thresholds).
- **Tốn bộ nhớ**: RPN + RoI Align + Head cần nhiều GPU memory.

---

### 3. So sánh YOLOv8 vs Faster R-CNN

| Tiêu chí | YOLOv8 | Faster R-CNN |
|----------|--------|--------------|
| **Loại detector** | 1-stage (single-shot) | 2-stage |
| **Tốc độ** | Rất nhanh (~200 FPS) | Chậm (~5-10 FPS) |
| **Độ chính xác** | Cao (mAP ~0.6-0.8) | Rất cao (mAP ~0.7-0.9) |
| **Anchor** | Anchor-free | Anchor-based (RPN) |
| **Kiến trúc** | CSPDarknet + PAN/FPN + Decoupled Head | ResNet + FPN + RPN + RoI Align + FC Head |
| **Phù hợp** | Realtime, edge devices, video streaming | Nghiên cứu, yêu cầu accuracy cao, offline processing |
| **Khó khăn** | Object nhỏ, sát nhau | Chậm, tốn tài nguyên |
| **Triển khai** | Dễ (Ultralytics API, export ONNX/TensorRT) | Khó hơn (cần optimize kỹ) |

**Kết luận**: 
- Dùng **YOLOv8** khi cần **tốc độ** (camera realtime, robot, drones).
- Dùng **Faster R-CNN** khi cần **độ chính xác tuyệt đối** (y tế, an ninh, phân tích ảnh chất lượng cao).

---

### 4. Các khái niệm nâng cao

#### 4.1. IoU (Intersection over Union)
- Độ đo overlap giữa 2 boxes A và B:

$$
\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}
$$

- Giá trị từ 0 (không overlap) đến 1 (overlap hoàn toàn).
- Dùng để:
  - Matching predictions với ground truth (IoU > 0.5 → match).
  - NMS (loại box có IoU > threshold với box score cao hơn).
  - Loss function (CIoU, DIoU, GIoU).

#### 4.2. NMS (Non-Maximum Suppression)
**Thuật toán**:
1. Sắp xếp tất cả boxes theo score giảm dần.
2. Chọn box có score cao nhất → thêm vào kết quả.
3. Loại bỏ tất cả boxes có IoU > threshold với box đã chọn.
4. Lặp lại bước 2-3 cho đến khi hết boxes.

**Ý nghĩa**: Tránh detect cùng 1 object nhiều lần.

#### 4.3. mAP (mean Average Precision)
- **Average Precision (AP)**: diện tích dưới Precision-Recall curve cho 1 class.
- **mAP**: trung bình AP của tất cả classes.
- **mAP@50**: tính AP với IoU threshold = 0.5.
- **mAP@50-95**: trung bình mAP với IoU từ 0.5 → 0.95 (bước 0.05) → đánh giá toàn diện hơn.

**Ý nghĩa**: mAP cao → model phát hiện chính xác nhiều objects ở nhiều IoU thresholds.

#### 4.4. Anchor Boxes
- **Pre-defined boxes** với tỷ lệ và kích thước cố định.
- Đặt tại mỗi vị trí feature map làm "template" để detect objects.
- Model học **offsets** (Δx, Δy, Δw, Δh) để điều chỉnh anchor thành bounding box cuối.
- **YOLOv8 bỏ anchors** → dự đoán trực tiếp center/size → đơn giản hơn, linh hoạt hơn.

#### 4.5. Feature Pyramid Network (FPN)
- Kiến trúc tạo kim tự tháp đặc trưng từ backbone.
- **Top-down pathway**: upsample feature maps từ tầng sâu (low resolution, semantic rich) và merge với tầng nông (high resolution, detail rich).
- **Mục đích**: kết hợp thông tin semantic và spatial → phát hiện tốt objects ở nhiều scales.

---

## KẾT LUẬN VỀ 2 THUẬT TOÁN

**YOLOv8** là lựa chọn tốt cho project này nếu:
- Cần triển khai trên thiết bị edge (Raspberry Pi, Jetson Nano).
- Xử lý video realtime (camera nông nghiệp, drone giám sát).
- Ưu tiên tốc độ huấn luyện và inference.

**Faster R-CNN** phù hợp nếu:
- Yêu cầu độ chính xác tuyệt đối (ứng dụng nghiên cứu, chẩn đoán quan trọng).
- Xử lý ảnh offline (không cần realtime).
- Có đủ tài nguyên GPU để huấn luyện lâu hơn.

Trong notebook `tomato_leaf.ipynb`, cả 2 models đều được huấn luyện và so sánh trên cùng dataset tomato disease detection, giúp bạn hiểu rõ trade-off giữa **speed vs accuracy**.

---

## CÂU HỎI THƯỜNG GẶP VỀ KIẾN TRÚC MODEL

### Q1: YOLOv8 và Faster R-CNN có bao nhiêu lớp? Lớp nào là lớp chính?

#### **YOLOv8s Architecture**
```
Tổng số layers: ~168 layers (modules)
```

**3 nhóm lớp chính:**

**1. Backbone (CSPDarknet) - ~80 layers**
- **Chức năng**: Trích xuất đặc trưng từ ảnh đầu vào
- **Thành phần**:
  - Conv layers: Các lớp tích chập cơ bản (3×3, 1×1 kernels)
  - **C2f blocks** (CSP Bottleneck with 2 convolutions): Lớp chính xuất hiện nhiều nhất
  - SPPF (Spatial Pyramid Pooling - Fast): Kết hợp features ở nhiều scale
- **Lớp chính**: **C2f (CSPLayer with 2 Convolutions Fast)**
  - Đây là building block cốt lõi của YOLOv8
  - Kết hợp CSP (Cross Stage Partial) để giảm computation
  - Xuất hiện trong cả Backbone và Neck
  - Cải thiện gradient flow và giảm parameters

**2. Neck (PAN/FPN) - ~40 layers**
- **Chức năng**: Kết hợp đặc trưng đa tỷ lệ (multi-scale feature fusion)
- **Thành phần**:
  - Upsample layers (top-down pathway): Tăng resolution
  - Concat layers (lateral connections): Kết nối skip connections
  - C2f blocks: Xử lý thông tin sau khi merge
  - Downsample layers (bottom-up pathway): Truyền thông tin từ low-level lên
- **Output**: 3 feature maps (P3, P4, P5) cho small/medium/large objects

**3. Head (Detection Head) - ~48 layers**
- **Chức năng**: Dự đoán bounding boxes và classes
- **Thành phần**:
  - 3 detection layers tương ứng 3 scales (P3: 80×80, P4: 40×40, P5: 20×20)
  - **Decoupled Head** (tách rời):
    - Classification branch: Conv layers dự đoán class probabilities
    - Box regression branch: Conv layers dự đoán (x, y, w, h)
  - **Anchor-free**: Không dùng anchor boxes, dự đoán trực tiếp

**Tại sao C2f là lớp chính?**
- Xuất hiện **nhiều nhất** trong toàn bộ kiến trúc (cả Backbone lẫn Neck)
- Chịu trách nhiệm **học đặc trưng** chính từ ảnh
- Kết hợp **CSP architecture** giúp:
  - Giảm 50% computation so với C3 (YOLOv5)
  - Tăng gradient flow (tránh vanishing gradient)
  - Duy trì accuracy cao

---

#### **Faster R-CNN ResNet-50-FPN Architecture**
```
Tổng số layers: ~175 layers (modules)
```

**4 nhóm lớp chính:**

**1. Backbone (ResNet-50) - ~50 layers convolution**
- **Chức năng**: Trích xuất đặc trưng sâu từ ảnh
- **Cấu trúc chi tiết**:
  - **Layer 1**: Conv1 (7×7 conv, stride=2) + BatchNorm + ReLU + MaxPool
  - **Layer 2-5**: 4 nhóm Residual Blocks:
    - Layer 2: 3 Bottleneck blocks → output (256 channels, H/4, W/4)
    - Layer 3: 4 Bottleneck blocks → output (512 channels, H/8, W/8)
    - Layer 4: 6 Bottleneck blocks → output (1024 channels, H/16, W/16)
    - Layer 5: 3 Bottleneck blocks → output (2048 channels, H/32, W/32)

- **Lớp chính**: **Residual Block (Bottleneck)**
  ```
  Input (x)
    ↓
  1×1 Conv (giảm channels) → BatchNorm → ReLU
    ↓
  3×3 Conv (extract features) → BatchNorm → ReLU
    ↓
  1×1 Conv (tăng channels) → BatchNorm
    ↓
  + Skip Connection (x) ← Đây là điểm đặc biệt!
    ↓
  ReLU → Output
  ```
  - **Tại sao là lớp chính?**
    - Giải quyết vấn đề **vanishing gradient** (mạng sâu 50+ layers vẫn train được)
    - Skip connection cho phép gradient flow trực tiếp
    - Học được đặc trưng **identity mapping** + residual features

**2. FPN (Feature Pyramid Network) - ~20 layers**
- **Chức năng**: Tạo kim tự tháp đặc trưng đa tỷ lệ
- **Thành phần**:
  - **Top-down pathway**: 
    - 1×1 conv giảm channels (2048→256, 1024→256, 512→256)
    - Upsample (×2) để tăng resolution
  - **Lateral connections**: 
    - 1×1 conv từ ResNet layers
    - Element-wise addition với top-down features
  - **Bottom-up pathway**:
    - 3×3 conv để refine merged features
  - **Output**: P2, P3, P4, P5, P6 (5 levels)

**3. RPN (Region Proposal Network) - ~10 layers**
- **Chức năng**: Tạo ~1000-2000 region proposals
- **Thành phần**:
  - **Sliding window**: 3×3 conv trượt trên feature map
    ```
    Input: (B, 256, H, W)
    3×3 Conv → (B, 512, H, W)
    ```
  - **2 nhánh song song**:
    - **Objectness branch**: 1×1 conv → (B, 2×num_anchors, H, W)
      - 2 classes: object / background
      - 9 anchors/cell (3 scales × 3 aspect ratios)
    - **Box regression branch**: 1×1 conv → (B, 4×num_anchors, H, W)
      - 4 coords: (Δx, Δy, Δw, Δh)

**4. RoI Head (Detection Head) - ~15 layers**
- **Chức năng**: Phân loại và refine proposals
- **Thành phần**:
  - **RoI Align**: Differentiable pooling (7×7 output)
    - Dùng bilinear interpolation (không quantization)
    - Input: proposals + feature maps
    - Output: (num_proposals, 256, 7, 7)
  - **2 FC layers**: Mỗi layer 1024 units
    ```
    (256×7×7) → Flatten → FC(1024) → ReLU → FC(1024) → ReLU
    ```
  - **2 nhánh song song**:
    - **Classification head**: FC(num_classes) → Softmax
    - **Box regression head**: FC(4×num_classes) → Box deltas

**Lớp chính: Residual Block (Bottleneck)**
- Chiếm **phần lớn computation** trong toàn bộ mạng
- Có trong 4 layers (Layer 2-5) với tổng 3+4+6+3 = **16 blocks**
- Mỗi block có 3 conv layers (1×1 → 3×3 → 1×1)
- **Tại sao quan trọng?**
  - Cho phép train mạng rất sâu (50-152 layers)
  - Skip connection = "highway" cho gradient
  - Learn cả identity mapping và residual features

---

### Q2: Trong code sử dụng bao nhiêu lớp? Tại sao?

#### **YOLOv8s trong code**
```python
model_yolo = YOLO('yolov8s.pt')  # Pre-trained weights
```

**Số lớp sử dụng: 168 layers**

**Tại sao chọn YOLOv8s (small)?**

| Lý do | Giải thích |
|-------|------------|
| **Cân bằng speed-accuracy** | mAP ~44.9% trên COCO, 200+ FPS trên V100 GPU |
| **Phù hợp Colab GPU** | Model size ~50MB, train được với RAM/VRAM hạn chế (T4 16GB) |
| **Pretrained trên COCO** | 80 classes đã học features cơ bản (edges, textures, shapes) |
| **Transfer learning hiệu quả** | Chỉ cần 15 epochs để fine-tune cho tomato diseases |
| **Không cần tune anchors** | Anchor-free → ít hyperparameters, dễ train |

**So sánh các versions YOLOv8:**
| Model | Params | mAP@50-95 | Speed (ms) | Khi nào dùng? |
|-------|--------|-----------|------------|---------------|
| YOLOv8n | 3.2M | 37.3% | 1.5ms | Edge devices (Raspberry Pi), real-time |
| **YOLOv8s** | 11.2M | 44.9% | 2.5ms | **Balance tốt nhất (code dùng)** |
| YOLOv8m | 25.9M | 50.2% | 4.5ms | Accuracy cao hơn, có GPU tốt |
| YOLOv8l | 43.7M | 52.9% | 6.5ms | Server-side, offline processing |
| YOLOv8x | 68.2M | 53.9% | 10ms | Yêu cầu accuracy tối đa |

**Code training:**
```python
results_yolo = model_yolo.train(
    data=yaml_path,
    epochs=15,        # Ít epoch vì dùng pretrained weights
    imgsz=256,        # Ảnh nhỏ (256×256) để train nhanh, tiết kiệm memory
    batch=16          # Batch size vừa phải cho GPU Colab T4
)
```

**Tại sao chỉ 15 epochs?**
- Pretrained model đã học 80% knowledge từ COCO
- Chỉ cần fine-tune để adapt với tomato disease patterns
- 15 epochs đủ để loss convergence (xem training curves)

---

#### **Faster R-CNN trong code**
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
```

**Số lớp sử dụng: ~175 layers**

**Tại sao chọn ResNet-50-FPN?**

| Lý do | Giải thích |
|-------|------------|
| **ResNet-50 là backbone chuẩn** | Đủ sâu (50 layers) mà không quá nặng như ResNet-101 |
| **FPN tích hợp sẵn** | Detect objects ở nhiều scales (bệnh nhỏ lẫn lớn trên lá) |
| **Pretrained trên COCO** | 80 classes object detection → đã biết detect "patterns" chung |
| **Fine-tune dễ dàng** | Chỉ thay classifier head, giữ nguyên backbone |

**Code thay đổi head:**
```python
def get_rcnn_model(num_classes):
    # Load pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Lấy input features của classifier hiện tại
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Thay chỉ phần classifier head cho 11 classes (10 diseases + 1 background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

num_classes = 11  # 10 bệnh + 1 background (class 0)
```

**Tại sao giữ nguyên backbone?**
- ResNet-50 đã học **low-level features** (edges, textures, colors)
- Features này **universal** → áp dụng được cho mọi domain
- Chỉ cần train lại **classifier head** để học mapping features → disease classes

**So sánh training configs:**
| Aspect | YOLOv8s | Faster R-CNN |
|--------|---------|--------------|
| **Layers** | 168 | 175 |
| **Pretrained** | COCO (80 classes) | COCO (80 classes) |
| **Modified parts** | Auto-adjusted head | Manual replace classifier |
| **Epochs** | 15 | 10 |
| **Batch size** | 16 | 8 |
| **Image size** | 256×256 | 256×256 |
| **Optimizer** | AdamW (auto) | SGD (manual) |
| **Training time** | ~30 mins (Colab T4) | ~45 mins (Colab T4) |

---

### Q3: Tại sao YOLOv8 chọn v8 mà không dùng version khác?

#### **Evolution của YOLO (2015-2024)**

```
YOLOv1 (2015) → YOLOv2 (2016) → YOLOv3 (2018) → YOLOv4 (2020)
  ↓                                                    ↓
YOLOv5 (2020) ← Ultralytics                    YOLOv7 (2022)
  ↓
YOLOv8 (2023) ← Code sử dụng
  ↓
YOLOv9 (2024) → YOLOv10 (2024) → YOLOv11 (2024)
```

#### **5 lý do chọn YOLOv8:**

**1. Anchor-Free Architecture (Quan trọng nhất!)**
```
YOLOv5/v7 (Anchor-based):
- Cần define trước 9 anchor boxes (3 scales × 3 ratios)
- Hyperparameters phức tạp: anchor_t, anchor_scale
- Phải tune anchors cho từng dataset
- Prediction: (objectness, Δx, Δy, Δw, Δh, class)

YOLOv8 (Anchor-free):
- Không cần anchor boxes
- Dự đoán trực tiếp (x, y, w, h, class)
- Đơn giản hơn, ít hyperparameters
- Flexible cho mọi object shapes
```

**Code comparison:**
```python
# YOLOv5 cần config anchors trong yaml:
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119] # P4/16
  - [116,90, 156,198, 373,326] # P5/32

# YOLOv8 không cần gì cả!
# Chỉ cần:
model = YOLO('yolov8s.pt')
results = model.train(data='tomato.yaml', epochs=15)
```

**2. C2f Block thay vì C3 (YOLOv5)**

| Aspect | YOLOv5 C3 | YOLOv8 C2f |
|--------|-----------|------------|
| **Conv layers** | 3 convolutions | 2 convolutions |
| **Computation** | Chậm hơn ~20% | **Nhanh hơn ~20%** |
| **Gradient flow** | Tốt | **Tốt hơn** |
| **Parameters** | Nhiều hơn | Ít hơn |
| **Accuracy** | High | High (tương đương) |

**Architecture comparison:**
```
C3 (YOLOv5):
Input → Conv1 → Conv2 → Conv3 → Concat → Conv4 → Output

C2f (YOLOv8):
Input → Conv1 → Conv2 → Concat → Conv3 → Output
        ↓        ↓
        Split → Multiple Bottlenecks (parallel)
```

**3. Improved Loss Function**

```
YOLOv5 Loss:
- Box Loss: CIoU (Complete IoU)
- Cls Loss: BCE (Binary Cross Entropy)
- Obj Loss: BCE

YOLOv8 Loss:
- Box Loss: CIoU + DFL (Distribution Focal Loss)
  ↑ DFL giúp dự đoán box regression chính xác hơn
- Cls Loss: BCE with improved weighting
- Obj Loss: Integrated into classification (không tách riêng)
```

**DFL (Distribution Focal Loss) - Điểm mới:**
- Thay vì predict 4 giá trị (x, y, w, h) trực tiếp
- Predict **distribution** của mỗi coordinate
- Model learn uncertainty → confident hơn khi predict boxes
- **Kết quả**: Localization accuracy tăng ~2-3% mAP

**4. Decoupled Head (Tách rời Classification và Box)**

```
YOLOv5 Coupled Head:
Feature Map → Shared Conv Layers → Split
                                      ↓
                                  [Box | Cls | Obj]
Problem: Box regression và classification conflict khi optimize

YOLOv8 Decoupled Head:
Feature Map → Box Branch → Conv layers → Box predictions
            ↓
            → Cls Branch → Conv layers → Class predictions

Benefit: 
- Box và Cls học độc lập
- Convergence nhanh hơn
- Accuracy tăng ~1-2% mAP
```

**5. API đơn giản hơn (Developer Experience)**

```python
# YOLOv5 (phức tạp):
# 1. Clone repo
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# 2. Prepare dataset in specific structure
# 3. Run training
python train.py --data tomato.yaml --weights yolov5s.pt --epochs 15 --imgsz 256

# 4. Inference
python detect.py --weights runs/train/exp/weights/best.pt --source test.jpg

# ----------------------------------------------------------------

# YOLOv8 (đơn giản):
pip install ultralytics

from ultralytics import YOLO

# Training
model = YOLO('yolov8s.pt')
results = model.train(data='tomato.yaml', epochs=15)

# Inference
results = model.predict('test.jpg')
results[0].plot()
```

#### **Tại sao KHÔNG dùng YOLOv9/v10/v11?**

**YOLOv9 (2024)**
- **Ưu điểm**: 
  - PGI (Programmable Gradient Information) → tăng accuracy ~5%
  - GELAN (Generalized ELAN) → efficient architecture
  - mAP@50-95: ~52% (cao hơn YOLOv8 ~7%)
- **Nhược điểm**:
  - ❌ **Nặng hơn nhiều** (50M+ parameters vs 11M của YOLOv8s)
  - ❌ **Chậm hơn** (~80 FPS vs 200+ FPS)
  - ❌ **Yêu cầu GPU mạnh** (V100/A100, không phù hợp Colab T4)
  - ❌ **Chưa stable** (ít tutorials, community nhỏ)

**YOLOv10 (2024)**
- **Ưu điểm**:
  - NMS-free (không cần Non-Maximum Suppression) → tăng tốc inference
  - Dual label assignments → học tốt hơn
- **Nhược điểm**:
  - ❌ **Mới ra** (May 2024) → ít documentation
  - ❌ **Breaking changes** API → code cũ không tương thích
  - ❌ **Chưa proven** trong production

**YOLOv11 (2024)**
- **Ưu điểm**:
  - Latest version (Oct 2024)
  - C3k2 blocks → efficient hơn
- **Nhược điểm**:
  - ❌ **Quá mới** → chưa có benchmark đầy đủ
  - ❌ **API changes** → migration cost cao
  - ❌ **Risk** cho production use

#### **Benchmark so sánh (COCO dataset):**

| Version | Year | mAP@50-95 | FPS (V100) | Params | Maturity |
|---------|------|-----------|------------|--------|----------|
| YOLOv5s | 2020 | 37.4% | 140 | 7.2M | ⭐⭐⭐⭐⭐ Mature |
| YOLOv7 | 2022 | 51.2% | 120 | 37M | ⭐⭐⭐⭐ Stable |
| **YOLOv8s** | 2023 | **44.9%** | **200+** | **11.2M** | ⭐⭐⭐⭐⭐ **Stable** |
| YOLOv9 | 2024 | 52.0% | 80 | 50M+ | ⭐⭐⭐ New |
| YOLOv10 | 2024 | 53.0% | 100 | 30M | ⭐⭐ Very New |
| YOLOv11 | 2024 | 54.0% | 90 | 28M | ⭐ Just Released |

**Kết luận**: YOLOv8s là **sweet spot** cho project này:
- ✅ **Balance** tốt nhất: Speed (200 FPS) + Accuracy (mAP ~45%)
- ✅ **Stable**: 1+ year proven in production
- ✅ **Colab-friendly**: Train được với T4 GPU (16GB VRAM)
- ✅ **Rich ecosystem**: Tutorials, community support, export options (ONNX, TensorRT, CoreML)
- ✅ **Anchor-free**: Dễ train, ít hyperparameters

**Khi nào upgrade lên YOLOv9+?**
- Khi có GPU mạnh (A100, H100)
- Khi accuracy quan trọng hơn speed
- Khi YOLOv9+ đã mature (6+ months sau release)

---

### Q4: Faster R-CNN có dùng tích chập không? Dùng như thế nào?

**Trả lời: CÓ! Faster R-CNN sử dụng convolution RẤT NHIỀU.**

Toàn bộ kiến trúc Faster R-CNN xoay quanh **tích chập 2D (2D Convolution)** - đây là "xương sống" của mọi CNN.

#### **4 Nơi Sử Dụng Tích Chập:**

---

#### **A. ResNet-50 Backbone (Convolution Core)**

**Cấu trúc chi tiết với số lượng conv:**

```
Input Image: (3, 256, 256)
    ↓
┌─────────────────────────────────────────────┐
│ CONV1 (Entry Block)                         │
│ - 7×7 Conv, 64 filters, stride=2, padding=3 │ ← Conv thứ 1
│ - BatchNorm + ReLU                          │
│ - 3×3 MaxPool, stride=2                     │
│ Output: (64, 64, 64)                        │
└─────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ LAYER 1 (3 Bottleneck Blocks)                │
│                                               │
│ Block 1:                                      │
│   1×1 Conv (64→64)   ← Conv 2                │
│   3×3 Conv (64→64)   ← Conv 3                │
│   1×1 Conv (64→256)  ← Conv 4                │
│   Skip: 1×1 Conv (3→256) ← Conv 5 (downsample) │
│                                               │
│ Block 2, 3: Tương tự (mỗi block 3 conv)      │
│ Tổng: 3 blocks × 3 = 9 conv + 1 downsample   │
│       = 10 conv layers                        │
│ Output: (256, 64, 64)                         │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ LAYER 2 (4 Bottleneck Blocks)                │
│ Tổng: 4 × 3 = 12 conv + 1 downsample         │
│       = 13 conv layers                        │
│ Output: (512, 32, 32)                         │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ LAYER 3 (6 Bottleneck Blocks)                │
│ Tổng: 6 × 3 = 18 conv + 1 downsample         │
│       = 19 conv layers                        │
│ Output: (1024, 16, 16)                        │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ LAYER 4 (3 Bottleneck Blocks)                │
│ Tổng: 3 × 3 = 9 conv + 1 downsample          │
│       = 10 conv layers                        │
│ Output: (2048, 8, 8)                          │
└──────────────────────────────────────────────┘

TỔNG CONV TRONG RESNET-50 BACKBONE:
1 (Conv1) + 10 (Layer1) + 13 (Layer2) + 19 (Layer3) + 10 (Layer4)
= 53 convolutional layers
```

**Chi tiết 1 Residual Block (Bottleneck):**

```python
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Conv 1: 1×1 conv giảm channels (dimensionality reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        # Conv 2: 3×3 conv học features
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, 
                                kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        
        # Conv 3: 1×1 conv tăng channels (dimensionality expansion)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        
        # Main path (3 convolutions)
        out = F.relu(self.bn1(self.conv1(x)))      # 1×1 conv
        out = F.relu(self.bn2(self.conv2(out)))    # 3×3 conv
        out = self.bn3(self.conv3(out))            # 1×1 conv
        
        # Skip connection
        if self.downsample:
            identity = self.downsample(x)           # 1×1 conv (nếu cần)
        
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out
```

**Công thức toán học tích chập 2D:**

$$
\text{Output}[i, j] = \sum_{c} \sum_{m} \sum_{n} \text{Input}[c, i+m, j+n] \times \text{Kernel}[c, m, n] + \text{Bias}
$$

Trong đó:
- $c$: channel index (duyệt tất cả input channels)
- $m, n$: kernel spatial dimensions (ví dụ: 0..2 cho 3×3 kernel)
- $i, j$: output spatial position

**Ví dụ cụ thể với 3×3 Conv:**

```
Input: (256, 64, 64)  # 256 channels, 64×64 spatial
Kernel: (512, 256, 3, 3)  # 512 output channels, 256 input channels, 3×3 kernel
Bias: (512,)

Computation:
- Với mỗi output channel k (0..511):
  - Với mỗi position (i, j):
    - Lấy 3×3 patch từ tất cả 256 input channels
    - Nhân element-wise với kernel[k]
    - Sum tất cả: Σ(256 × 3 × 3) = 2304 multiplications
    - Add bias[k]
    
Total operations cho 1 conv layer:
512 (output channels) × 64×64 (spatial) × 256×3×3 (kernel ops)
= ~604 million multiplications
```

---

#### **B. FPN (Feature Pyramid Network) - Convolution Fusion**

**3 loại convolution trong FPN:**

**1. Top-down pathway (Giảm channels):**

```python
# Giảm channels từ 2048 → 256
self.fpn_conv_c5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
# Input: (2048, 8, 8)
# Output: (256, 8, 8)

self.fpn_conv_c4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
# Input: (1024, 16, 16)
# Output: (256, 16, 16)

self.fpn_conv_c3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
# Input: (512, 32, 32)
# Output: (256, 32, 32)
```

**Tại sao dùng 1×1 Conv?**
- **Dimensionality reduction**: Giảm channels mà không thay đổi spatial size
- **Computation efficiency**: 1×1 conv rẻ hơn 3×3 rất nhiều
- **Feature transformation**: Learn linear combinations của input channels

**2. Lateral connections (Kết hợp features):**

```python
# Upsample từ tầng sâu lên
upsampled = F.interpolate(fpn_c5, scale_factor=2, mode='nearest')
# (256, 8, 8) → (256, 16, 16)

# Element-wise addition với lateral connection
fpn_p4 = upsampled + fpn_conv_c4(c4)
# (256, 16, 16) + (256, 16, 16) = (256, 16, 16)
```

**Cơ chế hoạt động:**
```
C5 (Deep, Semantic)     C4 (Mid-level)
(2048, 8, 8)            (1024, 16, 16)
    ↓ 1×1 Conv              ↓ 1×1 Conv
(256, 8, 8)             (256, 16, 16)
    ↓ Upsample ×2
(256, 16, 16) --------→ + --------→ P4 (256, 16, 16)
                        Add
```

**3. Smooth layers (3×3 Conv để giảm aliasing):**

```python
# Sau khi merge, dùng 3×3 conv để "smooth" features
self.fpn_smooth_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.fpn_smooth_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.fpn_smooth_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

# Apply
p4_final = self.fpn_smooth_p4(fpn_p4)
# Input: (256, 16, 16)
# Output: (256, 16, 16)  # Same size, but refined features
```

**Tại sao cần smooth conv?**
- Upsample (nearest/bilinear) gây **aliasing** (răng cưa)
- 3×3 conv làm mịn boundaries
- Tăng receptive field

**FPN Complete Flow:**

```
ResNet Backbone Output:
C2: (256, 64, 64)
C3: (512, 32, 32)
C4: (1024, 16, 16)
C5: (2048, 8, 8)

                                    ┌─ P2 (256, 64, 64)
                                    │
Top-Down:                           ├─ P3 (256, 32, 32)
C5 → 1×1Conv → P5 (256, 8, 8)      │
     ↓ Upsample                    ├─ P4 (256, 16, 16)
C4 → 1×1Conv → + → 3×3Conv → P4    │
     ↓ Upsample                    ├─ P5 (256, 8, 8)
C3 → 1×1Conv → + → 3×3Conv → P3    │
     ↓ Upsample                    └─ P6 (256, 4, 4) [từ P5 với stride=2]
C2 → 1×1Conv → + → 3×3Conv → P2

Total Conv layers trong FPN:
- 1×1 Conv: 4 layers (C2, C3, C4, C5 → 256 channels)
- 3×3 Conv: 4 layers (smooth P2, P3, P4, P5)
= 8 conv layers
```

---

#### **C. RPN (Region Proposal Network) - Convolution Sliding Window**

**RPN là "trái tim" của Faster R-CNN, sử dụng 3 conv layers:**

**1. Sliding Window Convolution:**

```python
class RPNHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9):
        # Intermediate conv (sliding window)
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        # Input: (256, H, W)  # Từ FPN
        # Output: (512, H, W)  # Same spatial size
        
    def forward(self, x):
        # Slide 3×3 window across feature map
        x = F.relu(self.conv(x))
        return x
```

**Cơ chế Sliding Window:**
```
Feature Map: (256, 40, 40)  # Từ FPN P4
    ↓
3×3 Conv kernel slides:
┌─────┬─────┬─────┬─────┐
│ ◉ ◉ ◉│     │     │     │  ← Position (0,0)
│ ◉ ◉ ◉│     │     │     │
│ ◉ ◉ ◉│     │     │     │
├─────┼─────┼─────┼─────┤
│     │ ◉ ◉ ◉│     │     │  ← Position (0,1)
│     │ ◉ ◉ ◉│     │     │
│     │ ◉ ◉ ◉│     │     │
└─────┴─────┴─────┴─────┘

Tại mỗi position (i, j):
- Extract 3×3×256 patch
- Convolve với kernel (512, 256, 3, 3)
- Produce 512-dim feature vector
- Repeat cho tất cả 40×40 = 1600 positions
```

**2. Classification Branch (Objectness):**

```python
# Predict objectness scores (object vs background)
self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
# Input: (512, H, W)
# Output: (18, H, W)  # 9 anchors × 2 classes

# Reshape output:
# (B, 18, H, W) → (B, H, W, 9, 2) → (B, H×W×9, 2)
```

**Anchors:**
```
Mỗi position trên feature map có 9 anchors:
- 3 scales: 128², 256², 512² pixels
- 3 aspect ratios: 1:1, 1:2, 2:1

Ví dụ position (10, 10) trên feature map 40×40:
Anchor 1: (256×1, 256×1)   # Square 256
Anchor 2: (256×0.5, 256×2) # Tall 128×512
Anchor 3: (256×2, 256×0.5) # Wide 512×128
Anchor 4: (512×1, 512×1)   # Square 512
...
Anchor 9: (128×2, 128×0.5) # Wide 256×64

Total anchors: 40 × 40 × 9 = 14,400 anchors
```

**3. Box Regression Branch:**

```python
# Predict bounding box deltas
self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
# Input: (512, H, W)
# Output: (36, H, W)  # 9 anchors × 4 coords (Δx, Δy, Δw, Δh)
```

**Box Regression Formula:**
```python
def apply_box_deltas(anchors, deltas):
    # anchors: (N, 4) [x, y, w, h]
    # deltas: (N, 4) [Δx, Δy, Δw, Δh]
    
    # Center coordinates
    anchor_widths = anchors[:, 2]
    anchor_heights = anchors[:, 3]
    anchor_centers_x = anchors[:, 0]
    anchor_centers_y = anchors[:, 1]
    
    # Apply deltas
    pred_centers_x = anchor_centers_x + deltas[:, 0] * anchor_widths
    pred_centers_y = anchor_centers_y + deltas[:, 1] * anchor_heights
    pred_widths = anchor_widths * torch.exp(deltas[:, 2])
    pred_heights = anchor_heights * torch.exp(deltas[:, 3])
    
    # Convert to (x1, y1, x2, y2)
    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_centers_x - 0.5 * pred_widths   # x1
    pred_boxes[:, 1] = pred_centers_y - 0.5 * pred_heights  # y1
    pred_boxes[:, 2] = pred_centers_x + 0.5 * pred_widths   # x2
    pred_boxes[:, 3] = pred_centers_y + 0.5 * pred_heights  # y2
    
    return pred_boxes
```

**RPN Complete Architecture:**

```
FPN Feature Map: (256, H, W)
    ↓
┌──────────────────────────────────┐
│ 3×3 Conv (256 → 512)             │ ← Sliding window
│ + ReLU                           │
│ Output: (512, H, W)              │
└────────────┬─────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ↓             ↓
┌─────────┐   ┌─────────┐
│ 1×1 Conv│   │ 1×1 Conv│
│ (512→18)│   │ (512→36)│
│         │   │         │
│  Class  │   │   Box   │
│  (18,   │   │  (36,   │
│   H, W) │   │   H, W) │
└─────────┘   └─────────┘
     │             │
     └──────┬──────┘
            ↓
    Apply NMS + Filter
            ↓
    Top-N Proposals (~1000)
```

**Total Conv trong RPN: 3 layers**
- 1 sliding window conv (3×3)
- 1 classification conv (1×1)
- 1 box regression conv (1×1)

---

#### **D. RoI Head - FC Layers (KHÔNG PHẢI Convolution)**

**Quan trọng: RoI Head trong code SỬ DỤNG FC LAYERS, không phải conv!**

```python
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        # RoI Pooled features: (in_channels, 7, 7)
        # Flatten: 7×7×in_channels = 12544 dims (nếu in_channels=256)
        
        # FC Layer 1
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        # Input: (12544,)
        # Output: (1024,)
        
        # FC Layer 2
        self.fc2 = nn.Linear(1024, 1024)
        # Input: (1024,)
        # Output: (1024,)
        
        # Classification head
        self.cls_score = nn.Linear(1024, num_classes)
        # Input: (1024,)
        # Output: (11,)  # 10 diseases + 1 background
        
        # Box regression head
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        # Input: (1024,)
        # Output: (44,)  # 11 classes × 4 coords
    
    def forward(self, x):
        # x: (num_proposals, 256, 7, 7)
        x = x.flatten(start_dim=1)  # (num_proposals, 12544)
        
        x = F.relu(self.fc1(x))     # (num_proposals, 1024)
        x = F.relu(self.fc2(x))     # (num_proposals, 1024)
        
        cls_scores = self.cls_score(x)    # (num_proposals, 11)
        bbox_deltas = self.bbox_pred(x)   # (num_proposals, 44)
        
        return cls_scores, bbox_deltas
```

**Tại sao dùng FC thay vì Conv?**
- RoI features đã được **RoI Align** pool về fixed size (7×7)
- Cần **global context** để classify (không phải local patterns)
- FC layers learn **holistic representations**

**Một số variants dùng Conv:**
```python
# Alternative: All-Conv Head (Mask R-CNN style)
class ConvFCHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        # 4 conv layers
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Flatten + FC
        self.fc = nn.Linear(256 * 7 * 7, 1024)
        
        # Heads
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
```

**Nhưng trong code `torchvision.models.detection.fasterrcnn_resnet50_fpn`:**
- **Default sử dụng FC layers** (không phải conv)
- Lý do: Faster, simpler, proven effective

---

### **TỔNG KẾT: Convolution trong Faster R-CNN**

| Component | Conv Layers | Kernel Sizes | Chức năng |
|-----------|-------------|--------------|-----------|
| **ResNet-50 Backbone** | 53 layers | 1×1, 3×3, 7×7 | Trích xuất features cơ bản |
| **FPN** | 8 layers | 1×1 (4), 3×3 (4) | Multi-scale feature fusion |
| **RPN** | 3 layers | 3×3 (1), 1×1 (2) | Generate proposals |
| **RoI Head** | 0 layers | N/A | Sử dụng FC layers |

**Total Convolution Layers: 53 + 8 + 3 = 64 conv layers**

**Công thức Convolution 2D (Recap):**

$$
\text{Output}[b, c_{\text{out}}, h, w] = \sum_{c_{\text{in}}} \sum_{m} \sum_{n} \left( \text{Input}[b, c_{\text{in}}, h \times \text{stride} + m, w \times \text{stride} + n] \times \text{Weight}[c_{\text{out}}, c_{\text{in}}, m, n] \right) + \text{Bias}[c_{\text{out}}]
$$

Trong đó:
- $b$: batch index
- $c_{\text{out}}$: output channel index (0..num_filters-1)
- $c_{\text{in}}$: input channel index (0..in_channels-1)
- $m, n$: kernel spatial dimensions (0..kernel_size-1)
- $h, w$: output spatial position
- stride: convolution stride (thường 1 hoặc 2)

**Ví dụ cụ thể:**

```python
import torch
import torch.nn as nn

# Define a 3×3 conv layer
conv = nn.Conv2d(
    in_channels=256,      # Input depth
    out_channels=512,     # Number of filters
    kernel_size=3,        # 3×3 kernel
    stride=1,             # Slide 1 pixel at a time
    padding=1,            # Pad 1 pixel to maintain size
    bias=True             # Include bias term
)

# Input tensor
x = torch.randn(8, 256, 40, 40)  # (batch, channels, height, width)

# Forward pass
output = conv(x)  # (8, 512, 40, 40)

# Number of parameters:
# Weights: 512 × 256 × 3 × 3 = 1,179,648
# Bias: 512
# Total: 1,180,160 parameters
```

**Computation (FLOPs):**

$$
\text{FLOPs} = 2 \times K_h \times K_w \times C_{\text{in}} \times C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}
$$

Ví dụ trên:
$$
\begin{align}
\text{FLOPs} &= 2 \times 3 \times 3 \times 256 \times 512 \times 40 \times 40 \\
&= 2 \times 9 \times 256 \times 512 \times 1600 \\
&\approx 3.8 \text{ billion operations}
\end{align}
$$

**Điểm khác biệt YOLOv8 vs Faster R-CNN:**

| Aspect | YOLOv8 | Faster R-CNN |
|--------|--------|--------------|
| **Convolution usage** | 100% conv (end-to-end) | Conv backbone + RPN, FC head |
| **Detection head** | Conv layers | **FC layers** |
| **Speed** | Faster (no FC bottleneck) | Slower (FC + RoI Align) |
| **Parallelization** | Full GPU parallelization | Sequential (RPN → RoI → FC) |

**Kết luận:**
- ✅ Faster R-CNN **SỬ DỤNG CONVOLUTION RẤT NHIỀU** (64 conv layers)
- ✅ Convolution xuất hiện ở: **Backbone, FPN, RPN**
- ✅ RoI Head dùng **FC layers** (không phải conv) trong implementation chuẩn
- ✅ Conv là "xương sống" để extract features, FCs để classify/regress

**Lưu ý vận hành / debug**
- Nếu training bị lỗi do OOM (Out of Memory): giảm `batch` hoặc `imgsz` hoặc dùng GPU lớn hơn.
- Nếu `results.csv` không sinh: có thể quá trình train dừng sớm do lỗi; kiểm tra logs trong cell huấn luyện.
- Kiểm tra `CLASS_NAMES` khớp với nhãn; lỗi mismatch dẫn tới đánh giá sai hoặc IndexError.
- Khi chuyển nhãn YOLO cho Faster R-CNN, nhớ +1 cho label index vì PyTorch detection dùng 0 là background.

**Hướng dẫn nhanh để chạy (Colab)**
1. Mount Drive (đã có trong notebook).  
2. Chỉnh `ROOT_DIR` trỏ đến thư mục dataset trong Drive.  
3. Chạy tuần tự các cell: cài đặt -> chuẩn bị dữ liệu (`prepare_data()`), -> ghi yaml (`write_yolo_yaml()`), -> phân tích phân bố (`analyze_distribution`) -> huấn luyện YOLO -> đánh giá -> chuẩn bị dataset RCNN -> huấn luyện RCNN -> đánh giá và so sánh.

**File đã tạo**
- `d:\colab\README.md` (bản bạn đang đọc).

Nếu bạn muốn, tôi có thể:
- Dịch toàn bộ README sang tiếng Anh.
- Tạo file `requirements.txt` dựa trên các import trong notebook.
- Tách README thành từng phần nhỏ hơn (mỗi phần một file) hoặc thêm ví dụ chạy nhanh (script .py) để chạy local.

---
Tôi đã tạo file `d:\colab\README.md` chứa giải thích chi tiết. Muốn chỉnh ngôn ngữ hay bổ sung thêm phần nào không?