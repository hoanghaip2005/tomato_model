# README â€” Giáº£i thÃ­ch `tomato_leaf.ipynb`

## TÃ“M Táº®T Tá»”NG QUAN

### 1. Má»¤C ÄÃCH BÃ€I TOÃN

**Váº¥n Ä‘á» thá»±c tiá»…n**: Trong nÃ´ng nghiá»‡p hiá»‡n Ä‘áº¡i, viá»‡c phÃ¡t hiá»‡n sá»›m vÃ  chÃ­nh xÃ¡c cÃ¡c bá»‡nh trÃªn cÃ¢y trá»“ng (Ä‘áº·c biá»‡t lÃ  cÃ  chua) lÃ  cá»±c ká»³ quan trá»ng Ä‘á»ƒ:
- Giáº£m thiá»‡t háº¡i kinh táº¿ cho nÃ´ng dÃ¢n
- Giáº£m lÆ°á»£ng thuá»‘c trá»« sÃ¢u (báº£o vá»‡ mÃ´i trÆ°á»ng vÃ  sá»©c khá»e)
- TÄƒng nÄƒng suáº¥t vÃ  cháº¥t lÆ°á»£ng nÃ´ng sáº£n

**BÃ i toÃ¡n Machine Learning**: XÃ¢y dá»±ng há»‡ thá»‘ng **Object Detection** (phÃ¡t hiá»‡n Ä‘á»‘i tÆ°á»£ng) tá»± Ä‘á»™ng nháº­n diá»‡n vÃ  phÃ¢n loáº¡i cÃ¡c bá»‡nh trÃªn lÃ¡ cÃ  chua thÃ´ng qua áº£nh chá»¥p. Há»‡ thá»‘ng cáº§n:
- XÃ¡c Ä‘á»‹nh **vá»‹ trÃ­** (bounding box) vÃ¹ng bá»‡nh trÃªn lÃ¡
- PhÃ¢n loáº¡i **loáº¡i bá»‡nh** (10 lá»›p: 9 bá»‡nh + 1 lÃ¡ khá»e máº¡nh)
- ÄÃ¡nh giÃ¡ hiá»‡u nÄƒng cá»§a 2 thuáº­t toÃ¡n SOTA: **YOLOv8s** (real-time) vs **Faster R-CNN** (high accuracy)

### 2. Má»¤C ÄÃCH Cá»¦A CODE

Notebook `tomato_leaf.ipynb` thá»±c hiá»‡n **pipeline hoÃ n chá»‰nh** cho bÃ i toÃ¡n Computer Vision tá»« A-Z:

**Giai Ä‘oáº¡n 1: Chuáº©n bá»‹ dá»¯ liá»‡u (Data Preparation)**
- Tá»• chá»©c láº¡i dataset theo chuáº©n YOLO (images/ vÃ  labels/)
- Chia dataset: 70% train, 15% val, 15% test (stratified split)
- PhÃ¢n tÃ­ch phÃ¢n bá»‘ lá»›p (class imbalance analysis)
- Táº¡o file cáº¥u hÃ¬nh `tomato.yaml` cho YOLO

**Giai Ä‘oáº¡n 2: Huáº¥n luyá»‡n YOLOv8s (YOLO Training)**
- Load pretrained weights `yolov8s.pt` (transfer learning)
- Fine-tune trÃªn tomato dataset (15 epochs, 256px, batch 16)
- ÄÃ¡nh giÃ¡ trÃªn test set (mAP@50, Precision, Recall, F1)
- Visualize training curves (loss, mAP) vÃ  confusion matrix

**Giai Ä‘oáº¡n 3: Huáº¥n luyá»‡n Faster R-CNN (Baseline Comparison)**
- Táº¡o custom PyTorch Dataset cho detection task
- Load pretrained Faster R-CNN ResNet-50-FPN (COCO weights)
- Fine-tune classifier head cho 11 classes (10 diseases + background)
- Huáº¥n luyá»‡n 10 epochs vá»›i SGD optimizer

**Giai Ä‘oáº¡n 4: So sÃ¡nh vÃ  ÄÃ¡nh giÃ¡ (Evaluation & Comparison)**
- TÃ­nh toÃ¡n metrics chuáº©n: mAP@50, Precision, Recall, F1-Score
- Váº½ biá»ƒu Ä‘á»“ so sÃ¡nh trá»±c quan giá»¯a 2 models
- Visualize káº¿t quáº£ dá»± Ä‘oÃ¡n trÃªn áº£nh thá»±c táº¿
- PhÃ¢n tÃ­ch trade-off giá»¯a tá»‘c Ä‘á»™ vÃ  Ä‘á»™ chÃ­nh xÃ¡c

### 3. PHá»I Há»¢P GIá»®A CODE VÃ€ TEMPLATE BÃO CÃO PDF

File PDF `á»¨NG Dá»¤NG DEEP LEARNING TRONG PHÃT HIá»†N Bá»†NH TRÃŠN LÃ CÃ€ CHUA.pdf` lÃ  **template bÃ¡o cÃ¡o há»c thuáº­t** phá»¥c vá»¥ má»¥c Ä‘Ã­ch:
- TrÃ¬nh bÃ y lÃ½ thuyáº¿t ná»n táº£ng (Introduction, Literature Review)
- MÃ´ táº£ phÆ°Æ¡ng phÃ¡p luáº­n (Methodology)
- BÃ¡o cÃ¡o káº¿t quáº£ thá»±c nghiá»‡m (Results & Discussion)

**CÃ¡ch phá»‘i há»£p Code â†” PDF Template**:

| Pháº§n trong PDF | Ná»™i dung tá»« Code | CÃ¡ch láº¥y |
|----------------|------------------|----------|
| **1. Introduction** | MÃ´ táº£ bÃ i toÃ¡n, dataset (10 classes) | Tá»« `CLASS_NAMES` vÃ  phÃ¢n tÃ­ch `analyze_distribution()` |
| **2. Dataset** | Sá»‘ lÆ°á»£ng áº£nh train/val/test, phÃ¢n bá»‘ lá»›p, imbalance ratio | Output cá»§a `prepare_data()` vÃ  `analyze_distribution()` |
| **3. Methodology** | Kiáº¿n trÃºc YOLOv8 & Faster R-CNN, hyperparameters | Code blocks huáº¥n luyá»‡n (epochs, imgsz, batch, optimizer) |
| **4. Experiments** | Training curves (loss, mAP theo epoch) | `results.csv` tá»« YOLO vÃ  `rcnn_loss_history` |
| **5. Results** | Báº£ng so sÃ¡nh mAP@50, Precision, Recall, F1 cá»§a 2 models | Biáº¿n `yolo_map50`, `rcnn_map50`, `yolo_f1`, `rcnn_f1` |
| **6. Visualization** | Confusion matrix, áº£nh dá»± Ä‘oÃ¡n cÃ³ bounding boxes | `confusion_matrix.png` vÃ  output tá»« cell 10 |
| **7. Discussion** | PhÃ¢n tÃ­ch Æ°u/nhÆ°á»£c Ä‘iá»ƒm YOLOv8 vs Faster R-CNN | So sÃ¡nh tá»‘c Ä‘á»™ (FPS), accuracy, use cases |

**Workflow chuáº©n**:
1. **Cháº¡y code** â†’ Thu tháº­p táº¥t cáº£ káº¿t quáº£ (metrics, biá»ƒu Ä‘á»“, áº£nh)
2. **Chá»¥p/LÆ°u outputs** â†’ Copy vÃ o cÃ¡c section tÆ°Æ¡ng á»©ng trong PDF
3. **Viáº¿t phÃ¢n tÃ­ch** â†’ Giáº£i thÃ­ch Ã½ nghÄ©a cá»§a káº¿t quáº£ trong pháº§n Discussion
4. **Tá»•ng káº¿t** â†’ Conclusion + Future Work

**LÆ°u Ã½ quan trá»ng**:
- Code sinh ra **raw data** (sá»‘ liá»‡u, hÃ¬nh áº£nh)
- PDF cáº§n **diá»…n giáº£i** raw data thÃ nh insights cÃ³ Ã½ nghÄ©a
- BÃ¡o cÃ¡o tá»‘t = Code cháº¡y Ä‘Ãºng + PhÃ¢n tÃ­ch sÃ¢u sáº¯c

---

## HÆ¯á»šNG DáºªN Sá»¬ Dá»¤NG

Má»¥c tiÃªu: giáº£i thÃ­ch chi tiáº¿t tá»«ng Ã´ (cell) cá»§a notebook `tomato_leaf.ipynb` Ä‘á»ƒ báº¡n hiá»ƒu luá»“ng lÃ m viá»‡c, tá»«ng hÃ m, vÃ  pháº§n thuáº­t toÃ¡n mÃ  khÃ´ng cáº§n cháº¡y láº¡i mÃ£. Ná»™i dung viáº¿t báº±ng tiáº¿ng Viá»‡t, kÃ¨m chÃº giáº£i vá» cÃ¡c khÃ¡i niá»‡m nhÆ° YOLO label format, IoU, mAP, Precision/Recall/F1, Faster R-CNN, vÃ  cÃ¡c lÆ°u Ã½ khi cháº¡y trÃªn Colab hoáº·c local.

**File**: `d:\colab\tomato_leaf.ipynb`

**YÃªu cáº§u mÃ´i trÆ°á»ng**:
- Cháº¡y tá»‘t trÃªn Google Colab (Ä‘Ã£ mount Google Drive trong notebook).
- ThÆ° viá»‡n chÃ­nh: `ultralytics` (YOLOv8), `torch`, `torchvision`, `torchmetrics`, `scikit-learn`, `opencv-python`, `matplotlib`, `pyyaml`.
- Náº¿u cháº¡y local Windows, cÃ i Python 3.8+ vÃ  GPU CUDA (náº¿u muá»‘n tÄƒng tá»‘c). VÃ­ dá»¥ lá»‡nh cÃ i Ä‘áº·t Colab (Ä‘Ã£ cÃ³ trong notebook):

```powershell
!pip install ultralytics torchmetrics
!pip install -U scikit-learn
```

**Cáº¥u trÃºc tá»•ng quan cá»§a notebook**
- Chuáº©n bá»‹: mount Drive, import thÆ° viá»‡n, set `device`.
- Chuáº©n hÃ³a dataset (copy images & labels sang cáº¥u trÃºc YOLO), chia train/val/test.
- Viáº¿t file cáº¥u hÃ¬nh YOLO (`tomato.yaml`).
- PhÃ¢n tÃ­ch phÃ¢n bá»‘ lá»›p dá»±a trÃªn file nhÃ£n YOLO, váº½ biá»ƒu Ä‘á»“.
- Huáº¥n luyá»‡n YOLOv8s (sá»­ dá»¥ng `ultralytics.YOLO`).
- Táº¡o dataset cho Faster R-CNN (PyTorch `Dataset`), huáº¥n luyá»‡n Faster R-CNN.
- ÄÃ¡nh giÃ¡ vÃ  so sÃ¡nh hai mÃ´ hÃ¬nh (mAP@50, Precision, Recall, F1), trá»±c quan hÃ³a káº¿t quáº£.

**Giáº£i thÃ­ch tá»«ng Ã´ / hÃ m / khá»‘i mÃ£**

**1) CÃ i Ä‘áº·t & import (Ã´ Ä‘áº§u)**
- Má»¥c Ä‘Ã­ch: cÃ i gÃ³i cáº§n thiáº¿t vÃ  import cÃ¡c thÆ° viá»‡n.
- LÆ°u Ã½: `!pip` chá»‰ cháº¡y trong mÃ´i trÆ°á»ng notebook (Colab). TrÃªn Windows PowerShell, dÃ¹ng `pip install` bÃ¬nh thÆ°á»ng.

**2) Mount Google Drive & chá»n device**
- Code mount Drive Ä‘á»ƒ truy cáº­p dataset lÆ°u trÃªn Drive.
- `device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` chá»n GPU náº¿u cÃ³.

**3) Biáº¿n cáº¥u hÃ¬nh vÃ  danh sÃ¡ch lá»›p**
- `ROOT_DIR` lÃ  nÆ¡i chá»©a `images/` vÃ  `labels/` (nhÃ£n YOLO *.txt).
- `WORK_DIR` lÃ  nÆ¡i sao chÃ©p vÃ  lÆ°u cáº¥u trÃºc má»›i `images/train, images/val, images/test` vÃ  `labels/...`.
- `CLASS_NAMES`: danh sÃ¡ch tÃªn lá»›p (pháº£i khá»›p vá»›i chá»‰ sá»‘ trong nhÃ£n YOLO).

**4) HÃ m `prepare_data()`**
- Má»¥c tiÃªu: táº¡o cáº·p (image, label) há»£p lá»‡, chia ngáº«u nhiÃªn thÃ nh train/val/test theo tá»· lá»‡ 70/15/15 vÃ  sao chÃ©p vÃ o `WORK_DIR`.
- CÆ¡ cháº¿:
  - Duyá»‡t táº¥t cáº£ `.jpg` trong `RAW_IMAGES`, kiá»ƒm tra file label tÆ°Æ¡ng á»©ng `.txt` trong `RAW_LABELS`.
  - DÃ¹ng `train_test_split` Ä‘á»ƒ chia: trÆ°á»›c chia test 15%, cÃ²n láº¡i chia train/val ~15% cho val.
  - Náº¿u `WORK_DIR` Ä‘Ã£ tá»“n táº¡i vÃ  sá»‘ lÆ°á»£ng file khá»›p, hÃ m sáº½ bá» qua sao chÃ©p Ä‘á»ƒ trÃ¡nh ghi Ä‘Ã¨ tá»‘n thá»i gian.
- LÆ°u Ã½: giá»¯ `random_state` cá»‘ Ä‘á»‹nh Ä‘á»ƒ tÃ¡i láº­p káº¿t quáº£.
- Output: thÆ° má»¥c `WORK_DIR` vá»›i cáº¥u trÃºc phÃ¹ há»£p YOLO.

**5) HÃ m `write_yolo_yaml()`**
- Má»¥c tiÃªu: táº¡o file config `tomato.yaml` Ä‘á»ƒ YOLOv8 biáº¿t Ä‘Æ°á»ng dáº«n dá»¯ liá»‡u, sá»‘ lá»›p vÃ  tÃªn lá»›p.
- Ná»™i dung YAML gá»“m `path`, `train`, `val`, `test`, `nc`, `names`.
- LÆ°u Ã½: `nc` pháº£i báº±ng sá»‘ lá»›p (á»Ÿ Ä‘Ã¢y 10), `names` tÆ°Æ¡ng á»©ng vá»›i chá»‰ sá»‘ nhÃ£n.

**6) HÃ m `plot_two_charts(class_counts, class_names, split_name, save_path=None)`**
- Má»¥c tiÃªu: váº½ 2 biá»ƒu Ä‘á»“ (bar Ä‘á»©ng vÃ  bar ngang) cho phÃ¢n bá»‘ nhÃ£n.
- Input:
  - `class_counts`: Counter mapping class_id -> count
  - `class_names`: danh sÃ¡ch tÃªn lá»›p
  - `split_name`: chuá»—i Ä‘á»ƒ hiá»ƒn thá»‹ tiÃªu Ä‘á» (Training/Validation/Test)
  - `save_path` (tÃ¹y chá»n)
- LÆ°u Ã½: mÃ£ tÃ­nh `pad` Ä‘á»ƒ Ä‘áº·t text nhÃ£n cÃ¡ch cá»™t.

**7) HÃ m `analyze_distribution(labels_dir, split_name)`**
- Má»¥c tiÃªu: Ä‘á»c táº¥t cáº£ file label `.txt` trong `labels_dir`, thá»‘ng kÃª:
  - `class_counts` tá»•ng sá»‘ annotation má»—i lá»›p,
  - `images_with_multiple_classes` sá»‘ áº£nh cÃ³ nhiá»u hÆ¡n 1 lá»›p,
  - `total_annotations` tá»•ng annotations.
- CÃ¡ch Ä‘á»c file YOLO label: má»—i dÃ²ng `class_id cx cy w h` vá»›i tá»a Ä‘á»™ normalized (0..1).
- Sau khi tÃ­nh toÃ¡n in ra báº£ng tÃ³m táº¯t vÃ  gá»i `plot_two_charts` Ä‘á»ƒ váº½ rá»“i lÆ°u áº£nh.
- LÆ°u Ã½: tá»‰ lá»‡ máº¥t cÃ¢n báº±ng Ä‘Æ°á»£c tÃ­nh báº±ng `max_c / min_c` trÃªn cÃ¡c lá»›p cÃ³ >0 annotation.

--- Thuáº­t toÃ¡n & Ä‘á»‹nh dáº¡ng nhÃ£n YOLO (giáº£i thÃ­ch chi tiáº¿t)
- Má»—i dÃ²ng file nhÃ£n YOLO: `class_id cx cy w h`:
  - `class_id`: sá»‘ nguyÃªn báº¯t Ä‘áº§u tá»« 0.
  - `(cx, cy)`: tÃ¢m há»™p giá»›i háº¡n (relative to image width/height, normalized 0..1).
  - `(w, h)`: chiá»u rá»™ng vÃ  chiá»u cao há»™p (normalized).
- Äá»ƒ chuyá»ƒn sang bbox dáº¡ng `[x1, y1, x2, y2]` theo kÃ­ch thÆ°á»›c áº£nh:
  - x1 = (cx - w/2) * W, y1 = (cy - h/2) * H, x2 = (cx + w/2) * W, y2 = (cy + h/2) * H.
- LÆ°u Ã½ quan trá»ng: nhÃ£n YOLO sá»­ dá»¥ng class_id báº¯t Ä‘áº§u tá»« 0. Khi dÃ¹ng Faster R-CNN, code Ä‘Ã£ cá»™ng thÃªm `+1` vÃ¬ PyTorch detection models mong nhÃ£n báº¯t Ä‘áº§u tá»« 1 (0 dÃ nh cho background).

**8) Khá»‘i huáº¥n luyá»‡n YOLOv8s**
- DÃ¹ng `ultralytics.YOLO` (vÃ­ dá»¥ `YOLO('yolov8s.pt')`) Ä‘á»ƒ táº£i weights sáºµn cÃ³ vÃ  gá»i `.train(...)`.
- Tham sá»‘ chÃ­nh:
  - `data=yaml_path`: file cáº¥u hÃ¬nh dataset
  - `epochs`, `imgsz`, `batch`, `project`, `name`.
- MÃ´ hÃ¬nh YOLOv8 sáº½ tá»± sinh cÃ¡c biá»ƒu Ä‘á»“ training vÃ  lÆ°u runs trong `runs/detect/...` (confusion matrix, results.csv).

**9) Khá»‘i Ä‘Ã¡nh giÃ¡ YOLO**
- Sau huáº¥n luyá»‡n, gá»i `model_yolo.val(split='test')` Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ trÃªn táº­p test.
- Káº¿t quáº£ chá»©a nhiá»u chá»‰ sá»‘ trong `metrics_yolo.box` nhÆ° `map50`, `map` (map50-95), `mp` (precision), `mr` (recall).
- F1 Ä‘Æ°á»£c tÃ­nh thá»§ cÃ´ng tá»« precision & recall: F1 = 2 * (P*R)/(P+R).

**10) Khá»‘i Ä‘á»c `results.csv` vÃ  váº½ trá»±c quan**
- `results.csv` chá»©a loss & metric theo epoch do YOLO xuáº¥t ra.
- Notebook tÃ¬m folder `runs/detect/*` má»›i nháº¥t, láº¥y `results.csv` vÃ  `confusion_matrix_normalized.png` Ä‘á»ƒ hiá»ƒn thá»‹.
- Váº½ loss curves vÃ  mAP curves náº¿u cÃ¡c cá»™t tá»“n táº¡i.

**11) Lá»›p dataset cho Faster R-CNN: `TomatoRCNNDataset`**
- Má»¥c tiÃªu: táº¡o `torch.utils.data.Dataset` tráº£ vá» `img_tensor` vÃ  `target` dáº¡ng dict nhÆ° PyTorch detection API mong Ä‘á»£i.
- `__init__(self, img_dir, label_dir, width, height, transforms=None)` lÆ°u Ä‘Æ°á»ng dáº«n vÃ  danh sÃ¡ch áº£nh.
- `__getitem__(self, idx)`:
  - Äá»c áº£nh báº±ng `cv2`, chuyá»ƒn sang RGB, resize vá» `(width, height)`.
  - DÃ¹ng `T.ToTensor()` Ä‘á»ƒ chuyá»ƒn sang tensor cÃ³ shape `[C,H,W]` vÃ  normalized 0..1.
  - Äá»c file label `.txt`, chuyá»ƒn má»—i dÃ²ng YOLO -> `[x1,y1,x2,y2]` trÃªn kÃ­ch thÆ°á»›c áº£nh resized.
  - `labels` Ä‘Æ°á»£c cá»™ng `+1` Ä‘á»ƒ reserve 0 cho background.
  - Náº¿u áº£nh khÃ´ng cÃ³ object, `boxes` lÃ  `torch.zeros((0,4))` vÃ  `labels` lÃ  `torch.zeros((0,))`.
  - `target` chá»©a `"boxes"`, `"labels"`, `"image_id"`.
- `collate_fn` dÃ¹ng Ä‘á»ƒ DataLoader ghÃ©p batch tráº£ vá» tuple of lists theo requirement cá»§a detection models.

**12) HÃ m `get_rcnn_model(num_classes)`**
- Táº£i `fasterrcnn_resnet50_fpn(pretrained=True)` tá»« `torchvision.models.detection`.
- Láº¥y `in_features` tá»« `roi_heads.box_predictor.cls_score.in_features` vÃ  thay predictor báº±ng `FastRCNNPredictor(in_features, num_classes)` Ä‘á»ƒ phÃ¹ há»£p sá»‘ lá»›p dataset.
- `num_classes` á»Ÿ Ä‘Ã¢y = 11 (10 disease + 1 background).

**13) Khá»‘i huáº¥n luyá»‡n Faster R-CNN**
- Chuáº©n bá»‹ `train_loader_rcnn` vÃ  `val_loader_rcnn`.
- DÃ¹ng optimizer SGD (lr=0.005, momentum=0.9, weight_decay=0.0005).
- VÃ²ng láº·p huáº¥n luyá»‡n cÆ¡ báº£n:
  - `model_rcnn.train()`
  - Vá»›i má»—i batch, chuyá»ƒn images & targets sang `device`.
  - `loss_dict = model_rcnn(images, targets)` tráº£ vá» dict cÃ¡c máº¥t mÃ¡t (classification loss, box regression, etc.).
  - `losses = sum(loss_dict.values())` rá»“i backward & step.
- Ghi láº¡i `rcnn_loss_history` Ä‘á»ƒ váº½ loss curve sau nÃ y.

**14) HÃ m `calculate_f1_rcnn(loader, model, device, conf_threshold=0.5, iou_threshold=0.5)`**
- Má»¥c tiÃªu: tÃ­nh Precision/Recall/F1 cho Faster R-CNN báº±ng phÃ©p khá»›p Ä‘Æ¡n giáº£n giá»¯a bbox dá»± Ä‘oÃ¡n vÃ  GT.
- CÃ¡ch hoáº¡t Ä‘á»™ng:
  - Dá»± Ä‘oÃ¡n `outputs = model(images)`; vá»›i má»—i áº£nh, láº¥y `pred_boxes`, `pred_scores`, `pred_labels`.
  - Lá»c cÃ¡c dá»± Ä‘oÃ¡n theo ngÆ°á»¡ng confidence `conf_threshold`.
  - Náº¿u khÃ´ng cÃ³ GT box: má»i dá»± Ä‘oÃ¡n lÃ  FP.
  - Náº¿u khÃ´ng cÃ³ dá»± Ä‘oÃ¡n: má»i GT box lÃ  FN.
  - TÃ­nh IoU giá»¯a `pred_boxes` vÃ  `gt_boxes` sá»­ dá»¥ng `torchvision.ops.box_iou`.
  - Vá»›i má»—i pred box, tÃ¬m GT cÃ³ IoU lá»›n nháº¥t; náº¿u IoU >= `iou_threshold` vÃ  nhÃ£n trÃ¹ng vÃ  GT chÆ°a Ä‘Æ°á»£c matched thÃ¬ TP++, else FP++.
  - Sau kiá»ƒm tra háº¿t predictions, FN += sá»‘ GT chÆ°a matched.
- Cuá»‘i cÃ¹ng tÃ­nh precision = TP/(TP+FP), recall = TP/(TP+FN), F1 theo cÃ´ng thá»©c chuáº©n.
- LÆ°u Ã½: Ä‘Ã¢y lÃ  cÃ¡ch Ä‘Ã¡nh giÃ¡ thá»§ cÃ´ng Ä‘Æ¡n giáº£n, cÃ³ thá»ƒ khÃ¡c so vá»›i cÃ¡ch tÃ­nh mAP Ä‘áº§y Ä‘á»§ (mAP xá»­ lÃ½ IoU thresholds vÃ  AP per class).

**15) Äo lÆ°á»ng mAP vá»›i `torchmetrics.detection.MeanAveragePrecision`**
- `MeanAveragePrecision(iou_type="bbox")` cung cáº¥p cÃ¡ch tÃ­nh mAP chuáº©n.
- Cáº§n format `preds` vÃ  `targets` theo spec cá»§a thÆ° viá»‡n (boxes, scores, labels).

**16) Khá»‘i so sÃ¡nh & váº½ biá»ƒu Ä‘á»“**
- So sÃ¡nh `yolo_map50` vs `rcnn_map50`, `yolo_f1` vs `rcnn_f1`.
- Váº½ 3 biá»ƒu Ä‘á»“ (map, f1, loss curve Faster R-CNN).

**17) Khá»‘i trá»±c quan káº¿t quáº£ (Visualization Final Block)**
- Láº¥y 1 áº£nh tá»« `WORK_DIR/images/test` chá»n ngáº«u nhiÃªn.
- Dá»± Ä‘oÃ¡n báº±ng YOLO (`model_yolo.predict(sample_img_path, imgsz=256)`), hiá»ƒn thá»‹ áº£nh do YOLO váº½.
- Dá»± Ä‘oÃ¡n báº±ng Faster R-CNN: Ä‘á»c áº£nh, resize 256x256, ToTensor, model_rcnn(img_tensor).
- Váº½ cÃ¡c há»™p cÃ³ score >0.5 lÃªn áº£nh vÃ  hiá»ƒn thá»‹ 2 áº£nh song song.
- Fix trong notebook: xá»­ lÃ½ vá»‹ trÃ­ text khi box sÃ¡t mÃ©p trÃªn (Ä‘áº·t text xuá»‘ng dÆ°á»›i náº¿u y1 < 15).

**CÃ¡c khÃ¡i niá»‡m thuáº­t toÃ¡n quan trá»ng (tÃ³m táº¯t ngáº¯n)**
- IoU (Intersection over Union): tá»‰ lá»‡ giao/ há»£p giá»¯a bbox dá»± Ä‘oÃ¡n vÃ  bbox thá»±c. DÃ¹ng lÃ m tiÃªu chÃ­ khá»›p.
- Precision: TP / (TP + FP) â€” tá»· lá»‡ dá»± Ä‘oÃ¡n Ä‘Ãºng trÃªn tá»•ng dá»± Ä‘oÃ¡n.
- Recall: TP / (TP + FN) â€” tá»· lá»‡ dá»± Ä‘oÃ¡n Ä‘Ãºng trÃªn tá»•ng GT.
- F1-score: 2 * (P*R)/(P+R) â€” trung bÃ¬nh Ä‘iá»u hÃ²a Precision vÃ  Recall.
- mAP@50: mean Average Precision vá»›i ngÆ°á»¡ng IoU=0.5. mAP@50-95 lÃ  trung bÃ¬nh mAP vá»›i IoU tá»« 0.5 Ä‘áº¿n 0.95.

---

## Ná»€N Táº¢NG LÃ THUYáº¾T CHI TIáº¾T Cá»¦A 2 THUáº¬T TOÃN

### 1. YOLOv8 (You Only Look Once version 8)

#### 1.1. Giá»›i thiá»‡u tá»•ng quan
YOLO lÃ  há» thuáº­t toÃ¡n Object Detection (phÃ¡t hiá»‡n Ä‘á»‘i tÆ°á»£ng) ra Ä‘á»i tá»« nÄƒm 2015 bá»Ÿi Joseph Redmon. KhÃ¡c vá»›i cÃ¡c phÆ°Æ¡ng phÃ¡p cá»• Ä‘iá»ƒn chia lÃ m nhiá»u giai Ä‘oáº¡n (nhÆ° R-CNN), YOLO xá»­ lÃ½ bÃ i toÃ¡n phÃ¡t hiá»‡n Ä‘á»‘i tÆ°á»£ng nhÆ° má»™t **bÃ i toÃ¡n há»“i quy duy nháº¥t** (single regression problem), dá»± Ä‘oÃ¡n trá»±c tiáº¿p bounding boxes vÃ  xÃ¡c suáº¥t lá»›p trong má»™t láº§n truyá»n qua máº¡ng (one-shot).

YOLOv8 lÃ  phiÃªn báº£n má»›i nháº¥t (2023) do Ultralytics phÃ¡t triá»ƒn, káº¿ thá»«a kiáº¿n trÃºc YOLOv5 nhÆ°ng Ä‘Æ°á»£c tá»‘i Æ°u vá» Ä‘á»™ chÃ­nh xÃ¡c, tá»‘c Ä‘á»™ vÃ  kháº£ nÄƒng triá»ƒn khai.

#### 1.2. Kiáº¿n trÃºc tá»•ng quan YOLOv8

YOLOv8 bao gá»“m 3 thÃ nh pháº§n chÃ­nh:

**a) Backbone (XÆ°Æ¡ng sá»‘ng - trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng)**
- Sá»­ dá»¥ng kiáº¿n trÃºc **CSPDarknet** (Cross Stage Partial Darknet) vá»›i cÃ¡c khá»‘i C2f (CSPLayer with 2 convolutions).
- Má»¥c Ä‘Ã­ch: trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng Ä‘a tá»· lá»‡ (multi-scale features) tá»« áº£nh Ä‘áº§u vÃ o.
- Qua nhiá»u lá»›p convolution, pooling, áº£nh Ä‘Æ°á»£c giáº£m kÃ­ch thÆ°á»›c dáº§n vÃ  táº¡o ra feature maps á»Ÿ cÃ¡c Ä‘á»™ phÃ¢n giáº£i khÃ¡c nhau (vÃ­ dá»¥: 80Ã—80, 40Ã—40, 20Ã—20 cho áº£nh 640Ã—640).
- **CÆ¡ cháº¿ hoáº¡t Ä‘á»™ng**: 
  - Input: áº£nh RGB shape `[B, 3, H, W]`
  - Output: nhiá»u feature maps á»Ÿ cÃ¡c scale khÃ¡c nhau, mÃ£ hÃ³a thÃ´ng tin tá»« chi tiáº¿t nhá» (texture) Ä‘áº¿n ngá»¯ cáº£nh lá»›n (object-level).

**b) Neck (Cá»• - káº¿t há»£p Ä‘áº·c trÆ°ng Ä‘a tá»· lá»‡)**
- Sá»­ dá»¥ng cáº¥u trÃºc **PAN (Path Aggregation Network)** vÃ  **FPN (Feature Pyramid Network)**.
- Má»¥c Ä‘Ã­ch: trá»™n thÃ´ng tin tá»« cÃ¡c táº§ng feature khÃ¡c nhau Ä‘á»ƒ cáº£i thiá»‡n kháº£ nÄƒng phÃ¡t hiá»‡n Ä‘á»‘i tÆ°á»£ng á»Ÿ nhiá»u kÃ­ch thÆ°á»›c.
- **PAN**: truyá»n thÃ´ng tin tá»« bottom-up (tá»« táº§ng cao xuá»‘ng táº§ng tháº¥p) Ä‘á»ƒ bá»• sung ngá»¯ cáº£nh.
- **FPN**: truyá»n thÃ´ng tin tá»« top-down (tá»« táº§ng tháº¥p lÃªn táº§ng cao) Ä‘á»ƒ tÄƒng Ä‘á»™ phÃ¢n giáº£i chi tiáº¿t.
- Káº¿t quáº£: táº¡o ra 3 Ä‘áº§u ra feature maps (thÆ°á»ng gá»i lÃ  P3, P4, P5) tÆ°Æ¡ng á»©ng vá»›i 3 kÃ­ch thÆ°á»›c Ä‘á»‘i tÆ°á»£ng: nhá», trung bÃ¬nh, lá»›n.

**c) Head (Äáº§u - dá»± Ä‘oÃ¡n káº¿t quáº£)**
- YOLOv8 sá»­ dá»¥ng **Decoupled Head** (tÃ¡ch rá»i) vÃ  **Anchor-free**.
- Thay vÃ¬ dÃ¹ng anchor boxes cá»‘ Ä‘á»‹nh nhÆ° YOLOv5, YOLOv8 dá»± Ä‘oÃ¡n trá»±c tiáº¿p tÃ¢m Ä‘á»‘i tÆ°á»£ng vÃ  kÃ­ch thÆ°á»›c box.
- Má»—i cell trong feature map dá»± Ä‘oÃ¡n:
  - **Bounding box**: 4 giÃ¡ trá»‹ `(x, y, w, h)` (tÃ¢m vÃ  kÃ­ch thÆ°á»›c).
  - **Objectness score**: xÃ¡c suáº¥t cÃ³ Ä‘á»‘i tÆ°á»£ng trong cell Ä‘Ã³.
  - **Class probabilities**: vector xÃ¡c suáº¥t cho má»—i lá»›p (10 lá»›p trong trÆ°á»ng há»£p nÃ y).
- **Anchor-free**: khÃ´ng cáº§n Ä‘á»‹nh nghÄ©a trÆ°á»›c anchor boxes, giÃºp mÃ´ hÃ¬nh linh hoáº¡t vÃ  dá»… huáº¥n luyá»‡n hÆ¡n.

#### 1.3. CÆ¡ cháº¿ hoáº¡t Ä‘á»™ng chi tiáº¿t

**BÆ°á»›c 1: Chia áº£nh thÃ nh lÆ°á»›i (Grid)**
- áº¢nh Ä‘áº§u vÃ o Ä‘Æ°á»£c chia thÃ nh lÆ°á»›i SÃ—S (vÃ­ dá»¥ 20Ã—20).
- Má»—i Ã´ lÆ°á»›i (cell) chá»‹u trÃ¡ch nhiá»‡m phÃ¡t hiá»‡n Ä‘á»‘i tÆ°á»£ng cÃ³ tÃ¢m náº±m trong Ã´ Ä‘Ã³.

**BÆ°á»›c 2: Dá»± Ä‘oÃ¡n Bounding Boxes**
- Vá»›i má»—i cell, máº¡ng dá»± Ä‘oÃ¡n nhiá»u bounding boxes (thÆ°á»ng 3 boxes/cell tÆ°Æ¡ng á»©ng 3 scale).
- Má»—i box Ä‘Æ°á»£c mÃ´ táº£ bá»Ÿi:
  - `(tx, ty)`: offset tá»« gÃ³c trÃªn-trÃ¡i cá»§a cell â†’ tÃ¢m box.
  - `(tw, th)`: log-space width/height.
  - CÃ´ng thá»©c tÃ­nh tá»a Ä‘á»™ thá»±c:
    ```
    bx = sigmoid(tx) + cx  (cx: cá»™t cá»§a cell)
    by = sigmoid(ty) + cy  (cy: hÃ ng cá»§a cell)
    bw = exp(tw)
    bh = exp(th)
    ```

**BÆ°á»›c 3: TÃ­nh Objectness vÃ  Class Probability**
- **Objectness**: xÃ¡c suáº¥t tá»« 0â†’1 ráº±ng cell Ä‘Ã³ chá»©a Ä‘á»‘i tÆ°á»£ng.
- **Class probability**: vá»›i 10 lá»›p bá»‡nh, máº¡ng xuáº¥t vector 10 chiá»u qua softmax.

**BÆ°á»›c 4: Ãp dá»¥ng Non-Maximum Suppression (NMS)**
- Sau khi cÃ³ hÃ ng ngÃ n boxes dá»± Ä‘oÃ¡n, NMS loáº¡i bá» cÃ¡c boxes trÃ¹ng láº·p:
  - Sáº¯p xáº¿p boxes theo objectness Ã— class_prob.
  - Giá»¯ box cÃ³ score cao nháº¥t, loáº¡i bá» cÃ¡c boxes khÃ¡c cÃ³ IoU > ngÆ°á»¡ng (thÆ°á»ng 0.45).
- Káº¿t quáº£: chá»‰ giá»¯ láº¡i cÃ¡c boxes tá»‘t nháº¥t, khÃ´ng overlap quÃ¡ nhiá»u.

#### 1.4. HÃ m máº¥t mÃ¡t (Loss Function)

YOLOv8 sá»­ dá»¥ng 3 thÃ nh pháº§n loss:

**a) Classification Loss (CLS Loss)**
- Äo sai lá»‡ch giá»¯a phÃ¢n bá»‘ lá»›p dá»± Ä‘oÃ¡n vÃ  nhÃ£n thá»±c.
- DÃ¹ng **Binary Cross Entropy** (BCE) hoáº·c **Focal Loss** Ä‘á»ƒ xá»­ lÃ½ class imbalance.

**b) Localization Loss (Box Loss)**
- Äo Ä‘á»™ chÃ­nh xÃ¡c vá»‹ trÃ­ bounding box.
- YOLOv8 dÃ¹ng **CIoU Loss** (Complete IoU):
  - CIoU = IoU loss + Distance penalty + Aspect ratio penalty
  - Má»¥c tiÃªu: boxes dá»± Ä‘oÃ¡n khÃ´ng chá»‰ overlap tá»‘t mÃ  cÃ²n cÃ³ tÃ¢m gáº§n nhau vÃ  tá»· lá»‡ khung hÃ¬nh Ä‘Ãºng.

**c) Objectness Loss (OBJ Loss)**
- Äo xÃ¡c suáº¥t cÃ³/khÃ´ng cÃ³ Ä‘á»‘i tÆ°á»£ng trong cell.
- Chá»‰ Ã¡p dá»¥ng cho cÃ¡c cell cÃ³ ground truth object.

**Tá»•ng Loss**:
```
Total Loss = Î»â‚ Ã— CLS_Loss + Î»â‚‚ Ã— Box_Loss + Î»â‚ƒ Ã— OBJ_Loss
```
(Î» lÃ  cÃ¡c há»‡ sá»‘ cÃ¢n báº±ng, tá»± Ä‘á»™ng Ä‘iá»u chá»‰nh theo dataset)

#### 1.5. Æ¯u Ä‘iá»ƒm vÃ  háº¡n cháº¿

**Æ¯u Ä‘iá»ƒm**:
- **Tá»‘c Ä‘á»™ cao**: xá»­ lÃ½ 1 áº£nh chá»‰ máº¥t 2-5ms trÃªn GPU (Ä‘áº¡t 200+ FPS), phÃ¹ há»£p realtime.
- **Dá»… huáº¥n luyá»‡n**: anchor-free, Ã­t hyperparameter cáº§n tune.
- **Linh hoáº¡t**: há»— trá»£ nhiá»u task (detection, segmentation, classification, pose estimation).
- **TÃ­ch há»£p tá»‘t**: Ultralytics cung cáº¥p API Ä‘Æ¡n giáº£n, dá»… export sang ONNX, TensorRT, CoreML.

**Háº¡n cháº¿**:
- **KhÃ³ phÃ¡t hiá»‡n object nhá» sÃ¡t nhau**: do má»—i cell chá»‰ dá»± Ä‘oÃ¡n sá»‘ box giá»›i háº¡n.
- **KÃ©m chÃ­nh xÃ¡c hÆ¡n 2-stage detectors** trong má»™t sá»‘ trÆ°á»ng há»£p phá»©c táº¡p (nhÆ°ng gap Ä‘Ã£ giáº£m Ä‘Ã¡ng ká»ƒ á»Ÿ YOLOv8).

---

### 2. Faster R-CNN (Region-based Convolutional Neural Network)

#### 2.1. Giá»›i thiá»‡u tá»•ng quan

Faster R-CNN lÃ  thuáº­t toÃ¡n Object Detection thuá»™c há» **2-stage detectors**, ra Ä‘á»i nÄƒm 2015 bá»Ÿi Shaoqing Ren, Kaiming He vÃ  nhÃ³m Microsoft Research.

KhÃ¡c vá»›i YOLO (1-stage), Faster R-CNN chia bÃ i toÃ¡n thÃ nh 2 bÆ°á»›c rÃµ rÃ ng:
1. **Stage 1**: Táº¡o cÃ¡c vÃ¹ng Ä‘á» xuáº¥t (Region Proposals) cÃ³ kháº£ nÄƒng chá»©a Ä‘á»‘i tÆ°á»£ng.
2. **Stage 2**: PhÃ¢n loáº¡i vÃ  tinh chá»‰nh (refine) cÃ¡c proposals Ä‘Ã³.

Kiáº¿n trÃºc nÃ y cho phÃ©p Ä‘áº¡t Ä‘á»™ chÃ­nh xÃ¡c cao nhÆ°ng tá»‘c Ä‘á»™ cháº­m hÆ¡n YOLO.

#### 2.2. Kiáº¿n trÃºc chi tiáº¿t Faster R-CNN

Faster R-CNN gá»“m 4 thÃ nh pháº§n chÃ­nh:

**a) Backbone Network (CNN trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng)**
- ThÆ°á»ng dÃ¹ng **ResNet-50** hoáº·c **ResNet-101** vá»›i **FPN** (Feature Pyramid Network).
- Input: áº£nh RGB `[B, 3, H, W]`.
- Output: feature maps `[B, C, H', W']` (vÃ­ dá»¥ C=2048 channels, H'=H/16, W'=W/16).
- **ResNet-50**: máº¡ng tÃ­ch cháº­p 50 lá»›p sá»­ dá»¥ng **residual connections** (skip connections) Ä‘á»ƒ trÃ¡nh vanishing gradient, há»c Ä‘Æ°á»£c Ä‘áº·c trÆ°ng sÃ¢u.
- **FPN**: táº¡o kim tá»± thÃ¡p Ä‘áº·c trÆ°ng á»Ÿ nhiá»u tá»· lá»‡ (P2, P3, P4, P5, P6) Ä‘á»ƒ phÃ¡t hiá»‡n objects á»Ÿ nhiá»u kÃ­ch thÆ°á»›c.

**b) Region Proposal Network (RPN)**
- ÄÃ¢y lÃ  **trÃ¡i tim cá»§a Faster R-CNN**, thay tháº¿ Selective Search (phÆ°Æ¡ng phÃ¡p cá»• Ä‘iá»ƒn cháº­m).
- **Má»¥c Ä‘Ã­ch**: quÃ©t feature map vÃ  Ä‘á» xuáº¥t ~1000-2000 vÃ¹ng cÃ³ kháº£ nÄƒng chá»©a Ä‘á»‘i tÆ°á»£ng.

**CÆ¡ cháº¿ hoáº¡t Ä‘á»™ng RPN**:
1. **Anchor Boxes**: táº¡i má»—i vá»‹ trÃ­ trÃªn feature map, Ä‘áº·t k anchor boxes (thÆ°á»ng k=9) vá»›i 3 tá»· lá»‡ (1:1, 1:2, 2:1) vÃ  3 kÃ­ch thÆ°á»›c (128Â², 256Â², 512Â² pixels).
2. **Sliding Window**: má»™t cá»­a sá»• trÆ°á»£t 3Ã—3 conv quÃ©t toÃ n bá»™ feature map.
3. **Dá»± Ä‘oÃ¡n 2 nhÃ¡nh**:
   - **Objectness score**: 2 giÃ¡ trá»‹ (object / non-object) cho má»—i anchor â†’ dÃ¹ng softmax.
   - **Box regression**: 4 giÃ¡ trá»‹ (Î”x, Î”y, Î”w, Î”h) Ä‘á»ƒ Ä‘iá»u chá»‰nh anchor thÃ nh proposal chÃ­nh xÃ¡c hÆ¡n.
4. **Lá»c proposals**: 
   - Loáº¡i bá» boxes ngoÃ i biÃªn áº£nh.
   - Ãp dá»¥ng NMS vá»›i IoU threshold (0.7) Ä‘á»ƒ giáº£m overlap.
   - Giá»¯ top-N proposals cÃ³ objectness cao nháº¥t (N~2000 khi train, N~1000 khi test).

**CÃ´ng thá»©c Box Regression**:
```
Predicted box:
x_pred = x_anchor + Î”x Ã— w_anchor
y_pred = y_anchor + Î”y Ã— h_anchor
w_pred = w_anchor Ã— exp(Î”w)
h_pred = h_anchor Ã— exp(Î”h)
```

**c) RoI Pooling / RoI Align**
- **Váº¥n Ä‘á»**: Proposals cÃ³ kÃ­ch thÆ°á»›c khÃ¡c nhau, nhÆ°ng FC layer cáº§n input cá»‘ Ä‘á»‹nh.
- **Giáº£i phÃ¡p**: 
  - **RoI Pooling** (cÅ©): chia má»—i proposal thÃ nh lÆ°á»›i cá»‘ Ä‘á»‹nh 7Ã—7, max pooling má»—i Ã´.
  - **RoI Align** (má»›i, chÃ­nh xÃ¡c hÆ¡n): dÃ¹ng bilinear interpolation Ä‘á»ƒ trÃ¡nh quantization errors.
- Output: feature vector cá»‘ Ä‘á»‹nh cho má»—i proposal (vÃ­ dá»¥ 7Ã—7Ã—C).

**d) Fast R-CNN Head (PhÃ¢n loáº¡i & Refine Box)**
- Má»—i RoI feature Ä‘i qua 2 nhÃ¡nh:

**NhÃ¡nh 1: Classification**
- Fully Connected layers + Softmax.
- Output: xÃ¡c suáº¥t cho (num_classes + 1) lá»›p (bao gá»“m background class).
- Loss: Cross Entropy Loss.

**NhÃ¡nh 2: Bounding Box Regression**
- FC layers dá»± Ä‘oÃ¡n 4 giÃ¡ trá»‹ refine (Î”x, Î”y, Î”w, Î”h) Ä‘á»ƒ Ä‘iá»u chá»‰nh proposal thÃ nh final box.
- Loss: Smooth L1 Loss.

#### 2.3. Luá»“ng xá»­ lÃ½ end-to-end

**Training Phase**:
1. áº¢nh â†’ Backbone â†’ Feature maps.
2. RPN nháº­n feature maps:
   - Táº¡o anchors táº¡i má»—i vá»‹ trÃ­.
   - Dá»± Ä‘oÃ¡n objectness & box deltas.
   - TÃ­nh RPN loss (cls loss + box regression loss).
3. Proposals Ä‘Æ°á»£c lá»c vÃ  match vá»›i Ground Truth:
   - IoU > 0.5 â†’ positive samples.
   - IoU < 0.3 â†’ negative samples.
4. RoI Align láº¥y features cho má»—i proposal.
5. Fast R-CNN Head:
   - Dá»± Ä‘oÃ¡n class vÃ  refine box.
   - TÃ­nh Fast R-CNN loss (cls loss + box regression loss).
6. **Multi-task Loss**:
   ```
   Total Loss = L_RPN_cls + Î»â‚ Ã— L_RPN_box + L_RCNN_cls + Î»â‚‚ Ã— L_RCNN_box
   ```

**Inference Phase**:
1. áº¢nh â†’ Backbone â†’ Feature maps.
2. RPN táº¡o proposals (~1000).
3. RoI Align + Fast R-CNN Head dá»± Ä‘oÃ¡n class & box cho má»—i proposal.
4. Ãp dá»¥ng NMS (IoU threshold ~0.5) Ä‘á»ƒ loáº¡i bá» overlap.
5. Output: final detections vá»›i class labels vÃ  boxes.

#### 2.4. Chi tiáº¿t cÃ¡c Loss Functions

**a) RPN Classification Loss**
- Binary Cross Entropy giá»¯a objectness dá»± Ä‘oÃ¡n vÃ  label (object/non-object).
- Chá»‰ tÃ­nh cho cÃ¡c anchors Ä‘Æ°á»£c assign (matched vá»›i GT hoáº·c background).

**b) RPN Box Regression Loss**
- Smooth L1 Loss giá»¯a predicted box deltas vÃ  target deltas.
- Chá»‰ tÃ­nh cho positive anchors (cÃ³ GT object).

**c) Fast R-CNN Classification Loss**
- Cross Entropy cho (num_classes + 1).
- Background class giÃºp model há»c phÃ¢n biá»‡t object/non-object rÃµ rÃ ng.

**d) Fast R-CNN Box Regression Loss**
- Smooth L1 Loss cho 4 giÃ¡ trá»‹ (Î”x, Î”y, Î”w, Î”h).
- Chá»‰ tÃ­nh cho proposals cÃ³ class â‰  background.

**Smooth L1 Loss Formula**:
```
SmoothL1(x) = 0.5 Ã— xÂ²         if |x| < 1
            = |x| - 0.5        otherwise
```
(Ã­t nháº¡y cáº£m vá»›i outliers hÆ¡n L2 loss)

#### 2.5. Æ¯u Ä‘iá»ƒm vÃ  háº¡n cháº¿

**Æ¯u Ä‘iá»ƒm**:
- **Äá»™ chÃ­nh xÃ¡c cao**: 2-stage design cho phÃ©p refine proposals ká»¹ lÆ°á»¡ng.
- **PhÃ¡t hiá»‡n tá»‘t object nhá» vÃ  overlap**: RPN táº¡o nhiá»u proposals Ä‘a dáº¡ng.
- **Backbone máº¡nh máº½**: ResNet-50-FPN há»c Ä‘Æ°á»£c Ä‘áº·c trÆ°ng phong phÃº.
- **á»”n Ä‘á»‹nh**: Ã­t bá»‹ miss detection so vá»›i 1-stage detectors.

**Háº¡n cháº¿**:
- **Cháº­m**: ~100-200ms/áº£nh (so vá»›i YOLO 2-5ms). KhÃ´ng phÃ¹ há»£p realtime.
- **Phá»©c táº¡p**: nhiá»u thÃ nh pháº§n cáº§n tune (anchor sizes, IoU thresholds, NMS thresholds).
- **Tá»‘n bá»™ nhá»›**: RPN + RoI Align + Head cáº§n nhiá»u GPU memory.

---

### 3. So sÃ¡nh YOLOv8 vs Faster R-CNN

| TiÃªu chÃ­ | YOLOv8 | Faster R-CNN |
|----------|--------|--------------|
| **Loáº¡i detector** | 1-stage (single-shot) | 2-stage |
| **Tá»‘c Ä‘á»™** | Ráº¥t nhanh (~200 FPS) | Cháº­m (~5-10 FPS) |
| **Äá»™ chÃ­nh xÃ¡c** | Cao (mAP ~0.6-0.8) | Ráº¥t cao (mAP ~0.7-0.9) |
| **Anchor** | Anchor-free | Anchor-based (RPN) |
| **Kiáº¿n trÃºc** | CSPDarknet + PAN/FPN + Decoupled Head | ResNet + FPN + RPN + RoI Align + FC Head |
| **PhÃ¹ há»£p** | Realtime, edge devices, video streaming | NghiÃªn cá»©u, yÃªu cáº§u accuracy cao, offline processing |
| **KhÃ³ khÄƒn** | Object nhá», sÃ¡t nhau | Cháº­m, tá»‘n tÃ i nguyÃªn |
| **Triá»ƒn khai** | Dá»… (Ultralytics API, export ONNX/TensorRT) | KhÃ³ hÆ¡n (cáº§n optimize ká»¹) |

**Káº¿t luáº­n**: 
- DÃ¹ng **YOLOv8** khi cáº§n **tá»‘c Ä‘á»™** (camera realtime, robot, drones).
- DÃ¹ng **Faster R-CNN** khi cáº§n **Ä‘á»™ chÃ­nh xÃ¡c tuyá»‡t Ä‘á»‘i** (y táº¿, an ninh, phÃ¢n tÃ­ch áº£nh cháº¥t lÆ°á»£ng cao).

---

### 4. CÃ¡c khÃ¡i niá»‡m nÃ¢ng cao

#### 4.1. IoU (Intersection over Union)
- Äá»™ Ä‘o overlap giá»¯a 2 boxes A vÃ  B:
  ```
  IoU = Area(A âˆ© B) / Area(A âˆª B)
  ```
- GiÃ¡ trá»‹ tá»« 0 (khÃ´ng overlap) Ä‘áº¿n 1 (overlap hoÃ n toÃ n).
- DÃ¹ng Ä‘á»ƒ:
  - Matching predictions vá»›i ground truth (IoU > 0.5 â†’ match).
  - NMS (loáº¡i box cÃ³ IoU > threshold vá»›i box score cao hÆ¡n).
  - Loss function (CIoU, DIoU, GIoU).

#### 4.2. NMS (Non-Maximum Suppression)
**Thuáº­t toÃ¡n**:
1. Sáº¯p xáº¿p táº¥t cáº£ boxes theo score giáº£m dáº§n.
2. Chá»n box cÃ³ score cao nháº¥t â†’ thÃªm vÃ o káº¿t quáº£.
3. Loáº¡i bá» táº¥t cáº£ boxes cÃ³ IoU > threshold vá»›i box Ä‘Ã£ chá»n.
4. Láº·p láº¡i bÆ°á»›c 2-3 cho Ä‘áº¿n khi háº¿t boxes.

**Ã nghÄ©a**: TrÃ¡nh detect cÃ¹ng 1 object nhiá»u láº§n.

#### 4.3. mAP (mean Average Precision)
- **Average Precision (AP)**: diá»‡n tÃ­ch dÆ°á»›i Precision-Recall curve cho 1 class.
- **mAP**: trung bÃ¬nh AP cá»§a táº¥t cáº£ classes.
- **mAP@50**: tÃ­nh AP vá»›i IoU threshold = 0.5.
- **mAP@50-95**: trung bÃ¬nh mAP vá»›i IoU tá»« 0.5 â†’ 0.95 (bÆ°á»›c 0.05) â†’ Ä‘Ã¡nh giÃ¡ toÃ n diá»‡n hÆ¡n.

**Ã nghÄ©a**: mAP cao â†’ model phÃ¡t hiá»‡n chÃ­nh xÃ¡c nhiá»u objects á»Ÿ nhiá»u IoU thresholds.

#### 4.4. Anchor Boxes
- **Pre-defined boxes** vá»›i tá»· lá»‡ vÃ  kÃ­ch thÆ°á»›c cá»‘ Ä‘á»‹nh.
- Äáº·t táº¡i má»—i vá»‹ trÃ­ feature map lÃ m "template" Ä‘á»ƒ detect objects.
- Model há»c **offsets** (Î”x, Î”y, Î”w, Î”h) Ä‘á»ƒ Ä‘iá»u chá»‰nh anchor thÃ nh bounding box cuá»‘i.
- **YOLOv8 bá» anchors** â†’ dá»± Ä‘oÃ¡n trá»±c tiáº¿p center/size â†’ Ä‘Æ¡n giáº£n hÆ¡n, linh hoáº¡t hÆ¡n.

#### 4.5. Feature Pyramid Network (FPN)
- Kiáº¿n trÃºc táº¡o kim tá»± thÃ¡p Ä‘áº·c trÆ°ng tá»« backbone.
- **Top-down pathway**: upsample feature maps tá»« táº§ng sÃ¢u (low resolution, semantic rich) vÃ  merge vá»›i táº§ng nÃ´ng (high resolution, detail rich).
- **Má»¥c Ä‘Ã­ch**: káº¿t há»£p thÃ´ng tin semantic vÃ  spatial â†’ phÃ¡t hiá»‡n tá»‘t objects á»Ÿ nhiá»u scales.

---

## Káº¾T LUáº¬N Vá»€ 2 THUáº¬T TOÃN

**YOLOv8** lÃ  lá»±a chá»n tá»‘t cho project nÃ y náº¿u:
- Cáº§n triá»ƒn khai trÃªn thiáº¿t bá»‹ edge (Raspberry Pi, Jetson Nano).
- Xá»­ lÃ½ video realtime (camera nÃ´ng nghiá»‡p, drone giÃ¡m sÃ¡t).
- Æ¯u tiÃªn tá»‘c Ä‘á»™ huáº¥n luyá»‡n vÃ  inference.

**Faster R-CNN** phÃ¹ há»£p náº¿u:
- YÃªu cáº§u Ä‘á»™ chÃ­nh xÃ¡c tuyá»‡t Ä‘á»‘i (á»©ng dá»¥ng nghiÃªn cá»©u, cháº©n Ä‘oÃ¡n quan trá»ng).
- Xá»­ lÃ½ áº£nh offline (khÃ´ng cáº§n realtime).
- CÃ³ Ä‘á»§ tÃ i nguyÃªn GPU Ä‘á»ƒ huáº¥n luyá»‡n lÃ¢u hÆ¡n.

Trong notebook `tomato_leaf.ipynb`, cáº£ 2 models Ä‘á»u Ä‘Æ°á»£c huáº¥n luyá»‡n vÃ  so sÃ¡nh trÃªn cÃ¹ng dataset tomato disease detection, giÃºp báº¡n hiá»ƒu rÃµ trade-off giá»¯a **speed vs accuracy**.

---

## CÃ‚U Há»ŽI THÆ¯á»œNG Gáº¶P Vá»€ KIáº¾N TRÃšC MODEL

### Q1: YOLOv8 vÃ  Faster R-CNN cÃ³ bao nhiÃªu lá»›p? Lá»›p nÃ o lÃ  lá»›p chÃ­nh?

#### **YOLOv8s Architecture**
```
Tá»•ng sá»‘ layers: ~168 layers (modules)
```

**3 nhÃ³m lá»›p chÃ­nh:**

**1. Backbone (CSPDarknet) - ~80 layers**
- **Chá»©c nÄƒng**: TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng tá»« áº£nh Ä‘áº§u vÃ o
- **ThÃ nh pháº§n**:
  - Conv layers: CÃ¡c lá»›p tÃ­ch cháº­p cÆ¡ báº£n (3Ã—3, 1Ã—1 kernels)
  - **C2f blocks** (CSP Bottleneck with 2 convolutions): Lá»›p chÃ­nh xuáº¥t hiá»‡n nhiá»u nháº¥t
  - SPPF (Spatial Pyramid Pooling - Fast): Káº¿t há»£p features á»Ÿ nhiá»u scale
- **Lá»›p chÃ­nh**: **C2f (CSPLayer with 2 Convolutions Fast)**
  - ÄÃ¢y lÃ  building block cá»‘t lÃµi cá»§a YOLOv8
  - Káº¿t há»£p CSP (Cross Stage Partial) Ä‘á»ƒ giáº£m computation
  - Xuáº¥t hiá»‡n trong cáº£ Backbone vÃ  Neck
  - Cáº£i thiá»‡n gradient flow vÃ  giáº£m parameters

**2. Neck (PAN/FPN) - ~40 layers**
- **Chá»©c nÄƒng**: Káº¿t há»£p Ä‘áº·c trÆ°ng Ä‘a tá»· lá»‡ (multi-scale feature fusion)
- **ThÃ nh pháº§n**:
  - Upsample layers (top-down pathway): TÄƒng resolution
  - Concat layers (lateral connections): Káº¿t ná»‘i skip connections
  - C2f blocks: Xá»­ lÃ½ thÃ´ng tin sau khi merge
  - Downsample layers (bottom-up pathway): Truyá»n thÃ´ng tin tá»« low-level lÃªn
- **Output**: 3 feature maps (P3, P4, P5) cho small/medium/large objects

**3. Head (Detection Head) - ~48 layers**
- **Chá»©c nÄƒng**: Dá»± Ä‘oÃ¡n bounding boxes vÃ  classes
- **ThÃ nh pháº§n**:
  - 3 detection layers tÆ°Æ¡ng á»©ng 3 scales (P3: 80Ã—80, P4: 40Ã—40, P5: 20Ã—20)
  - **Decoupled Head** (tÃ¡ch rá»i):
    - Classification branch: Conv layers dá»± Ä‘oÃ¡n class probabilities
    - Box regression branch: Conv layers dá»± Ä‘oÃ¡n (x, y, w, h)
  - **Anchor-free**: KhÃ´ng dÃ¹ng anchor boxes, dá»± Ä‘oÃ¡n trá»±c tiáº¿p

**Táº¡i sao C2f lÃ  lá»›p chÃ­nh?**
- Xuáº¥t hiá»‡n **nhiá»u nháº¥t** trong toÃ n bá»™ kiáº¿n trÃºc (cáº£ Backbone láº«n Neck)
- Chá»‹u trÃ¡ch nhiá»‡m **há»c Ä‘áº·c trÆ°ng** chÃ­nh tá»« áº£nh
- Káº¿t há»£p **CSP architecture** giÃºp:
  - Giáº£m 50% computation so vá»›i C3 (YOLOv5)
  - TÄƒng gradient flow (trÃ¡nh vanishing gradient)
  - Duy trÃ¬ accuracy cao

---

#### **Faster R-CNN ResNet-50-FPN Architecture**
```
Tá»•ng sá»‘ layers: ~175 layers (modules)
```

**4 nhÃ³m lá»›p chÃ­nh:**

**1. Backbone (ResNet-50) - ~50 layers convolution**
- **Chá»©c nÄƒng**: TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng sÃ¢u tá»« áº£nh
- **Cáº¥u trÃºc chi tiáº¿t**:
  - **Layer 1**: Conv1 (7Ã—7 conv, stride=2) + BatchNorm + ReLU + MaxPool
  - **Layer 2-5**: 4 nhÃ³m Residual Blocks:
    - Layer 2: 3 Bottleneck blocks â†’ output (256 channels, H/4, W/4)
    - Layer 3: 4 Bottleneck blocks â†’ output (512 channels, H/8, W/8)
    - Layer 4: 6 Bottleneck blocks â†’ output (1024 channels, H/16, W/16)
    - Layer 5: 3 Bottleneck blocks â†’ output (2048 channels, H/32, W/32)

- **Lá»›p chÃ­nh**: **Residual Block (Bottleneck)**
  ```
  Input (x)
    â†“
  1Ã—1 Conv (giáº£m channels) â†’ BatchNorm â†’ ReLU
    â†“
  3Ã—3 Conv (extract features) â†’ BatchNorm â†’ ReLU
    â†“
  1Ã—1 Conv (tÄƒng channels) â†’ BatchNorm
    â†“
  + Skip Connection (x) â† ÄÃ¢y lÃ  Ä‘iá»ƒm Ä‘áº·c biá»‡t!
    â†“
  ReLU â†’ Output
  ```
  - **Táº¡i sao lÃ  lá»›p chÃ­nh?**
    - Giáº£i quyáº¿t váº¥n Ä‘á» **vanishing gradient** (máº¡ng sÃ¢u 50+ layers váº«n train Ä‘Æ°á»£c)
    - Skip connection cho phÃ©p gradient flow trá»±c tiáº¿p
    - Há»c Ä‘Æ°á»£c Ä‘áº·c trÆ°ng **identity mapping** + residual features

**2. FPN (Feature Pyramid Network) - ~20 layers**
- **Chá»©c nÄƒng**: Táº¡o kim tá»± thÃ¡p Ä‘áº·c trÆ°ng Ä‘a tá»· lá»‡
- **ThÃ nh pháº§n**:
  - **Top-down pathway**: 
    - 1Ã—1 conv giáº£m channels (2048â†’256, 1024â†’256, 512â†’256)
    - Upsample (Ã—2) Ä‘á»ƒ tÄƒng resolution
  - **Lateral connections**: 
    - 1Ã—1 conv tá»« ResNet layers
    - Element-wise addition vá»›i top-down features
  - **Bottom-up pathway**:
    - 3Ã—3 conv Ä‘á»ƒ refine merged features
  - **Output**: P2, P3, P4, P5, P6 (5 levels)

**3. RPN (Region Proposal Network) - ~10 layers**
- **Chá»©c nÄƒng**: Táº¡o ~1000-2000 region proposals
- **ThÃ nh pháº§n**:
  - **Sliding window**: 3Ã—3 conv trÆ°á»£t trÃªn feature map
    ```
    Input: (B, 256, H, W)
    3Ã—3 Conv â†’ (B, 512, H, W)
    ```
  - **2 nhÃ¡nh song song**:
    - **Objectness branch**: 1Ã—1 conv â†’ (B, 2Ã—num_anchors, H, W)
      - 2 classes: object / background
      - 9 anchors/cell (3 scales Ã— 3 aspect ratios)
    - **Box regression branch**: 1Ã—1 conv â†’ (B, 4Ã—num_anchors, H, W)
      - 4 coords: (Î”x, Î”y, Î”w, Î”h)

**4. RoI Head (Detection Head) - ~15 layers**
- **Chá»©c nÄƒng**: PhÃ¢n loáº¡i vÃ  refine proposals
- **ThÃ nh pháº§n**:
  - **RoI Align**: Differentiable pooling (7Ã—7 output)
    - DÃ¹ng bilinear interpolation (khÃ´ng quantization)
    - Input: proposals + feature maps
    - Output: (num_proposals, 256, 7, 7)
  - **2 FC layers**: Má»—i layer 1024 units
    ```
    (256Ã—7Ã—7) â†’ Flatten â†’ FC(1024) â†’ ReLU â†’ FC(1024) â†’ ReLU
    ```
  - **2 nhÃ¡nh song song**:
    - **Classification head**: FC(num_classes) â†’ Softmax
    - **Box regression head**: FC(4Ã—num_classes) â†’ Box deltas

**Lá»›p chÃ­nh: Residual Block (Bottleneck)**
- Chiáº¿m **pháº§n lá»›n computation** trong toÃ n bá»™ máº¡ng
- CÃ³ trong 4 layers (Layer 2-5) vá»›i tá»•ng 3+4+6+3 = **16 blocks**
- Má»—i block cÃ³ 3 conv layers (1Ã—1 â†’ 3Ã—3 â†’ 1Ã—1)
- **Táº¡i sao quan trá»ng?**
  - Cho phÃ©p train máº¡ng ráº¥t sÃ¢u (50-152 layers)
  - Skip connection = "highway" cho gradient
  - Learn cáº£ identity mapping vÃ  residual features

---

### Q2: Trong code sá»­ dá»¥ng bao nhiÃªu lá»›p? Táº¡i sao?

#### **YOLOv8s trong code**
```python
model_yolo = YOLO('yolov8s.pt')  # Pre-trained weights
```

**Sá»‘ lá»›p sá»­ dá»¥ng: 168 layers**

**Táº¡i sao chá»n YOLOv8s (small)?**

| LÃ½ do | Giáº£i thÃ­ch |
|-------|------------|
| **CÃ¢n báº±ng speed-accuracy** | mAP ~44.9% trÃªn COCO, 200+ FPS trÃªn V100 GPU |
| **PhÃ¹ há»£p Colab GPU** | Model size ~50MB, train Ä‘Æ°á»£c vá»›i RAM/VRAM háº¡n cháº¿ (T4 16GB) |
| **Pretrained trÃªn COCO** | 80 classes Ä‘Ã£ há»c features cÆ¡ báº£n (edges, textures, shapes) |
| **Transfer learning hiá»‡u quáº£** | Chá»‰ cáº§n 15 epochs Ä‘á»ƒ fine-tune cho tomato diseases |
| **KhÃ´ng cáº§n tune anchors** | Anchor-free â†’ Ã­t hyperparameters, dá»… train |

**So sÃ¡nh cÃ¡c versions YOLOv8:**
| Model | Params | mAP@50-95 | Speed (ms) | Khi nÃ o dÃ¹ng? |
|-------|--------|-----------|------------|---------------|
| YOLOv8n | 3.2M | 37.3% | 1.5ms | Edge devices (Raspberry Pi), real-time |
| **YOLOv8s** | 11.2M | 44.9% | 2.5ms | **Balance tá»‘t nháº¥t (code dÃ¹ng)** |
| YOLOv8m | 25.9M | 50.2% | 4.5ms | Accuracy cao hÆ¡n, cÃ³ GPU tá»‘t |
| YOLOv8l | 43.7M | 52.9% | 6.5ms | Server-side, offline processing |
| YOLOv8x | 68.2M | 53.9% | 10ms | YÃªu cáº§u accuracy tá»‘i Ä‘a |

**Code training:**
```python
results_yolo = model_yolo.train(
    data=yaml_path,
    epochs=15,        # Ãt epoch vÃ¬ dÃ¹ng pretrained weights
    imgsz=256,        # áº¢nh nhá» (256Ã—256) Ä‘á»ƒ train nhanh, tiáº¿t kiá»‡m memory
    batch=16          # Batch size vá»«a pháº£i cho GPU Colab T4
)
```

**Táº¡i sao chá»‰ 15 epochs?**
- Pretrained model Ä‘Ã£ há»c 80% knowledge tá»« COCO
- Chá»‰ cáº§n fine-tune Ä‘á»ƒ adapt vá»›i tomato disease patterns
- 15 epochs Ä‘á»§ Ä‘á»ƒ loss convergence (xem training curves)

---

#### **Faster R-CNN trong code**
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
```

**Sá»‘ lá»›p sá»­ dá»¥ng: ~175 layers**

**Táº¡i sao chá»n ResNet-50-FPN?**

| LÃ½ do | Giáº£i thÃ­ch |
|-------|------------|
| **ResNet-50 lÃ  backbone chuáº©n** | Äá»§ sÃ¢u (50 layers) mÃ  khÃ´ng quÃ¡ náº·ng nhÆ° ResNet-101 |
| **FPN tÃ­ch há»£p sáºµn** | Detect objects á»Ÿ nhiá»u scales (bá»‡nh nhá» láº«n lá»›n trÃªn lÃ¡) |
| **Pretrained trÃªn COCO** | 80 classes object detection â†’ Ä‘Ã£ biáº¿t detect "patterns" chung |
| **Fine-tune dá»… dÃ ng** | Chá»‰ thay classifier head, giá»¯ nguyÃªn backbone |

**Code thay Ä‘á»•i head:**
```python
def get_rcnn_model(num_classes):
    # Load pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Láº¥y input features cá»§a classifier hiá»‡n táº¡i
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Thay chá»‰ pháº§n classifier head cho 11 classes (10 diseases + 1 background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

num_classes = 11  # 10 bá»‡nh + 1 background (class 0)
```

**Táº¡i sao giá»¯ nguyÃªn backbone?**
- ResNet-50 Ä‘Ã£ há»c **low-level features** (edges, textures, colors)
- Features nÃ y **universal** â†’ Ã¡p dá»¥ng Ä‘Æ°á»£c cho má»i domain
- Chá»‰ cáº§n train láº¡i **classifier head** Ä‘á»ƒ há»c mapping features â†’ disease classes

**So sÃ¡nh training configs:**
| Aspect | YOLOv8s | Faster R-CNN |
|--------|---------|--------------|
| **Layers** | 168 | 175 |
| **Pretrained** | COCO (80 classes) | COCO (80 classes) |
| **Modified parts** | Auto-adjusted head | Manual replace classifier |
| **Epochs** | 15 | 10 |
| **Batch size** | 16 | 8 |
| **Image size** | 256Ã—256 | 256Ã—256 |
| **Optimizer** | AdamW (auto) | SGD (manual) |
| **Training time** | ~30 mins (Colab T4) | ~45 mins (Colab T4) |

---

### Q3: Táº¡i sao YOLOv8 chá»n v8 mÃ  khÃ´ng dÃ¹ng version khÃ¡c?

#### **Evolution cá»§a YOLO (2015-2024)**

```
YOLOv1 (2015) â†’ YOLOv2 (2016) â†’ YOLOv3 (2018) â†’ YOLOv4 (2020)
  â†“                                                    â†“
YOLOv5 (2020) â† Ultralytics                    YOLOv7 (2022)
  â†“
YOLOv8 (2023) â† Code sá»­ dá»¥ng
  â†“
YOLOv9 (2024) â†’ YOLOv10 (2024) â†’ YOLOv11 (2024)
```

#### **5 lÃ½ do chá»n YOLOv8:**

**1. Anchor-Free Architecture (Quan trá»ng nháº¥t!)**
```
YOLOv5/v7 (Anchor-based):
- Cáº§n define trÆ°á»›c 9 anchor boxes (3 scales Ã— 3 ratios)
- Hyperparameters phá»©c táº¡p: anchor_t, anchor_scale
- Pháº£i tune anchors cho tá»«ng dataset
- Prediction: (objectness, Î”x, Î”y, Î”w, Î”h, class)

YOLOv8 (Anchor-free):
- KhÃ´ng cáº§n anchor boxes
- Dá»± Ä‘oÃ¡n trá»±c tiáº¿p (x, y, w, h, class)
- ÄÆ¡n giáº£n hÆ¡n, Ã­t hyperparameters
- Flexible cho má»i object shapes
```

**Code comparison:**
```python
# YOLOv5 cáº§n config anchors trong yaml:
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119] # P4/16
  - [116,90, 156,198, 373,326] # P5/32

# YOLOv8 khÃ´ng cáº§n gÃ¬ cáº£!
# Chá»‰ cáº§n:
model = YOLO('yolov8s.pt')
results = model.train(data='tomato.yaml', epochs=15)
```

**2. C2f Block thay vÃ¬ C3 (YOLOv5)**

| Aspect | YOLOv5 C3 | YOLOv8 C2f |
|--------|-----------|------------|
| **Conv layers** | 3 convolutions | 2 convolutions |
| **Computation** | Cháº­m hÆ¡n ~20% | **Nhanh hÆ¡n ~20%** |
| **Gradient flow** | Tá»‘t | **Tá»‘t hÆ¡n** |
| **Parameters** | Nhiá»u hÆ¡n | Ãt hÆ¡n |
| **Accuracy** | High | High (tÆ°Æ¡ng Ä‘Æ°Æ¡ng) |

**Architecture comparison:**
```
C3 (YOLOv5):
Input â†’ Conv1 â†’ Conv2 â†’ Conv3 â†’ Concat â†’ Conv4 â†’ Output

C2f (YOLOv8):
Input â†’ Conv1 â†’ Conv2 â†’ Concat â†’ Conv3 â†’ Output
        â†“        â†“
        Split â†’ Multiple Bottlenecks (parallel)
```

**3. Improved Loss Function**

```
YOLOv5 Loss:
- Box Loss: CIoU (Complete IoU)
- Cls Loss: BCE (Binary Cross Entropy)
- Obj Loss: BCE

YOLOv8 Loss:
- Box Loss: CIoU + DFL (Distribution Focal Loss)
  â†‘ DFL giÃºp dá»± Ä‘oÃ¡n box regression chÃ­nh xÃ¡c hÆ¡n
- Cls Loss: BCE with improved weighting
- Obj Loss: Integrated into classification (khÃ´ng tÃ¡ch riÃªng)
```

**DFL (Distribution Focal Loss) - Äiá»ƒm má»›i:**
- Thay vÃ¬ predict 4 giÃ¡ trá»‹ (x, y, w, h) trá»±c tiáº¿p
- Predict **distribution** cá»§a má»—i coordinate
- Model learn uncertainty â†’ confident hÆ¡n khi predict boxes
- **Káº¿t quáº£**: Localization accuracy tÄƒng ~2-3% mAP

**4. Decoupled Head (TÃ¡ch rá»i Classification vÃ  Box)**

```
YOLOv5 Coupled Head:
Feature Map â†’ Shared Conv Layers â†’ Split
                                      â†“
                                  [Box | Cls | Obj]
Problem: Box regression vÃ  classification conflict khi optimize

YOLOv8 Decoupled Head:
Feature Map â†’ Box Branch â†’ Conv layers â†’ Box predictions
            â†“
            â†’ Cls Branch â†’ Conv layers â†’ Class predictions

Benefit: 
- Box vÃ  Cls há»c Ä‘á»™c láº­p
- Convergence nhanh hÆ¡n
- Accuracy tÄƒng ~1-2% mAP
```

**5. API Ä‘Æ¡n giáº£n hÆ¡n (Developer Experience)**

```python
# YOLOv5 (phá»©c táº¡p):
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

# YOLOv8 (Ä‘Æ¡n giáº£n):
pip install ultralytics

from ultralytics import YOLO

# Training
model = YOLO('yolov8s.pt')
results = model.train(data='tomato.yaml', epochs=15)

# Inference
results = model.predict('test.jpg')
results[0].plot()
```

#### **Táº¡i sao KHÃ”NG dÃ¹ng YOLOv9/v10/v11?**

**YOLOv9 (2024)**
- **Æ¯u Ä‘iá»ƒm**: 
  - PGI (Programmable Gradient Information) â†’ tÄƒng accuracy ~5%
  - GELAN (Generalized ELAN) â†’ efficient architecture
  - mAP@50-95: ~52% (cao hÆ¡n YOLOv8 ~7%)
- **NhÆ°á»£c Ä‘iá»ƒm**:
  - âŒ **Náº·ng hÆ¡n nhiá»u** (50M+ parameters vs 11M cá»§a YOLOv8s)
  - âŒ **Cháº­m hÆ¡n** (~80 FPS vs 200+ FPS)
  - âŒ **YÃªu cáº§u GPU máº¡nh** (V100/A100, khÃ´ng phÃ¹ há»£p Colab T4)
  - âŒ **ChÆ°a stable** (Ã­t tutorials, community nhá»)

**YOLOv10 (2024)**
- **Æ¯u Ä‘iá»ƒm**:
  - NMS-free (khÃ´ng cáº§n Non-Maximum Suppression) â†’ tÄƒng tá»‘c inference
  - Dual label assignments â†’ há»c tá»‘t hÆ¡n
- **NhÆ°á»£c Ä‘iá»ƒm**:
  - âŒ **Má»›i ra** (May 2024) â†’ Ã­t documentation
  - âŒ **Breaking changes** API â†’ code cÅ© khÃ´ng tÆ°Æ¡ng thÃ­ch
  - âŒ **ChÆ°a proven** trong production

**YOLOv11 (2024)**
- **Æ¯u Ä‘iá»ƒm**:
  - Latest version (Oct 2024)
  - C3k2 blocks â†’ efficient hÆ¡n
- **NhÆ°á»£c Ä‘iá»ƒm**:
  - âŒ **QuÃ¡ má»›i** â†’ chÆ°a cÃ³ benchmark Ä‘áº§y Ä‘á»§
  - âŒ **API changes** â†’ migration cost cao
  - âŒ **Risk** cho production use

#### **Benchmark so sÃ¡nh (COCO dataset):**

| Version | Year | mAP@50-95 | FPS (V100) | Params | Maturity |
|---------|------|-----------|------------|--------|----------|
| YOLOv5s | 2020 | 37.4% | 140 | 7.2M | â­â­â­â­â­ Mature |
| YOLOv7 | 2022 | 51.2% | 120 | 37M | â­â­â­â­ Stable |
| **YOLOv8s** | 2023 | **44.9%** | **200+** | **11.2M** | â­â­â­â­â­ **Stable** |
| YOLOv9 | 2024 | 52.0% | 80 | 50M+ | â­â­â­ New |
| YOLOv10 | 2024 | 53.0% | 100 | 30M | â­â­ Very New |
| YOLOv11 | 2024 | 54.0% | 90 | 28M | â­ Just Released |

**Káº¿t luáº­n**: YOLOv8s lÃ  **sweet spot** cho project nÃ y:
- âœ… **Balance** tá»‘t nháº¥t: Speed (200 FPS) + Accuracy (mAP ~45%)
- âœ… **Stable**: 1+ year proven in production
- âœ… **Colab-friendly**: Train Ä‘Æ°á»£c vá»›i T4 GPU (16GB VRAM)
- âœ… **Rich ecosystem**: Tutorials, community support, export options (ONNX, TensorRT, CoreML)
- âœ… **Anchor-free**: Dá»… train, Ã­t hyperparameters

**Khi nÃ o upgrade lÃªn YOLOv9+?**
- Khi cÃ³ GPU máº¡nh (A100, H100)
- Khi accuracy quan trá»ng hÆ¡n speed
- Khi YOLOv9+ Ä‘Ã£ mature (6+ months sau release)

---

### Q4: Faster R-CNN cÃ³ dÃ¹ng tÃ­ch cháº­p khÃ´ng? DÃ¹ng nhÆ° tháº¿ nÃ o?

**Tráº£ lá»i: CÃ“! Faster R-CNN sá»­ dá»¥ng convolution Ráº¤T NHIá»€U.**

ToÃ n bá»™ kiáº¿n trÃºc Faster R-CNN xoay quanh **tÃ­ch cháº­p 2D (2D Convolution)** - Ä‘Ã¢y lÃ  "xÆ°Æ¡ng sá»‘ng" cá»§a má»i CNN.

#### **4 NÆ¡i Sá»­ Dá»¥ng TÃ­ch Cháº­p:**

---

#### **A. ResNet-50 Backbone (Convolution Core)**

**Cáº¥u trÃºc chi tiáº¿t vá»›i sá»‘ lÆ°á»£ng conv:**

```
Input Image: (3, 256, 256)
    â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ CONV1 (Entry Block)                         â”‚
â”‚ - 7Ã—7 Conv, 64 filters, stride=2, padding=3 â”‚ â† Conv thá»© 1
â”‚ - BatchNorm + ReLU                          â”‚
â”‚ - 3Ã—3 MaxPool, stride=2                     â”‚
â”‚ Output: (64, 64, 64)                        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ LAYER 1 (3 Bottleneck Blocks)                â”‚
â”‚                                               â”‚
â”‚ Block 1:                                      â”‚
â”‚   1Ã—1 Conv (64â†’64)   â† Conv 2                â”‚
â”‚   3Ã—3 Conv (64â†’64)   â† Conv 3                â”‚
â”‚   1Ã—1 Conv (64â†’256)  â† Conv 4                â”‚
â”‚   Skip: 1Ã—1 Conv (3â†’256) â† Conv 5 (downsample) â”‚
â”‚                                               â”‚
â”‚ Block 2, 3: TÆ°Æ¡ng tá»± (má»—i block 3 conv)      â”‚
â”‚ Tá»•ng: 3 blocks Ã— 3 = 9 conv + 1 downsample   â”‚
â”‚       = 10 conv layers                        â”‚
â”‚ Output: (256, 64, 64)                         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ LAYER 2 (4 Bottleneck Blocks)                â”‚
â”‚ Tá»•ng: 4 Ã— 3 = 12 conv + 1 downsample         â”‚
â”‚       = 13 conv layers                        â”‚
â”‚ Output: (512, 32, 32)                         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ LAYER 3 (6 Bottleneck Blocks)                â”‚
â”‚ Tá»•ng: 6 Ã— 3 = 18 conv + 1 downsample         â”‚
â”‚       = 19 conv layers                        â”‚
â”‚ Output: (1024, 16, 16)                        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ LAYER 4 (3 Bottleneck Blocks)                â”‚
â”‚ Tá»•ng: 3 Ã— 3 = 9 conv + 1 downsample          â”‚
â”‚       = 10 conv layers                        â”‚
â”‚ Output: (2048, 8, 8)                          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Tá»”NG CONV TRONG RESNET-50 BACKBONE:
1 (Conv1) + 10 (Layer1) + 13 (Layer2) + 19 (Layer3) + 10 (Layer4)
= 53 convolutional layers
```

**Chi tiáº¿t 1 Residual Block (Bottleneck):**

```python
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Conv 1: 1Ã—1 conv giáº£m channels (dimensionality reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        # Conv 2: 3Ã—3 conv há»c features
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, 
                                kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        
        # Conv 3: 1Ã—1 conv tÄƒng channels (dimensionality expansion)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        
        # Main path (3 convolutions)
        out = F.relu(self.bn1(self.conv1(x)))      # 1Ã—1 conv
        out = F.relu(self.bn2(self.conv2(out)))    # 3Ã—3 conv
        out = self.bn3(self.conv3(out))            # 1Ã—1 conv
        
        # Skip connection
        if self.downsample:
            identity = self.downsample(x)           # 1Ã—1 conv (náº¿u cáº§n)
        
        out += identity  # Element-wise addition
        out = F.relu(out)
        return out
```

**CÃ´ng thá»©c toÃ¡n há»c tÃ­ch cháº­p 2D:**

```
Output[i, j] = Î£ Î£ Î£ Input[c, i+m, j+n] Ã— Kernel[c, m, n] + Bias
               c m n

Trong Ä‘Ã³:
- c: channel index (duyá»‡t táº¥t cáº£ input channels)
- m, n: kernel spatial dimensions (vÃ­ dá»¥: 0..2 cho 3Ã—3 kernel)
- i, j: output spatial position
```

**VÃ­ dá»¥ cá»¥ thá»ƒ vá»›i 3Ã—3 Conv:**

```
Input: (256, 64, 64)  # 256 channels, 64Ã—64 spatial
Kernel: (512, 256, 3, 3)  # 512 output channels, 256 input channels, 3Ã—3 kernel
Bias: (512,)

Computation:
- Vá»›i má»—i output channel k (0..511):
  - Vá»›i má»—i position (i, j):
    - Láº¥y 3Ã—3 patch tá»« táº¥t cáº£ 256 input channels
    - NhÃ¢n element-wise vá»›i kernel[k]
    - Sum táº¥t cáº£: Î£(256 Ã— 3 Ã— 3) = 2304 multiplications
    - Add bias[k]
    
Total operations cho 1 conv layer:
512 (output channels) Ã— 64Ã—64 (spatial) Ã— 256Ã—3Ã—3 (kernel ops)
= ~604 million multiplications
```

---

#### **B. FPN (Feature Pyramid Network) - Convolution Fusion**

**3 loáº¡i convolution trong FPN:**

**1. Top-down pathway (Giáº£m channels):**

```python
# Giáº£m channels tá»« 2048 â†’ 256
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

**Táº¡i sao dÃ¹ng 1Ã—1 Conv?**
- **Dimensionality reduction**: Giáº£m channels mÃ  khÃ´ng thay Ä‘á»•i spatial size
- **Computation efficiency**: 1Ã—1 conv ráº» hÆ¡n 3Ã—3 ráº¥t nhiá»u
- **Feature transformation**: Learn linear combinations cá»§a input channels

**2. Lateral connections (Káº¿t há»£p features):**

```python
# Upsample tá»« táº§ng sÃ¢u lÃªn
upsampled = F.interpolate(fpn_c5, scale_factor=2, mode='nearest')
# (256, 8, 8) â†’ (256, 16, 16)

# Element-wise addition vá»›i lateral connection
fpn_p4 = upsampled + fpn_conv_c4(c4)
# (256, 16, 16) + (256, 16, 16) = (256, 16, 16)
```

**CÆ¡ cháº¿ hoáº¡t Ä‘á»™ng:**
```
C5 (Deep, Semantic)     C4 (Mid-level)
(2048, 8, 8)            (1024, 16, 16)
    â†“ 1Ã—1 Conv              â†“ 1Ã—1 Conv
(256, 8, 8)             (256, 16, 16)
    â†“ Upsample Ã—2
(256, 16, 16) --------â†’ + --------â†’ P4 (256, 16, 16)
                        Add
```

**3. Smooth layers (3Ã—3 Conv Ä‘á»ƒ giáº£m aliasing):**

```python
# Sau khi merge, dÃ¹ng 3Ã—3 conv Ä‘á»ƒ "smooth" features
self.fpn_smooth_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.fpn_smooth_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.fpn_smooth_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

# Apply
p4_final = self.fpn_smooth_p4(fpn_p4)
# Input: (256, 16, 16)
# Output: (256, 16, 16)  # Same size, but refined features
```

**Táº¡i sao cáº§n smooth conv?**
- Upsample (nearest/bilinear) gÃ¢y **aliasing** (rÄƒng cÆ°a)
- 3Ã—3 conv lÃ m má»‹n boundaries
- TÄƒng receptive field

**FPN Complete Flow:**

```
ResNet Backbone Output:
C2: (256, 64, 64)
C3: (512, 32, 32)
C4: (1024, 16, 16)
C5: (2048, 8, 8)

                                    â”Œâ”€ P2 (256, 64, 64)
                                    â”‚
Top-Down:                           â”œâ”€ P3 (256, 32, 32)
C5 â†’ 1Ã—1Conv â†’ P5 (256, 8, 8)      â”‚
     â†“ Upsample                    â”œâ”€ P4 (256, 16, 16)
C4 â†’ 1Ã—1Conv â†’ + â†’ 3Ã—3Conv â†’ P4    â”‚
     â†“ Upsample                    â”œâ”€ P5 (256, 8, 8)
C3 â†’ 1Ã—1Conv â†’ + â†’ 3Ã—3Conv â†’ P3    â”‚
     â†“ Upsample                    â””â”€ P6 (256, 4, 4) [tá»« P5 vá»›i stride=2]
C2 â†’ 1Ã—1Conv â†’ + â†’ 3Ã—3Conv â†’ P2

Total Conv layers trong FPN:
- 1Ã—1 Conv: 4 layers (C2, C3, C4, C5 â†’ 256 channels)
- 3Ã—3 Conv: 4 layers (smooth P2, P3, P4, P5)
= 8 conv layers
```

---

#### **C. RPN (Region Proposal Network) - Convolution Sliding Window**

**RPN lÃ  "trÃ¡i tim" cá»§a Faster R-CNN, sá»­ dá»¥ng 3 conv layers:**

**1. Sliding Window Convolution:**

```python
class RPNHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9):
        # Intermediate conv (sliding window)
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        # Input: (256, H, W)  # Tá»« FPN
        # Output: (512, H, W)  # Same spatial size
        
    def forward(self, x):
        # Slide 3Ã—3 window across feature map
        x = F.relu(self.conv(x))
        return x
```

**CÆ¡ cháº¿ Sliding Window:**
```
Feature Map: (256, 40, 40)  # Tá»« FPN P4
    â†“
3Ã—3 Conv kernel slides:
â”Œâ”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”
â”‚ â—‰ â—‰ â—‰â”‚     â”‚     â”‚     â”‚  â† Position (0,0)
â”‚ â—‰ â—‰ â—‰â”‚     â”‚     â”‚     â”‚
â”‚ â—‰ â—‰ â—‰â”‚     â”‚     â”‚     â”‚
â”œâ”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”¤
â”‚     â”‚ â—‰ â—‰ â—‰â”‚     â”‚     â”‚  â† Position (0,1)
â”‚     â”‚ â—‰ â—‰ â—‰â”‚     â”‚     â”‚
â”‚     â”‚ â—‰ â—‰ â—‰â”‚     â”‚     â”‚
â””â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”˜

Táº¡i má»—i position (i, j):
- Extract 3Ã—3Ã—256 patch
- Convolve vá»›i kernel (512, 256, 3, 3)
- Produce 512-dim feature vector
- Repeat cho táº¥t cáº£ 40Ã—40 = 1600 positions
```

**2. Classification Branch (Objectness):**

```python
# Predict objectness scores (object vs background)
self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
# Input: (512, H, W)
# Output: (18, H, W)  # 9 anchors Ã— 2 classes

# Reshape output:
# (B, 18, H, W) â†’ (B, H, W, 9, 2) â†’ (B, HÃ—WÃ—9, 2)
```

**Anchors:**
```
Má»—i position trÃªn feature map cÃ³ 9 anchors:
- 3 scales: 128Â², 256Â², 512Â² pixels
- 3 aspect ratios: 1:1, 1:2, 2:1

VÃ­ dá»¥ position (10, 10) trÃªn feature map 40Ã—40:
Anchor 1: (256Ã—1, 256Ã—1)   # Square 256
Anchor 2: (256Ã—0.5, 256Ã—2) # Tall 128Ã—512
Anchor 3: (256Ã—2, 256Ã—0.5) # Wide 512Ã—128
Anchor 4: (512Ã—1, 512Ã—1)   # Square 512
...
Anchor 9: (128Ã—2, 128Ã—0.5) # Wide 256Ã—64

Total anchors: 40 Ã— 40 Ã— 9 = 14,400 anchors
```

**3. Box Regression Branch:**

```python
# Predict bounding box deltas
self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
# Input: (512, H, W)
# Output: (36, H, W)  # 9 anchors Ã— 4 coords (Î”x, Î”y, Î”w, Î”h)
```

**Box Regression Formula:**
```python
def apply_box_deltas(anchors, deltas):
    # anchors: (N, 4) [x, y, w, h]
    # deltas: (N, 4) [Î”x, Î”y, Î”w, Î”h]
    
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
    â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 3Ã—3 Conv (256 â†’ 512)             â”‚ â† Sliding window
â”‚ + ReLU                           â”‚
â”‚ Output: (512, H, W)              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
             â”‚
      â”Œâ”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”
      â”‚             â”‚
      â†“             â†“
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ 1Ã—1 Convâ”‚   â”‚ 1Ã—1 Convâ”‚
â”‚ (512â†’18)â”‚   â”‚ (512â†’36)â”‚
â”‚         â”‚   â”‚         â”‚
â”‚  Class  â”‚   â”‚   Box   â”‚
â”‚  (18,   â”‚   â”‚  (36,   â”‚
â”‚   H, W) â”‚   â”‚   H, W) â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
     â”‚             â”‚
     â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜
            â†“
    Apply NMS + Filter
            â†“
    Top-N Proposals (~1000)
```

**Total Conv trong RPN: 3 layers**
- 1 sliding window conv (3Ã—3)
- 1 classification conv (1Ã—1)
- 1 box regression conv (1Ã—1)

---

#### **D. RoI Head - FC Layers (KHÃ”NG PHáº¢I Convolution)**

**Quan trá»ng: RoI Head trong code Sá»¬ Dá»¤NG FC LAYERS, khÃ´ng pháº£i conv!**

```python
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        # RoI Pooled features: (in_channels, 7, 7)
        # Flatten: 7Ã—7Ã—in_channels = 12544 dims (náº¿u in_channels=256)
        
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
        # Output: (44,)  # 11 classes Ã— 4 coords
    
    def forward(self, x):
        # x: (num_proposals, 256, 7, 7)
        x = x.flatten(start_dim=1)  # (num_proposals, 12544)
        
        x = F.relu(self.fc1(x))     # (num_proposals, 1024)
        x = F.relu(self.fc2(x))     # (num_proposals, 1024)
        
        cls_scores = self.cls_score(x)    # (num_proposals, 11)
        bbox_deltas = self.bbox_pred(x)   # (num_proposals, 44)
        
        return cls_scores, bbox_deltas
```

**Táº¡i sao dÃ¹ng FC thay vÃ¬ Conv?**
- RoI features Ä‘Ã£ Ä‘Æ°á»£c **RoI Align** pool vá» fixed size (7Ã—7)
- Cáº§n **global context** Ä‘á»ƒ classify (khÃ´ng pháº£i local patterns)
- FC layers learn **holistic representations**

**Má»™t sá»‘ variants dÃ¹ng Conv:**
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

**NhÆ°ng trong code `torchvision.models.detection.fasterrcnn_resnet50_fpn`:**
- **Default sá»­ dá»¥ng FC layers** (khÃ´ng pháº£i conv)
- LÃ½ do: Faster, simpler, proven effective

---

### **Tá»”NG Káº¾T: Convolution trong Faster R-CNN**

| Component | Conv Layers | Kernel Sizes | Chá»©c nÄƒng |
|-----------|-------------|--------------|-----------|
| **ResNet-50 Backbone** | 53 layers | 1Ã—1, 3Ã—3, 7Ã—7 | TrÃ­ch xuáº¥t features cÆ¡ báº£n |
| **FPN** | 8 layers | 1Ã—1 (4), 3Ã—3 (4) | Multi-scale feature fusion |
| **RPN** | 3 layers | 3Ã—3 (1), 1Ã—1 (2) | Generate proposals |
| **RoI Head** | 0 layers | N/A | Sá»­ dá»¥ng FC layers |

**Total Convolution Layers: 53 + 8 + 3 = 64 conv layers**

**CÃ´ng thá»©c Convolution 2D (Recap):**

```
Output[b, c_out, h, w] = Î£ Î£ Î£ (
                         c_in m n
    Input[b, c_in, hÃ—stride + m, wÃ—stride + n] Ã— 
    Weight[c_out, c_in, m, n]
) + Bias[c_out]

Trong Ä‘Ã³:
- b: batch index
- c_out: output channel index (0..num_filters-1)
- c_in: input channel index (0..in_channels-1)
- m, n: kernel spatial dimensions (0..kernel_size-1)
- h, w: output spatial position
- stride: convolution stride (thÆ°á»ng 1 hoáº·c 2)
```

**VÃ­ dá»¥ cá»¥ thá»ƒ:**

```python
import torch
import torch.nn as nn

# Define a 3Ã—3 conv layer
conv = nn.Conv2d(
    in_channels=256,      # Input depth
    out_channels=512,     # Number of filters
    kernel_size=3,        # 3Ã—3 kernel
    stride=1,             # Slide 1 pixel at a time
    padding=1,            # Pad 1 pixel to maintain size
    bias=True             # Include bias term
)

# Input tensor
x = torch.randn(8, 256, 40, 40)  # (batch, channels, height, width)

# Forward pass
output = conv(x)  # (8, 512, 40, 40)

# Number of parameters:
# Weights: 512 Ã— 256 Ã— 3 Ã— 3 = 1,179,648
# Bias: 512
# Total: 1,180,160 parameters
```

**Computation (FLOPs):**
```
FLOPs = 2 Ã— Kernel_Size Ã— Kernel_Size Ã— In_Channels Ã— Out_Channels Ã— Output_Height Ã— Output_Width

VÃ­ dá»¥ trÃªn:
FLOPs = 2 Ã— 3 Ã— 3 Ã— 256 Ã— 512 Ã— 40 Ã— 40
      = 2 Ã— 9 Ã— 256 Ã— 512 Ã— 1600
      = ~3.8 billion operations
```

**Äiá»ƒm khÃ¡c biá»‡t YOLOv8 vs Faster R-CNN:**

| Aspect | YOLOv8 | Faster R-CNN |
|--------|--------|--------------|
| **Convolution usage** | 100% conv (end-to-end) | Conv backbone + RPN, FC head |
| **Detection head** | Conv layers | **FC layers** |
| **Speed** | Faster (no FC bottleneck) | Slower (FC + RoI Align) |
| **Parallelization** | Full GPU parallelization | Sequential (RPN â†’ RoI â†’ FC) |

**Káº¿t luáº­n:**
- âœ… Faster R-CNN **Sá»¬ Dá»¤NG CONVOLUTION Ráº¤T NHIá»€U** (64 conv layers)
- âœ… Convolution xuáº¥t hiá»‡n á»Ÿ: **Backbone, FPN, RPN**
- âœ… RoI Head dÃ¹ng **FC layers** (khÃ´ng pháº£i conv) trong implementation chuáº©n
- âœ… Conv lÃ  "xÆ°Æ¡ng sá»‘ng" Ä‘á»ƒ extract features, FCs Ä‘á»ƒ classify/regress

**LÆ°u Ã½ váº­n hÃ nh / debug**
- Náº¿u training bá»‹ lá»—i do OOM (Out of Memory): giáº£m `batch` hoáº·c `imgsz` hoáº·c dÃ¹ng GPU lá»›n hÆ¡n.
- Náº¿u `results.csv` khÃ´ng sinh: cÃ³ thá»ƒ quÃ¡ trÃ¬nh train dá»«ng sá»›m do lá»—i; kiá»ƒm tra logs trong cell huáº¥n luyá»‡n.
- Kiá»ƒm tra `CLASS_NAMES` khá»›p vá»›i nhÃ£n; lá»—i mismatch dáº«n tá»›i Ä‘Ã¡nh giÃ¡ sai hoáº·c IndexError.
- Khi chuyá»ƒn nhÃ£n YOLO cho Faster R-CNN, nhá»› +1 cho label index vÃ¬ PyTorch detection dÃ¹ng 0 lÃ  background.

**HÆ°á»›ng dáº«n nhanh Ä‘á»ƒ cháº¡y (Colab)**
1. Mount Drive (Ä‘Ã£ cÃ³ trong notebook).  
2. Chá»‰nh `ROOT_DIR` trá» Ä‘áº¿n thÆ° má»¥c dataset trong Drive.  
3. Cháº¡y tuáº§n tá»± cÃ¡c cell: cÃ i Ä‘áº·t -> chuáº©n bá»‹ dá»¯ liá»‡u (`prepare_data()`), -> ghi yaml (`write_yolo_yaml()`), -> phÃ¢n tÃ­ch phÃ¢n bá»‘ (`analyze_distribution`) -> huáº¥n luyá»‡n YOLO -> Ä‘Ã¡nh giÃ¡ -> chuáº©n bá»‹ dataset RCNN -> huáº¥n luyá»‡n RCNN -> Ä‘Ã¡nh giÃ¡ vÃ  so sÃ¡nh.

**File Ä‘Ã£ táº¡o**
- `d:\colab\README.md` (báº£n báº¡n Ä‘ang Ä‘á»c).

Náº¿u báº¡n muá»‘n, tÃ´i cÃ³ thá»ƒ:
- Dá»‹ch toÃ n bá»™ README sang tiáº¿ng Anh.
- Táº¡o file `requirements.txt` dá»±a trÃªn cÃ¡c import trong notebook.
- TÃ¡ch README thÃ nh tá»«ng pháº§n nhá» hÆ¡n (má»—i pháº§n má»™t file) hoáº·c thÃªm vÃ­ dá»¥ cháº¡y nhanh (script .py) Ä‘á»ƒ cháº¡y local.

---
TÃ´i Ä‘Ã£ táº¡o file `d:\colab\README.md` chá»©a giáº£i thÃ­ch chi tiáº¿t. Muá»‘n chá»‰nh ngÃ´n ngá»¯ hay bá»• sung thÃªm pháº§n nÃ o khÃ´ng?#   t o m a t o _ m o d e l 
 
 