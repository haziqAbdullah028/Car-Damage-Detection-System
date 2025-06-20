# Car Damage Detection using Mask R-CNN and Detectron2

This project implements a car damage detection system using **Mask R-CNN** with the **Detectron2** framework by Facebook AI. It performs both object detection and instance segmentation to identify and localize car damage areas (e.g., dents, scratches) from images.

## 📁 Dataset

* **Format**: COCO-style JSON annotations
* **Classes**: `damage` (1 class)
* **Total Images**:
  * Training: 59 images
  * Validation: 16 images
    
## ⚙️ Installation
* pip install detectron2
* pip install opencv-python matplotlib
* pip install numpy==1.26.4
* pip uninstall -y tensorboard && pip install tensorboard==2.18.0


## 🧠 Model Architecture
* **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
* **Model**: `mask_rcnn_R_50_FPN_3x`
* **Anchor Generator**: RPN (Region Proposal Network)
* **Head**: ROIAlign with FastRCNNConvFCHead and MaskHead

## Model Training
## 1. Register Datasets
from detectron2.data.datasets import register_coco_instances
register_coco_instances("car_damage_train", {}, "/content/car_damage_dataset/train/COCO_train_annos.json", "/content/car_damage_dataset/train")
register_coco_instances("car_damage_val", {}, "/content/car_damage_dataset/val/COCO_val_annos.json", "/content/car_damage_dataset/val")

## 2. Download Config Files
* !mkdir -p configs/COCO-InstanceSegmentation
* !wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml -O configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
* !wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/Base-RCNN-FPN.yaml -O configs/Base-RCNN-FPN.yaml
* !git clone https://github.com/facebookresearch/detectron2.git
  
## Download the output folder here:  
👉 [Google Drive Link](https://drive.google.com/drive/folders/1ejoy9Lw55NENUVjY8E39lMZ5crqr_nXu?usp=drive_link)

## 3.⚙️ Configuration Highlights
```python
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("car_damage_val", )
```
## 4.Training
```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("car_damage_train",)
cfg.DATASETS.TEST = ("car_damage_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.OUTPUT_DIR = "/content/output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.SOLVER.IMS_PER_BATCH = 2
.
.
.
.
.
```
## 📊 Evaluation Metrics (COCO)
```python
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

trainer.resume_or_load(resume=True)
evaluator = COCOEvaluator("car_damage_val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
val_loader = build_detection_test_loader(cfg, "car_damage_val")
.
.
.
```
### Bounding Box (bbox):

| Metric | Value |
| ------ | ----- |
| AP     | 9.997 |
| AP50   | 30.47 |
| AP75   | 5.13  |
| APs    | 0.00  |
| APm    | 9.77  |
| APl    | 12.73 |

### Segmentation (segm):

| Metric | Value |
| ------ | ----- |
| AP     | 4.70  |
| AP50   | 12.47 |
| AP75   | 4.95  |
| APs    | 0.00  |
| APm    | 6.51  |
| APl    | 3.26  |

## 🖼️ Prediction Visualization

* Predictions are visualized using Detectron2's `Visualizer`.
* Top 5 random images are displayed using Matplotlib.
* All validation predictions are saved to `/content/output/val_predictions/`.
```python
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
import cv2
import random

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("car_damage_val",)
predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get("car_damage_val")
dataset_dicts = DatasetCatalog.get("car_damage_val")

output_folder = "/content/output/val_predictions"
os.makedirs(output_folder, exist_ok=True)
.
.
.
.
```
## 📌 Requirements

* Python 3.8+
* Detectron2
* OpenCV
* PyTorch
* Matplotlib

## 📬 Results & Observations

* The model detects large damages better than small ones.
* Improvement possible by adding more labeled data or data augmentation.

## ✍️ Author

Haziq AbdullahEmail: haziqabdullah028@gmail.com

## 📧 Contact

For questions, reach out to the author via GitHub issues or email.

---

**Note**: This project is for research and educational purposes.

## 📓 Notebook

➡️ See full code in: Car_Damage_Detection.ipynb



