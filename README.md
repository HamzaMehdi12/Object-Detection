# 🚗 YOLOMine Object Detection Model

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-Research/Education-yellow)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![Stars](https://img.shields.io/github/stars/YourUsername/YOLOMine?style=social)
![Forks](https://img.shields.io/github/forks/YourUsername/YOLOMine?style=social)

A custom YOLO-based object detection model implementation in **PyTorch** for **vehicle damage detection**, featuring a **DarkNet-style backbone**, **FPN neck**, and **multi-scale detection head**.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Dataset Format](#-dataset-format)
- [Training](#-training)
- [Validation](#-validation)
- [Inference](#-inference)
- [Output](#-output)
- [Model Performance](#-model-performance)
- [Key Components](#️-key-components)
- [Customization](#-customization)
- [Notes](#-notes)
- [Visual Examples](#-visual-examples)
- [Author](#-author)

---

## 🔍 Overview

**YOLOMine** is a lightweight object detection model designed for detecting **vehicle damage** across five categories:

- Headlamp  
- Rear Bumper  
- Door  
- Hood  
- Front Bumper  

**Model Size:** ~19.89M parameters

---

## 🏗️ Model Architecture

### **Backbone (DarkNet-inspired)**
- **Conv Layers:** Progressive feature extraction with stride-2 convolutions  
- **C2f Blocks:** Cross Stage Partial (CSP) bottleneck modules  
- **SPPF:** Spatial Pyramid Pooling - Fast for multi-scale feature aggregation  
- **Output:** Three feature maps at different scales (P3, P4, P5)

### **Neck (FPN-PAN)**
- **Top-down pathway:** High-level semantic feature fusion  
- **Bottom-up pathway:** Path aggregation for localization improvement  
- **Multi-scale fusion:** Concatenation at each scale  

### **Head (Multi-scale Detection)**
- **Detection scales:** 8x, 16x, 32x strides  
- **DFL (Distribution Focal Loss):** For precise bounding box regression  
- **Separate branches:** Box coordinates (64 channels) + Class predictions (80 channels)

---

## ✨ Features

✅ COCO-format dataset support  
✅ Custom loss function with weighted components (box, objectness, classification)  
✅ Region highlighting with contour-based visualization  
✅ COCO evaluation metrics (mAP@0.5, mAP@0.75, etc.)  
✅ Training & validation loss tracking  
✅ Inference with confidence thresholding  
✅ Results saved as annotated images + text files  

---

## 📦 Installation

```bash
# Required dependencies
pip install torch torchvision opencv-python numpy matplotlib pillow pycocotools tqdm
```

---

## Directory Structure:
```
project/
├── main.py           # Training & inference script
├── model.py          # Model architecture (Backbone, Neck, Head)
├── loss.py           # Custom loss functions
├── dataset.py        # Dataset loader
├── data/
│   ├── train/
│   │   ├── COCO.json
│   │   └── images/
│   ├── val/
│   │   ├── COCO.json
│   │   └── images/
│   └── test/
│       └── images/
└── output/
    ├── inference/    # Test predictions
    └── coco_val/     # Validation results
```
---

## 📊 Dataset Format
### COCO JSON Example:
```
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [x, y, width, height]}
  ],
  "categories": [
    {"id": 0, "name": "headlamp"},
    {"id": 1, "name": "rear_bumper"}
  ]
}
```

---

## 🚀 Training
### Hyperparameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001

### Loss weights
lambda_box = 7.5   # Bounding box loss
lambda_obj = 1.0   # Objectness loss
lambda_cls = 0.5   # Classification loss

### Run training
python main.py

### Training Output:
- Progress bar with per-batch loss 
- Epoch-level average loss
- Real-time loss curve visualization

---

## ✅ Validation
### Includes:
- Validation loss computation
- COCO-style evaluation metrics:
- mAP@[.5:.95] (Primary metric)
- mAP@0.5
- mAP@0.75
- Recall
- Precision

---

## 🔮 Inference
### Test configuration
conf_threshold = 0.5  # Confidence threshold for detections

model.eval()
### Processes all images in data/test/

---

## 📤 Output
### Visual Output
1. Red contours around detected damage
2. Semi-transparent overlay (25% opacity)
3. Canny edge detection for precision

## 📈 Model Performance
### Training Metrics
- Device: CPU/CUDA (auto-detected)
- Parameters: 19,894,977 (~19.89M)
- Optimizer: AdamW
- Visualization: Real-time training curve

---

## 🛠️ Key Components
### Loss Function (MyLoss)

total_loss = λ_box * L_box + λ_obj * L_obj + λ_cls * L_cls
- Box loss: MSE on coordinates
- Objectness loss: BCE for object presence
- Class loss: BCE for multi-class prediction

### Detection Decoding
- Sigmoid activation for centers & confidences
- Convert center-based → corner format
- Confidence filtering

### Visualization Enhancement
- Gaussian blur + Canny edge detection
- Morphological dilation for contours
- Contour area filtering (>80 px)
- Alpha blending for transparency

---

## 📝 Notes
- Model uses normalized coordinates [0, 1] internally
- Supports variable input sizes (default 640×640)
- Trained on COCO classes, customized for 5 damage types
- Pure in-memory processing (no external APIs)

---

## 🔧 Customization
- To adapt for new classes:
- Update num_classes in model initialization
- Modify class_names in inference section
- Update COCO JSON categories
- Retrain the model

---

## 🖼️ Visual Examples

Below are sample detection results from YOLOMine showing vehicle damage localization:

<p align="center">
   <img width="1000" height="800" alt="test_1" src="https://github.com/user-attachments/assets/bc7681e7-a6ac-450f-b4ae-9469993e5349" />
</p>

<p align="center">
   <img width="1000" height="800" alt="test_2" src="https://github.com/user-attachments/assets/ccb7153f-1e6a-4449-86b8-c403647be234" />
</p>

<p align="center">
   <img width="1000" height="800" alt="test_3" src="https://github.com/user-attachments/assets/01f63ff9-b4bb-47fc-b47f-f8436cd9327d" />
</p>


<p align="center">
   <img width="1000" height="800" alt="train_losses" src="https://github.com/user-attachments/assets/35c6524e-f9e8-4c1a-bc20-d3d49c12f467" />
</p>

---

## 👤 Author

**Muhammad Hamza Mehdi** \
**Framework: PyTorch** \
**License: Research / Educational Use** \

---

⭐ **If you find this project helpful, please give it a star!**
