# 🚗 YOLOMine Object Detection Model

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

