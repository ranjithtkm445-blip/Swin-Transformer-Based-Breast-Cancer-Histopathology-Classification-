

---

# **Breast Cancer Histopathology Classification with Spatial Attention Visualization**

**Author:** Ranjith Kumar
**Deployment:** HF Space – ranjith445-breakhis-swin.hf.space

---

## **Project Overview**

A **Swin Transformer-based deep learning system** for classifying breast histopathology images into **Benign** and **Malignant** categories.

The model integrates **spatial attention visualization**, enabling interpretability by highlighting tissue regions that influence predictions.

---

## **Key Results**

| Metric            | Score  | Notes                            |
| ----------------- | ------ | -------------------------------- |
| **Test Accuracy** | 90.7%  |                                  |
| **F1 Score**      | 92.86% | Strong class balance performance |
| **Precision**     | 100.0% | No false positives               |
| **Recall**        | 86.67% | Slight miss on malignant cases   |
| **Validation F1** | 92.5%  | Best at epoch 15                 |

---

## **Dataset**

* **Name:** BreakHis (Breast Cancer Histopathological Image Classification)
* **Total Images:** 7,909 (82 patients)
* **Magnifications:** 40X, 100X, 200X, 400X

### **Filtered Dataset (Used)**

* **Magnification:** 40X only
* **Total:** 1,850 images

**Class Distribution:**

* **Benign (625):** Adenosis (A), Fibroadenoma (F), Phyllodes Tumor (PT), Tubular Adenoma (TA)
* **Malignant (1225):** Ductal Carcinoma (DC), Lobular Carcinoma (MC), Mucinous Carcinoma

---

## **Model Architecture**

* **Backbone:** Swin-Tiny (`swin_tiny_patch4_window7_224`)
* **Pretraining:** ImageNet-1k (via timm)
* **Task:** Binary Classification (Benign vs Malignant)

### **Custom Classification Head**

LayerNorm → Dropout → Linear (768 → 128) → GELU → Linear (128 → 2)

### **Fine-tuning Strategy**

* Last **two Swin stages unfrozen**
* Earlier layers kept frozen for stability on small dataset

---

## **Methodology**

### **1. Data Pipeline**

* Patient-level train/val/test split (prevents leakage)
* WeightedRandomSampler (handles 1:2.2 imbalance)
* Macenko stain normalization (H&E consistency)
* Augmentations:

  * Random flip
  * Rotation
  * Color jitter

---

### **2. Training Configuration**

* **Optimizer:** AdamW (lr = 1e-5, weight_decay = 1e-4)
* **Scheduler:** CosineAnnealingLR
* **Loss:** CrossEntropy (with class weights)
* **Epochs:** 30 (early stopping at 15)
* **Early Stopping Metric:** Validation F1

---

### **3. Spatial Attention Visualization**

* Extracted from **final Swin stage (7×7 feature map)**
* Upsampled to **224×224 resolution**
* Overlayed as **heatmaps** on original images

**Purpose:**

* Improves interpretability
* Highlights tumor-relevant tissue regions
* Provides model reasoning insights

---

## **Why Swin Transformer over ViT**

| Feature              | ViT-Small   | Swin-Tiny                |
| -------------------- | ----------- | ------------------------ |
| Attention            | Global      | Local window-based       |
| Hierarchy            | Flat        | Multi-scale hierarchical |
| Patch Size           | 16×16       | 4×4 (finer detail)       |
| Data Requirement     | High (>10k) | Works on small datasets  |
| Tissue Understanding | Limited     | Cell-to-tissue hierarchy |
| F1 Score             | 68.35%      | **92.86%**               |

---

## **Application Features**

* Preloaded sample images (8 tumor subtypes)
* Attention heatmap overlay visualization
* Transformer interpretability insights
* Performance metrics display
* ViT vs Swin comparison module

---

## **Tech Stack**

* **PyTorch 2.0** – Training & inference
* **timm 0.9.12** – Model backbone
* **Streamlit** – Web interface
* **scikit-learn** – Metrics
* **Matplotlib** – Visualization
* **Docker** – Deployment
* **Hugging Face Spaces** – Hosting

---

## **Disclaimer**

This model is trained on a **small subset (245 images)** for demonstration purposes only.
It is **not intended for clinical use** and may not generalize to real-world medical settings.

---

