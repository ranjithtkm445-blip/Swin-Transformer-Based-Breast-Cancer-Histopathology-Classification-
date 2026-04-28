
 Breast Cancer Histopathology Classification with Explainable Swin Transformer

**Author:** Ranjith Kumar
**Live Demo:** [https://ranjith445-breakhis-swin.hf.space/](https://ranjith445-breakhis-swin.hf.space/)

---

 1. Problem (Simple Story)

Doctors diagnose breast cancer by examining tissue images under a microscope.
This process is:

* Time-consuming
* Requires high expertise
* Can vary slightly between doctors

At the same time, many AI models can predict results but **fail to explain why**, making them hard to trust.

The challenge is to build a system that:

* Can **classify tissue images**
* And also **show what it is looking at**

---

 2. Solution Overview

This project builds a **proof-of-concept AI system** using a **Swin Transformer** to classify histopathology images as **Benign** or **Malignant**, while also providing visual explanations.

 How the model makes decisions (relational understanding)

Instead of looking at the image as a whole, the model works by **relating different regions of the image**:

1. **Splits the image into small patches**
   Each patch represents a tiny part of the tissue

2. **Learns relationships between nearby patches**
   Checks if:

   * Cells are evenly spaced
   * Neighboring regions follow consistent patterns
   * Tissue structure is organized

   👉 Organized → Benign
   👉 Irregular → Malignant

3. **Connects neighboring regions (shifted windows)**
   Builds relationships across the image

4. **Builds a bigger picture step-by-step**
   Cells → tissue → overall structure

5. **Makes the final prediction**
   Based on how consistent or disrupted the tissue patterns are

---

 3. Project Scope (Important)

* This is a **demonstration project**
* Focus is on:

  * End-to-end pipeline
  * Interpretability (Attention + Grad-CAM)
  * Deployment

Trained on a small subset of 245 images

* Not optimized for clinical accuracy
* Not intended for medical use

---

4. Key Results

| Metric    | Score  | Insight                          |
| --------- | ------ | -------------------------------- |
| Accuracy  | 90.7%  | Good performance on demo dataset |
| F1 Score  | 92.86% | Balanced classification          |
| Precision | 100%   | No false positives               |
| Recall    | 86.67% | Some malignant cases missed      |

Results are based on a **limited dataset (trained on 245 images) and are for demonstration only.

---

## 🧬 5. Dataset

**BreakHis Dataset**

* Total images: 7,909
* Patients: 82

### Used in this project:

* Magnification: **40X**
* Filtered images: 1,850

### 📊 Demo Training Setup

| Split      | Images Used |
| ---------- | ----------- |
| Training   | **245**     |
| Validation | ~50         |
| Test       | ~50         |

👉 A reduced subset is used to keep the project lightweight and focused on pipeline demonstration.

Class Distribution:

* Benign: 625
* Malignant: 1225

---

 6. Model Architecture

* Backbone: Swin-Tiny (`swin_tiny_patch4_window7_224`)
* Pretrained on ImageNet
* Fine-tuned for binary classification

Custom Head

```id="5nkccp"
LayerNorm → Dropout → Linear → GELU → Linear
(768 → 128 → 2)
```
 Strategy

* Last 2 stages unfrozen
* Earlier layers frozen (for stability on small dataset)

---

7. Training Pipeline

### Data Processing

* Patient-level split
* Macenko stain normalization
* Augmentations:

  * Flip
  * Rotation
  * Color jitter

### Handling Imbalance

* Weighted sampling
* Class-weighted loss

### Training Setup

* Optimizer: AdamW
* Learning Rate: 1e-5
* Scheduler: Cosine Annealing
* Early stopping based on F1 score

---

 8. Explainability (Core Feature)

### 🔹 Attention Visualization

* Shows where the model is focusing

### 🔹 Grad-CAM

* Highlights regions influencing predictions

---

### 🔍 How to Interpret Heatmaps

* **Red/Yellow:** Important regions
* **Blue:** Less important
 Helps understand model focus

---

 Important Note

* These are **approximate explanations**
* With limited training data (**245 images**), highlighted regions may not always match true medical features

---
 9. How the Model Thinks

The **Swin Transformer**:

* Looks at small regions
* Compares nearby patterns
* Builds full understanding step-by-step

---

 10. Application Features

* Image upload
* Benign/Malignant prediction
* Attention heatmaps
* Grad-CAM visualization
* Confidence score

---

11. Tech Stack

* PyTorch
* timm
* Streamlit
* scikit-learn
* Matplotlib
* Docker
* Hugging Face Spaces

---

 12. Limitations

* Trained on **only 245 images**
* No external validation
* Grad-CAM is coarse
* Recall can be improved

---
 13. Future Improvements

* Grad-CAM++ / HiResCAM
* Larger dataset training
* Cross-validation
* Better recall
* Sharper visualizations

---

 14. Disclaimer

This project is for **educational and demonstration purposes only**.
Not intended for clinical use.

---
 Summary

> “I built a proof-of-concept explainable AI system using a Swin Transformer, trained on a small dataset of 245 images, focusing on learning relationships between tissue regions and providing visual explanations using attention maps and Grad-CAM.”

---

