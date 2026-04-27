
Breast Cancer Histopathology Classification with Spatial Attention Visualization
Author: Ranjith Kumar | HF Space: ranjith445-breakhis-swin.hf.space
Project Description
Swin Transformer trained on 350 BreakHis histopathology images classifies breast tissue as Benign or Malignant with 92.86% F1 and 100% precision, using spatial attention maps to visualize tissue regions influencing each prediction.
Model
Architecture : Swin-Tiny (swin_tiny_patch4_window7_224)
Task : Binary classification — Benign vs Malignant
Subtypes : Benign (A, F, PT, TA) vs Malignant (DC, MC)
Magnification : 40X
Training : 245 images (subset of BreakHis)
Pretrained : ImageNet-1k via timm
Results
Metric | Score | Notes
Test Accuracy | 90.7% |
Test F1 Score | 92.86% |
Precision | 100.0% | Zero false positives
Recall | 86.67% |
Val F1 | 92.5% | Best epoch 15/30
Dataset
Name : BreakHis (Breast Cancer Histopathological Image Classification)
Total images : 7,909 microscopic images from 82 patients
Magnifications: 40X, 100X, 200X, 400X
Used subset : 40X only — 1,850 images after filtering
Benign : 625 images (Adenosis, Fibroadenoma, Phyllodes Tumor, Tubular Adenoma)
Malignant : 1,225 images (Ductal Carcinoma, Lobular Carcinoma, Mucinous Carcinoma)
Methodology
Data Pipeline
— Patient-level train/val/test split — prevents data leakage across splits
— WeightedRandomSampler — handles 1:2.2 benign/malignant class imbalance
— Macenko stain normalization — standardizes H&E color variation across slides
— Train augmentation: Random flip, rotation, color jitter
Model Architecture
— Backbone: Swin-Tiny pretrained on ImageNet-1k via timm
— Last 2 Swin stages unfrozen for fine-tuning on histopathology
— Custom head: LayerNorm -> Dropout -> Linear(768->128) -> GELU -> Linear(128->2)
— Spatial attention maps from last Swin stage (7x7 -> upsampled to 224x224)
Training
— Optimizer : AdamW (lr=1e-5, weight_decay=1e-4)
— Scheduler : CosineAnnealingLR
— Loss : CrossEntropyLoss with class weights
— Early stopping: patience=15, metric=val F1
— Epochs : 30 max (best at epoch 15)
Why Swin Transformer over ViT
Feature | ViT-small (old) | Swin-Tiny (current)
Attention type | Global — all 196 patches | Local windows, hierarchical
Data requirement | 10,000+ images | Works on 245 images
Tissue reading | Flat, no hierarchy | Cell-level to tissue-level
Patch size | 16x16 | 4x4 (finer detail)
Our F1 score | 68.35% | 92.86%
App Features
— Preloaded samples — 1 image per tumor subtype (8 total)
— Attention heatmap overlay — spatial attention visualization
— Transformer relationship insights — what the model found in tissue
— Model performance metrics — accuracy, F1, precision, recall
— ViT vs Swin comparison table
Technical Stack
PyTorch 2.0 — model training and inference
timm 0.9.12 — Swin-Tiny pretrained backbone
Streamlit — web application
scikit-learn — metrics and evaluation
Matplotlib — attention heatmap visualization
Docker — containerized deployment
HF Spaces — cloud deployment
Disclaimer
This model is trained on a subset of 245 images for demonstration purposes only. Not intended for clinical use. Results may not generalize to real-world clinical settings.Sonnet 4.6
