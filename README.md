# Brain Tumor MRI Classification Using Deep Learning

A complete computer vision and deep learning pipeline for classifying brain MRI images into three tumor categories using a custom Convolutional Neural Network (CNN).

> **Note:** This is an educational machine learning project. It is **not** a clinical diagnostic tool and must not be used to make medical decisions.

---

## Project Overview

This project develops a CNN that classifies brain MRI scans into:

| Label | Class |
|-------|-------|
| 0 | Meningioma |
| 1 | Glioma |
| 2 | Pituitary Tumor |

The full pipeline covers dataset auditing, patient-level data splitting (to prevent leakage), MRI preprocessing, model architecture, training with early stopping, evaluation, explainability via Grad-CAM, and final model export.

---

## Dataset

- **Source:** [Brain Tumor Dataset — Cheng et al.](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) via Kaggle (`nahin333/brain-tumor-dataset`)
- **Format:** MATLAB v7.3 `.mat` files, each containing an MRI image, tumor mask, class label, and patient ID
- **Size:** 3,064 images from **233 unique patients**
- **Class distribution:**
  - Glioma: 1,426 images
  - Pituitary: 930 images
  - Meningioma: 708 images

---

## Repository Structure

```
brain-tumor-mri-classification-using-deeplearning/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── brain-tumor-classification-using-deep-learning.ipynb
│
├── models/
│   └── best_brain_tumor_cnn.keras
│
└── reports/
    ├── confusion_matrix.png
    ├── roc_auc.png
    ├── training_accuracy.png
    ├── training_loss.png
    ├── classification_report.txt
    └── predictions.csv
```

---

## Model Architecture

- **Type:** Custom CNN (built with TensorFlow / Keras)
- **Input:** 224 × 224 × 1 (grayscale, normalized to [0, 1])
- **Output:** 3-class softmax
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Regularization:** Class weights, early stopping, model checkpointing

---

## Results

| Split | Accuracy |
|-------|----------|
| Training (final epoch) | 77.58% |
| Validation (best epoch) | 73.09% |
| **Test** | **70.76%** |

**Per-class test performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Meningioma | 0.4815 | 0.6265 | 0.5445 | 166 |
| Glioma | 0.8505 | 0.5749 | 0.6861 | 287 |
| Pituitary | 0.7983 | 0.9789 | 0.8794 | 190 |
| **Macro avg** | **0.7101** | **0.7268** | **0.7033** | 643 |

**Confusion matrix:**

|  | Pred: Meningioma | Pred: Glioma | Pred: Pituitary |
|--|-----------------|-------------|----------------|
| **Actual: Meningioma** | 104 | 27 | 35 |
| **Actual: Glioma** | 110 | 165 | 12 |
| **Actual: Pituitary** | 2 | 2 | 186 |

---

## Report Artifacts

| File | Description |
|------|-------------|
| `reports/training_accuracy.png` | Accuracy learning curves (train vs. validation) |
| `reports/training_loss.png` | Loss learning curves (train vs. validation) |
| `reports/confusion_matrix.png` | Confusion matrix on the test set |
| `reports/roc_auc.png` | Multi-class ROC-AUC curves (one-vs-rest) |
| `reports/classification_report.txt` | Full classification report with all metrics |
| `reports/predictions.csv` | Per-sample predictions on the test set |

---

## Installation

```bash
git clone https://github.com/21Oli/brain-tumor-mri-classification-using-deeplearning.git
cd brain-tumor-mri-classification-using-deeplearning
pip install -r requirements.txt
```

---

## Running the Notebook

1. Download the dataset from Kaggle (`nahin333/brain-tumor-dataset`) and place it at the path configured in Section 02 of the notebook.
2. Open the notebook:

```bash
jupyter notebook notebooks/brain-tumor-classification-using-deep-learning.ipynb
```

---

## Environment

| Library | Version |
|---------|---------|
| Python | 3.12.13 |
| TensorFlow | 2.20.0 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| Pandas | 2.2.3 |
| Pillow | 11.1.0 |
| h5py | 3.12.1 |

---

## Key Design Decisions

- **Patient-level splitting** — images are split by patient ID (not randomly) to prevent data leakage. Each of the 233 patients appears in exactly one fold.
- **Predefined CV folds** — the dataset's `cvind.mat` file is used to assign patients to folds 1–5. Folds 1–3 = train, fold 4 = validation, fold 5 = test.
- **Image preprocessing** — raw 512×512 / 256×256 grayscale MRIs are normalized to [0, 1] and resized to 224×224×1.
- **Class imbalance** — addressed with per-class weights during training.
- **Explainability** — Grad-CAM is applied to visualize which MRI regions the CNN focuses on.

---

## Limitations

- Dataset contains only 3,064 images from 233 patients — a small clinical sample.
- Meningioma is underrepresented relative to glioma, impacting per-class F1.
- The model was trained and evaluated on a single dataset; external validation is needed before any real-world use.
- This project is **not a medical diagnostic system**.

---

## License

MIT — see [LICENSE](LICENSE).
