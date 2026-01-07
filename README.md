# Brain Tumor Classification

This is a **practice deep learning project** for image classification using PyTorch.  
The goal was to understand the **end-to-end CNN pipeline** — data loading, transforms, model design, debugging shape errors, and inference — not to build a production-grade medical model.

---

## 📌 Project Overview

- **Task**: Binary image classification (Tumor vs No Tumor)
- **Framework**: PyTorch
- **Dataset**: Small Brain Tumor Image Dataset (~253 images)
- **Classes**:
  - `no` → No tumor
  - `yes` → Tumor

⚠️ **Note**: The dataset is very small, so model performance is limited. This project is strictly for learning and experimentation.

---

## 🧠 Model Architecture

- Custom CNN built from scratch
- Convolution + ReLU + MaxPooling layers
- Fully connected classifier head
- Adaptive pooling used to handle fixed feature size

> This model is **not suitable for real medical use** due to dataset size and lack of clinical validation.

---

## 🔄 Data Preprocessing

- Resize images to `256 × 256`
- Random augmentations (training only):
  - Horizontal flip
  - Rotation
- Normalization:
  ```python
  mean = (0.5, 0.5, 0.5)
  std  = (0.5, 0.5, 0.5)
  
## 📊 Evaluation

* Metric used: Accuracy
* Si gle-image inference supported
* Class prediction mapped using ImageFolder.classes
* ⚠️ Accuracy is not reliable for this dataset size.
False negatives are possible.

## 🧪 What This Project Covers

* Building a CNN from scratch
* Debugging tensor shape mismatches
* Understanding feature map sizes
* Training and evaluation loop
* Single image inference
* Interpreting model predictions

## ❌ Limitations

* Very small dataset (~253 images)
* No class imbalance handling
* No advanced metrics (recall, ROC-AUC)
* No transfer learning
* Not medically reliable

## ✅ Future Improvements

* Use a larger dataset (Kaggle)
* Apply transfer learning (ResNet / EfficientNet)
* Add confusion matrix and recall metrics
* Improve handling of class imbalance
* Visualize predictions using Grad-CAM