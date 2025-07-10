# Batik Besurek Multi-Label Classification using Transfer Learning

This repository contains the code, models, and supporting files for a deep learning project focused on multi-label classification of Batik Besurek motifs using various transfer learning architectures.

## 📌 Project Overview

This study explores and compares the performance of 9 pre-trained CNN architectures to classify Batik Besurek images into 11 motif labels. The models are trained and evaluated under two scenarios: with and without data augmentation.

## 🧠 Models Used

The following pre-trained models were evaluated using feature extraction (no fine-tuning):

- DenseNet201 ✅ *(Best-performing model)*
- InceptionV3
- InceptionResNetV2
- MobileNetV2
- Xception
- VGG16
- VGG19
- ResNet50
- EfficientNetB3 ❌ *(Worst-performing model)*

## 📂 Dataset

- **Source**: Original Batik Besurek images collected from local artisans and augmented using Roboflow.
- **Classes**: 11 motif categories 
- **Total Images**: 
  - Before Augmentation: 4,114 images  
  - After Augmentation: 9,474 images
- **Link Dataset**
  - **Original Data** : https://drive.google.com/drive/folders/1x98XKIF6FvkUozV_NDbXZB3-qjszy3Jr?usp=sharing
  - **After Augmentatiton** : https://drive.google.com/drive/folders/1RPDGy3dJHjGId4SGXj3W9oDDLNiqOSjR?usp=sharing
## 🧪 Evaluation Metrics

Each model was evaluated using the following metrics:

- AUC
- F1-Score
- Cohen’s Kappa
- Accuracy
- Final Score (average of all metrics)

## 🔍 Key Findings

- **DenseNet201** achieved the highest performance with Final Score of `0.9989` on augmented data.
- **EfficientNetB3** struggled to generalize and heavily biased toward the dominant class, with Final Score below `0.31`.
- Data augmentation helped reduce overfitting and improve generalization in almost all models.

## 📈 Visualizations

- Training history (Loss & AUC)
- Label distribution (with and without augmentation)
- Confusion matrices
- Prediction vs Ground Truth samples

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Roboflow (for annotation and augmentation)


