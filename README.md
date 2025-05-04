# 🎯 AI-Powered Prediction App

This project is part of **Data Science Assignment 4**, focusing on **binary classification of audio deepfakes** and **multi-label classification of bug reports**. The final deliverable is a **real-time Streamlit application** allowing users to upload `.wav` files or enter text and get live predictions using different ML models.

---

## 🧠 Tasks Implemented

### 🔹 Part 1: Audio Deepfake Detection (Binary Classification)
- Synthetic audio signal generation for bonafide/spoof
- MFCC feature extraction with `librosa`
- Trained classifiers:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Perceptron
  - Deep Neural Network (2 hidden layers)
- Evaluation: Accuracy, F1-Score, ROC-AUC, Precision, Recall

### 🔹 Part 2: Bug Report Defect Prediction (Multi-label Classification)
- Tfidf vectorization of short bug report texts
- Manual multi-label vectors (blocker, bug, regression, etc.)
- Models: OneVsRest with LR, SVM, Perceptron + DNN
- Metrics: Hamming Loss, Micro/Macro F1 Score

### 🔹 Part 3: Streamlit App
- Upload `.wav` files for real-time deepfake detection
- Enter bug report text for defect label predictions
- Select model at runtime (SVM, LR, DNN, Perceptron)
- Waveform plot with full/short view toggle
- Confidence scores shown as progress bars
- Fully modular and interactive interface

---

## 🚀 Getting Started

### ▶️ Run the App

streamlit run my_app.py

---

## 🔧 Tech Stack

- **Python 3.10+**
- `streamlit`
- `scikit-learn`
- `librosa`
- `PyTorch`
- `matplotlib`

---

## 🙌 Acknowledgment

This app was created for the **Data Science A4 Assignment** under the guidance of Sir Usama.
## Medium Blog Post
https://medium.com/@f223646/building-an-ai-powered-streamlit-app-for-audio-deepfake-multi-label-text-classification-ed7458b5d65b
## LinkedIn Post
https://www.linkedin.com/posts/ayesha-naseem-061b35344_machinelearning-streamlit-audioprocessing-activity-7324757886209851392-oS4K?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFZPGQkBvcNGbwsecx5RVfmrL_gOMqte_1c
