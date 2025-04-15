# Histopathology Image Classifier - Streamlit Web App

This repository contains a **Streamlit-based web application** for classifying histopathology images into two categories: **Benign (no tumor)** and **Malignant (tumor present)**. The model is trained using the **PCAM (PatchCamelyon) dataset** and deployed for easy use through a web interface.

---

## 🧠 Model Details

- **Architecture**: The model uses **ResNet34**, a deep convolutional neural network architecture well-suited for image classification tasks. ResNet34 is pre-trained on ImageNet and fine-tuned on the PCAM dataset.
- **Training Framework**: Trained using the **Fastai** library on top of **PyTorch**.
- **Training Dataset**: PCAM (PatchCamelyon) histopathology image dataset containing 96x96 pixel image patches.
- **Classification Classes**:
  - `0`: Benign (healthy tissue)
  - `1`: Malignant (cancerous tissue)
- **Accuracy**: Achieved **~97.7% validation accuracy**
- 
## 🚀 Features

- Upload histopathology images in **JPG**, **JPEG**, **PNG**, or **TIFF** format.
- Model classifies the image as **Benign (healthy)** or **Malignant (tumor present)**.
- Provides a **confidence score (%)** for the prediction.
- Simple, user-friendly web interface using **Streamlit**.



