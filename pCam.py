import streamlit as st
from fastai.vision.all import *
import pathlib
import sys
import streamlit as st
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImageLib

# Patch for Windows
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

import os
import requests
import streamlit as st
from fastai.vision.all import *

MODEL_URL = (
    "https://drive.google.com/uc?export=download&id=1nrM5JPM4g_VZ2cN3uyxEuIo0svFF3I-o"
)
MODEL_PATH = "model.pkl"


def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            st.success("Model downloaded successfully!")


download_model()
learn = load_learner(MODEL_PATH)

# # Load the trained model
# learn = load_learner("model.pkl")
# Display the title of the web app
st.title("Histopathology Tumor Classifier")

# Description of the app
st.write(
    """
    This model is trained on the PCam dataset to classify histopathology images as **Benign (Healthy)** or **Malignant (Tumor)**.
    It predicts whether the given histopathology image contains a tumor or not based on the trained deep learning model.
    The model provides a prediction with a confidence score.
"""
)

# Upload image section
uploaded_file = st.file_uploader(
    "Choose a histopathology image...", type=["tif", "jpg", "jpeg", "png"]
)

# Ensure the image is uploaded
if uploaded_file is not None:
    # Convert the uploaded file to a PIL Image
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (resize to the same size as during training)
    img = PILImage.create(uploaded_file)
    img = img.resize(
        (224, 224)
    )  # Adjust the size if needed to match model input size (e.g., 224x224)

    # Make prediction
    pred_class, pred_idx, probs = learn.predict(img)

    # Display the predicted class and confidence score
    class_labels = (
        learn.dls.vocab
    )  # ['Benign', 'Malignant'] or your model's class labels

    # Map the predicted index to the label
    predicted_label = class_labels[pred_idx]

    # Replace the class labels with more understandable terms
    if predicted_label == 0:
        human_readable_label = "Benign (Healthy)"
    else:
        human_readable_label = "Malignant (Tumor)"

    # Print the class and the confidence score
    st.write(f"Predicted Class: {human_readable_label}")
    st.write(f"Prediction Confidence: {probs[pred_idx] * 100:.2f}%")
