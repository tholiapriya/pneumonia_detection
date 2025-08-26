import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------------------
# Load model from Google Drive
# ---------------------------
@st.cache_resource
def load_pneumonia_model():
    url = "https://drive.google.com/uc?id=15GAlu1opKuFLqmJ7cOvaNQf_9FEIY6XW"  # new Drive file ID
    output = "pneumonia_model.h5"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return load_model(output)

model = load_pneumonia_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)  # updated to avoid warning

    # ---------------------------
    # Preprocess image (auto-adapts to model input)
    # ---------------------------
    _, h, w, c = model.input_shape  # e.g. (None, 150, 150, 3)

    # Convert to correct color mode
    if c == 1:
        img = img.convert("L")   # grayscale
    else:
        img = img.convert("RGB") # 3 channels

    # Resize
    img = img.resize((w, h))

    # Convert to array
    x = image.img_to_array(img)

    # If grayscale, ensure correct shape
    if c == 1:
        x = np.expand_dims(x, axis=-1)

    # Add batch dim + normalize
    x = np.expand_dims(x, axis=0) / 255.0

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = model.predict(x)

    if prediction[0][0] > 0.5:
        st.error("⚠️ The model predicts **Pneumonia**")
    else:
        st.success("✅ The model predicts **Normal**")
