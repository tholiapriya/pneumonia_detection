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
    url = "https://drive.google.com/uc?id=1aZ9JKFvIcqV4I0d9Kf6jcj6qq_-LD_H2"  # your Drive file ID
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
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))  # adjust to match your model input
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Prediction
    prediction = model.predict(x)
    if prediction[0][0] > 0.5:
        st.error("⚠️ The model predicts **Pneumonia**")
    else:
        st.success("✅ The model predicts **Normal**")
