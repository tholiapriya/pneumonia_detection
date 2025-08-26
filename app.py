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
    file_id = "1XZSNBfgBbqTsh7YVEhq8PviRSnRJQAgx"  # your correct file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "pneumonia_model.h5"

    if not os.path.exists(output):
        try:
            gdown.download(url, output, quiet=False)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

    return load_model(output)

# Load model
model = load_pneumonia_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # Preprocess image
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

    # Ensure shape matches model input
    if c == 1:
        x = np.expand_dims(x, axis=-1)

    # Add batch dim + normalize
    x = np.expand_dims(x, axis=0) / 255.0

    # ---------------------------
    # Prediction
    # ---------------------------
    try:
        prediction = model.predict(x)
        if prediction[0][0] > 0.5:
            st.error("⚠️ The model predicts **Pneumonia**")
        else:
            st.success("✅ The model predicts **Normal**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
