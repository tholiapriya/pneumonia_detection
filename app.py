import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------
# Download & Load Model from Google Drive
# ---------------------------
@st.cache_resource
def load_pneumonia_model():
    file_id = "1XZSNBfgBbqTsh7YVEhq8PviRSnRJQAgx"  # replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "pneumonia_model.h5"
    gdown.download(url, output, quiet=False)
    model = load_model(output)
    return model

model = load_pneumonia_model()

# ---------------------------
# Preprocess function
# ---------------------------
def preprocess(img: Image.Image):
    img = img.convert("RGB")             # match training (RGB)
    img = img.resize((224, 224))         # match training size
    img = np.array(img) / 255.0          # normalize
    img = np.expand_dims(img, axis=0)    # add batch dimension
    return img

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

# Add threshold slider
threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        x = preprocess(img)
        prob = float(model.predict(x)[0][0])

        if prob > threshold:
            st.error(f"Prediction: Pneumonia (Confidence: {prob:.2f})")
        else:
            st.success(f"Prediction: Normal (Confidence: {1-prob:.2f})")
