from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from anomaly_detection import build_heatmap, compute_image_score, compute_reconstruction_error
from config import DATASET_ROOT, IMAGE_SIZE, MODELS_DIR
from preprocessing import load_rgb_image

st.set_page_config(page_title="Industrial Anomaly Detector", layout="wide")

CATEGORY_CHOICES = ["bottle", "cable", "capsule", "hazelnut"]
MODEL_CHOICES = ["ae_cnn", "denoising_ae_cnn", "vae_cnn"]


def load_model_for_prediction(category: str, model_name: str):
    model_path = MODELS_DIR / f"{category}_{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found for {category}/{model_name}. Train it first.")
    return tf.keras.models.load_model(model_path)


st.title("Industrial Anomaly Detection Demo")
st.write("Upload an image and compare reconstruction-based anomaly scores across models.")

with st.sidebar:
    selected_category = st.selectbox("Product category", CATEGORY_CHOICES)
    selected_model = st.selectbox("Model", MODEL_CHOICES)
    threshold_percentile = st.slider("Threshold percentile", 80, 99, 95, 1)

uploaded_file = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        model = load_model_for_prediction(selected_category, selected_model)
        reconstructed = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
        error_map = compute_reconstruction_error(image_array, reconstructed)
        score = compute_image_score(error_map)
        threshold = np.percentile(np.asarray([score], dtype=np.float32), threshold_percentile)
        heatmap = build_heatmap(error_map)

        col1, col2, col3 = st.columns(3)
        col1.image(image_array, caption="Original", use_column_width=True)
        col2.image(reconstructed, caption="Reconstruction", use_column_width=True)
        col3.image(heatmap, caption="Heatmap", use_column_width=True, clamp=True)

        st.write(f"Anomaly score: {score:.6f}")
        st.write(f"Selected threshold: {threshold:.6f}")
        label = "Anomalous" if score > threshold else "Normal"
        st.success(f"Prediction: {label}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
else:
    st.info("Please upload an image to evaluate.")
