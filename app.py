import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Google Drive model file ID
file_id = "1-6KoTZHvirLz8IHxLId3o8NJdku-n6Tx"
model_path = "WCE_Curated_Colon_Detection_Model.h5"

# Download the model from Google Drive
@st.cache_resource
def load_model():
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# Define class names
class_names = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Streamlit UI
st.title("🔬 WCE Curated Colon Disease Detection")
st.write("Upload an image to classify colon disease.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        predicted_class, confidence = predict(image)
        st.success(f"✅ Prediction: {predicted_class}")
        st.info(f"📊 Confidence: {confidence:.2f}%")
