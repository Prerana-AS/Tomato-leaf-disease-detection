import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown  # for downloading model from Google Drive

st.set_page_config(page_title="Tomato Disease Detector 🍅", layout="centered")

st.title("🍅 Tomato Leaf Disease Detection")
# Google Drive model setup
MODEL_PATH = "model_inception.h5"
DRIVE_LINK = "https://drive.google.com/uc?id=1M502iP248Ivu417jdzYafUdH6C3qThEI"

# Download model automatically if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... Please wait ⏳")
    gdown.download(DRIVE_LINK, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully ✅")



# Load model once
@st.cache_resource
def load_tomato_model():
    return load_model("model_inception.h5")

model = load_tomato_model()

# Class labels (your dataset order)
class_names = [
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Early_blight',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Bacterial_spot',
    'Tomato___Target_Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Late_blight',
    'Tomato___healthy',
    'Tomato___Leaf_Mold'
]

uploaded_file = st.file_uploader("📷 Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))  # match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing the leaf..."):
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    st.success(f"✅ **Predicted Disease:** {class_names[class_index]}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

    if "healthy" in class_names[class_index].lower():
        st.balloons()
        st.write("🎉 The plant looks healthy!")
    else:
        st.warning("⚠️ The plant seems affected. Consider checking treatment options.")

st.markdown("---")
st.caption("Developed by Prerana 🌱 | Powered by TensorFlow & Streamlit")
