import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
import gdown

# Import DTypePolicy from Keras 3 (for compatibility)
try:
    from keras.dtype_policies import DTypePolicy
except ImportError:
    st.error("Keras 3 not installed. Run 'pip install --upgrade keras' and restart.")
    st.stop()

# 🏷️ Title
st.title("🍅 Tomato Disease Detection")

# 📁 Model path
MODEL_PATH = "tomato_disease_model.h5"

# Replace with your Google Drive file ID
FILE_ID = "1uUP-IPSgeIJxm0AmZiGaJZuR1UShoHWA"
if not os.path.exists(MODEL_PATH):
    st.info("⏬ Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ✅ Custom InputLayer to fix old model config issues
from tensorflow.keras.layers import InputLayer

class CustomInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super().from_config(config)

@st.cache_resource
def load_model_cached():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            'InputLayer': CustomInputLayer,
            'DTypePolicy': DTypePolicy
        }
    )
    return model

model = load_model_cached()

# 🗂️ Load class names
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
except Exception as e:
    st.error(f"Error loading class names: {e}")
    class_names = [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites_Two_spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
# 🌿 Remedies for each tomato leaf disease
remedies = {
    "Tomato___Bacterial_spot": 
    "• Remove infected leaves.\n• Apply copper-based bactericides.\n• Avoid overhead watering.\n• Practice crop rotation.",

    "Tomato___Early_blight": 
    "• Remove affected leaves.\n• Spray copper fungicide.\n• Maintain proper spacing between plants.\n• Avoid wetting leaves.",

    "Tomato___Late_blight":
    "• Remove and destroy infected plants.\n• Apply mancozeb or chlorothalonil.\n• Improve air circulation.\n• Avoid overhead irrigation.",

    "Tomato___Leaf_Mold":
    "• Increase ventilation.\n• Apply sulfur or copper fungicide.\n• Keep leaves dry.",

    "Tomato___Septoria_leaf_spot":
    "• Remove infected leaves.\n• Use copper fungicides.\n• Use drip irrigation.",

    "Tomato___Spider_mites_Two_spotted_spider_mite":
    "• Spray neem oil.\n• Increase humidity.\n• Remove heavily infested leaves.",

    "Tomato___Target_Spot":
    "• Apply azoxystrobin or mancozeb.\n• Keep foliage dry.\n• Remove diseased leaves.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
    "• No cure.\n• Remove infected plants.\n• Control whiteflies using neem oil.\n• Use resistant varieties.",

    "Tomato___Tomato_mosaic_virus":
    "• No direct cure.\n• Remove infected plants.\n• Disinfect hands and tools.\n• Avoid tobacco products near plants.",

    "Tomato___healthy":
    "• Plant is healthy! Continue regular watering, sunlight and nutrients."
}


if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((128, 128))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[class_index]
        confidence = np.max(prediction) * 100

       


        # Display result
        st.success(f"###🌿 Predicted Disease: **{predicted_class}**")
        st.info(f"🧠 Confidence: {confidence:.2f}%")
         # Show remedy
        st.write("# 🌱 Recommended Remedy:")
        st.info(remedies[predicted_class])

        if "healthy" in predicted_class.lower():
            st.balloons()
            st.write("🎉 The plant looks healthy!")
        else:
            st.warning("⚠️ The plant seems affected. Consider checking treatment options.")

st.markdown("---")
st.caption("Developed by Prerana A S")


