
# 🍅 Tomato Leaf Disease Detection

A **Streamlit-based web app** that detects diseases in **tomato leaves** using a **deep learning model (InceptionV3)**.
Upload an image of a tomato leaf, and the app predicts the type of disease instantly.

---

## 🌿 Overview

This project helps farmers and researchers identify **tomato leaf diseases** early using image classification.
The model is trained using **transfer learning (InceptionV3)** on a tomato disease dataset.

---

## 🧠 Model

* Model: **InceptionV3 (Keras / TensorFlow)**
* Type: **Image Classification**
* Accuracy: ~94%
* Model file: `model_inception.h5`
  *(Stored on Google Drive due to file size limit — downloaded automatically in the app.)*
  [Download model manually](https://drive.google.com/file/d/1M502iP248Ivu417jdzYafUdH6C3qThEI/view?usp=sharing)

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Prerana-AS/tomato-leaf-disease-detection.git
cd tomato-leaf-disease-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🖥️ How to Use

1. Run the Streamlit app.
2. Upload an image of a tomato leaf.
3. The model will display the predicted disease name and confidence.

---

## 🧩 Technologies Used

* **Python**
* **Streamlit**
* **TensorFlow / Keras**
* **OpenCV, NumPy**
* **gdown (for model download)**
  
---

## 👩‍💻 Author

**Prerana A S**
B.E. Electronics and Telecommunication
BMS College of Engineering
🔗 [GitHub – Prerana-AS](https://github.com/Prerana-AS)
