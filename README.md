
# 🍅 Tomato Leaf Disease Detection

A **CNN-based model** using **TensorFlow** and **Streamlit** to detect tomato leaf diseases from images. Trained on the **New Plant Diseases Dataset (Augmented)** from Kaggle.

---

## 🧠 Model Summary

Custom **Convolutional Neural Network (CNN)**

* Conv2D (32, 3×3) → MaxPool
* Conv2D (64, 3×3) → MaxPool
* Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, Softmax)
* **Total Params:** 7.39M

---

## 📊 Dataset

**Classes (10):**
Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

Images resized to **128×128** and normalized.

---

## ⚙️ Training

* Epochs: 10
* Batch Size: 32

**Test Accuracy:** ✅ **88.92%**

---

## 📈 Results

* Stable accuracy and loss curves
* Confusion matrix plotted for performance visualization

---

## 💻 Streamlit App

Upload a tomato leaf image → Predict disease → View confidence

Run:

```bash
streamlit run app.py
```

---

## 🧩 Tech Used

TensorFlow · Keras · Streamlit · NumPy · Matplotlib · scikit-learn

---


## 👩‍💻 Author

**Prerana A S**
B.E. Electronics and Telecommunication
BMS College of Engineering
🔗 [GitHub – Prerana-AS](https://github.com/Prerana-AS)
