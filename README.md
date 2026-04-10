# 🍎 Smart Retail Object Detection & Price Calculation System

## 📌 Project Overview

This project presents an **AI-powered Smart Retail Checkout System** that automatically detects products and calculates total price using computer vision.

The system supports **multiple deep learning models**:

* YOLOv8 (Real-time detection ⚡)
* Faster R-CNN (High accuracy 🎯)
* SSD (Balanced performance ⚖️)

Users can upload images or use a camera, and the system will:

1. Detect objects (e.g., fruits)
2. Draw bounding boxes
3. Identify item classes
4. Calculate total price automatically

---

## 🎯 Objectives

* Automate retail checkout process
* Compare different object detection algorithms
* Improve detection accuracy for **dense objects (stacked fruits)**
* Build a deployable AI application using Streamlit

---

## 🧠 Models Used

| Model        | Strength                   | Weakness                   |
| ------------ | -------------------------- | -------------------------- |
| YOLOv8       | Fastest, real-time         | May merge objects if dense |
| Faster R-CNN | High accuracy              | Slower                     |
| SSD          | Balanced speed & detection | Less precise than FRCNN    |

---

## 🗂️ Project Structure

```bash
Smart-Retail-System/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── models/                 # (Ignored in GitHub)
│   ├── best.pt
│   ├── fasterrcnn_fruit.pth
│   ├── ssd_fruit.pth
│
├── dataset/                # (Not uploaded)
│
├── utils/ (optional)
│
├── .gitignore
└── .gitattributes
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📦 Model Download (IMPORTANT)

Due to GitHub file size limitations, models are hosted on **Google Drive**.

The system will automatically download:

* YOLOv8 model (`best.pt`)
* Faster R-CNN model
* SSD model

👉 Make sure you have **internet connection** when running the app.

---

## 🧾 Price List Example

```python
PRICE_LIST = {
    "apple": 2.50,
    "banana": 1.50,
    "orange": 2.00,
    "bottle": 3.50,
    "cup": 2.00,
    "pencil": 1.00
}
```

---

## 🖼️ Features

### ✅ Object Detection

* Detect multiple objects in a single image
* Works with dense fruit arrangements

### ✅ Multi-Model Selection

* Compare YOLO, FRCNN, SSD in one system

### ✅ Automatic Billing

* Total price calculated instantly

### ✅ User-Friendly UI

* Built with Streamlit
* Clean and responsive interface

---

## 📊 Dataset

* Source: Roboflow Universe (Fruit Detection Dataset)
* Format: YOLO format (bounding boxes)
* Enhanced with:

  * Dense object images
  * Multi-object scenes
  * Real-world retail conditions

---

## ⚠️ Challenges & Solutions

### ❗ Problem: YOLO detects one large box

✔ Solution:

* Improved dataset (dense annotations)
* Adjusted IOU & confidence threshold
* Added more training images

---

### ❗ Problem: Model size too large for GitHub

✔ Solution:

* Store models in Google Drive
* Auto-download in Streamlit

---

### ❗ Problem: Detection inconsistency

✔ Solution:

* Use multiple models
* Cross-compare outputs

---

## 📈 Future Improvements

* Add barcode detection
* Integrate real-time video tracking
* Improve dataset diversity
* Deploy on cloud (Streamlit Cloud / AWS)

---

## 👨‍💻 Technologies Used

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Streamlit

---

## 📌 Conclusion

This system demonstrates how AI can transform retail checkout by:

* Reducing manual effort
* Improving efficiency
* Providing real-time object detection

It also highlights the importance of:

* Dataset quality
* Model selection
* System integration

---

## 🙌 Acknowledgement

Dataset provided by:

* Roboflow Universe

---

## 📬 Contact

For questions or improvements, feel free to reach out.

```
```
