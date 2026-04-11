# FruitScanAI: Smart Retail Checkout System

## Project Overview
FruitScanAI is an AI-powered Smart Retail Checkout System developed as a prototype for automated product detection and billing in a retail environment. The system allows users to upload product images or use a webcam image for object detection, and it automatically generates an estimated checkout summary based on the detected items.

This project focuses on fruit object detection and compares the performance of three deep learning models:
- YOLO
- Faster R-CNN
- SSD

The system is deployed as a Streamlit web application and supports multi-image upload, model comparison, confidence-based detection, and automatic billing calculation.

---

## Objectives
The main objectives of this project are:
- To develop an intelligent retail checkout prototype using computer vision
- To detect fruit items automatically from images
- To compare the detection performance of YOLO, Faster R-CNN, and SSD
- To generate automatic billing results based on detected products
- To provide an interactive and user-friendly web interface for demonstration purposes

---

## Features
- Upload at least 3 product images for testing
- Webcam image capture support
- Fruit object detection using deep learning models
- Compare YOLO, Faster R-CNN, and SSD on the same images
- Automatic billing and checkout summary
- Multi-image detection results display
- Confidence threshold adjustment
- Clean and interactive Streamlit user interface
- Hugging Face model hosting support

---

## Technologies Used
- **Python**
- **Streamlit**
- **PyTorch**
- **Torchvision**
- **Ultralytics YOLO**
- **OpenCV**
- **Pillow**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Hugging Face**

---

## Models Used
This project uses the following trained models:
- **YOLO model**: best.pt
- **Faster R-CNN model**: fasterrcnn_fruit.pth
- **SSD model**: ssd_fruit.pth

The models are hosted on Hugging Face due to file size limitations on GitHub.

### Hugging Face Model Repository
https://huggingface.co/Gooi0212/fruit-detection-models

---

## Supported Classes
The system currently supports the following fruit classes:
- Apple
- Banana
- Orange
- Mango
- Pineapple
- Watermelon

---

## Pricing List
The estimated billing uses the following sample price list:

- Apple — RM2.50
- Banana — RM1.50
- Orange — RM2.20
- Mango — RM4.00
- Pineapple — RM5.50
- Watermelon — RM8.00

---

## System Workflow
1. User uploads product images or captures an image using webcam
2. The selected deep learning model processes the image
3. The system detects fruit objects and draws bounding boxes
4. Detected items are counted
5. The billing module calculates the estimated total price
6. Users may compare all three models for performance evaluation

---

## Project Structure
```bash
FruitScanAI/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .gitignore
├── .gitattributes
└── models/   # optional local folder if running locally
