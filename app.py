import requests
import streamlit as st
import numpy as np
import pandas as pd
import torch
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torchvision
import os
import time
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Retail Checkout",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS
# =========================
def apply_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        html, body, [class*="css"] {
            color: #0f172a;
        }

        h1, h2, h3, h4, h5, h6,
        p, span, label, div, small {
            color: #0f172a !important;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
            padding: 28px 30px;
            border-radius: 22px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }

        .hero-card,
        .hero-card * {
            color: #ffffff !important;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.95;
        }

        .status-pill {
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.28);
            color: #ffffff !important;
            display: inline-block;
            margin-bottom: 12px;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }

        .mini-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 16px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            height: 100%;
        }

        .label-text {
            font-size: 0.85rem;
            color: #64748b !important;
        }

        .value-text {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] {
            background: #f8fbff !important;
            border-right: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #0f172a !important;
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
        }

        div[data-baseweb="select"] * {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        div[data-baseweb="popover"] {
            background: #ffffff !important;
        }

        div[data-baseweb="popover"] * {
            color: #0f172a !important;
        }

        ul[role="listbox"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }

        li[role="option"] {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        li[role="option"] * {
            color: #0f172a !important;
        }

        li[role="option"]:hover {
            background: #e0e7ff !important;
        }

        li[aria-selected="true"] {
            background: #dbeafe !important;
        }

        li[aria-selected="true"] * {
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        div[role="radiogroup"] label,
        div[role="radiogroup"] label * {
            color: #0f172a !important;
        }

        div[data-testid="stSlider"] label,
        div[data-testid="stSlider"] span,
        div[data-testid="stSlider"] * {
            color: #0f172a !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            color: #0f172a !important;
            background: #ffffff !important;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 18px;
            padding: 14px;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stMetric"] * {
            color: #0f172a !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #e8eefc !important;
            border-radius: 12px !important;
            padding: 0.55rem 1rem !important;
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        .stTabs [data-baseweb="tab"] * {
            color: #0f172a !important;
        }

        .stTabs [aria-selected="true"] {
            background: #2563eb !important;
        }

        .stTabs [aria-selected="true"],
        .stTabs [aria-selected="true"] * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploader"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 12px !important;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        }

        [data-testid="stFileUploader"] * {
            color: #0f172a !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border-radius: 14px !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border: 2px dashed #2563eb !important;
            background: #f8fbff !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: #2563eb !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploaderDropzone"] button * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] p {
            color: #0f172a !important;
        }

        /* uploaded file chip -> white text */
        [data-testid="stFileUploaderFile"] {
            background: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 12px !important;
        }

        [data-testid="stFileUploaderFile"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        [data-testid="stFileUploaderFileName"] {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploaderFileData"] {
            color: #e2e8f0 !important;
        }

        [data-testid="stFileUploaderFile"] button {
            background: transparent !important;
            color: #ef4444 !important;
            border: none !important;
        }

        [data-testid="stFileUploaderFile"] button * {
            color: #ef4444 !important;
            fill: #ef4444 !important;
        }

        [data-testid="stFileUploader"] svg {
            color: #2563eb !important;
            fill: #2563eb !important;
        }

        [data-testid="stCameraInput"] {
            background: #ffffff !important;
            border-radius: 16px !important;
            padding: 10px !important;
            border: 1px solid #e2e8f0 !important;
        }

        [data-testid="stCameraInput"] * {
            color: #0f172a !important;
        }

        [data-testid="stCameraInput"] button {
            background: #0f172a !important;
            color: #ffffff !important;
        }

        [data-testid="stCameraInput"] button * {
            color: #ffffff !important;
        }

        .stButton button {
            background: #ffffff !important;
            color: #0f172a !important;
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
        }

        .stDownloadButton button {
            color: #0f172a !important;
        }

        [data-testid="stDataFrame"] {
            background: #ffffff !important;
            border-radius: 12px !important;
        }

        [data-testid="stDataFrame"] * {
            color: #0f172a !important;
        }

        [data-testid="stTable"] * {
            color: #0f172a !important;
        }

        .stAlert, .stInfo, .stSuccess, .stWarning {
            border-radius: 12px !important;
        }

        .stAlert *,
        .stInfo *,
        .stSuccess *,
        .stWarning * {
            color: #0f172a !important;
        }

        button[title="Fullscreen"],
        button[title="View fullscreen"],
        [data-testid="stElementToolbarButton"] {
            background: rgba(15, 23, 42, 0.85) !important;
            border-radius: 8px !important;
        }

        button[title="Fullscreen"],
        button[title="View fullscreen"],
        [data-testid="stElementToolbarButton"],
        [data-testid="stElementToolbarButton"] * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        div[role="tooltip"],
        div[role="tooltip"] * {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_custom_css()

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["apple", "banana", "orange", "mango", "pineapple", "watermelon"]

PRICE_LIST = {
    "apple": 2.50,
    "banana": 1.50,
    "orange": 2.20,
    "mango": 4.00,
    "pineapple": 5.50,
    "watermelon": 8.00
}

YOLO_MODEL_PATH = "models/best.pt"
YOLO_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1onDCGuzge1fYdVtJifxHwUDk4Zhpgncg"
FRCNN_MODEL_PATH = "models/fasterrcnn_fruit.pth"
FRCNN_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1fcqjporjX9IKuA-YQNR3vwqo8mj_Xncm"
SSD_MODEL_PATH = "models/ssd_fruit.pth"
SSD_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1ft9Yr5UbPBnWjIeDHo6JOIQM2jf46QGj"

# =========================
# HELPERS
# =========================
def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def calculate_bill(detected_items):
    filtered_items = [item for item in detected_items if item in PRICE_LIST]
    item_counts = Counter(filtered_items)

    bill_rows = []
    total_price = 0.0

    for item, qty in item_counts.items():
        unit_price = PRICE_LIST[item]
        subtotal = unit_price * qty
        total_price += subtotal
        bill_rows.append(
            {
                "Item": item.title(),
                "Quantity": qty,
                "Unit Price (RM)": unit_price,
                "Subtotal (RM)": subtotal,
            }
        )

    return bill_rows, total_price


def draw_boxes_pil(image_np, boxes, labels, scores):
    image_pil = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        text = f"{label} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(text) * 7
            text_h = 15

        text_x1 = x1
        text_y1 = max(0, y1 - text_h - 8)
        text_x2 = x1 + text_w + 8
        text_y2 = text_y1 + text_h + 6

        draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill="red")
        draw.text((text_x1 + 4, text_y1 + 3), text, fill="white", font=font)

    return np.array(image_pil)


def download_from_drive(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    session = requests.Session()
    response = session.get(url, stream=True)

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_url = url + "&confirm=" + value
            response = session.get(confirm_url, stream=True)
            break

    total_size = int(response.headers.get("content-length", 0))
    progress = st.progress(0, text=f"Downloading {os.path.basename(save_path)}...")

    downloaded = 0

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = min(downloaded / total_size, 1.0)
                    progress.progress(
                        percent,
                        text=f"Downloading {os.path.basename(save_path)}... {int(percent * 100)}%"
                    )

    progress.empty()


def ensure_model(path, url):
    if not os.path.exists(path):
        st.warning(f"{os.path.basename(path)} not found. Downloading from Google Drive...")
        download_from_drive(url, path)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_yolo():
    ensure_model(YOLO_MODEL_PATH, YOLO_DRIVE_URL)
    model = YOLO(YOLO_MODEL_PATH)
    return model


@st.cache_resource
def load_frcnn():
    ensure_model(FRCNN_MODEL_PATH, FRCNN_DRIVE_URL)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=len(CLASSES) + 1
    )

    checkpoint = torch.load(FRCNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_ssd():
    ensure_model(SSD_MODEL_PATH, SSD_DRIVE_URL)

    model = torchvision.models.detection.ssd300_vgg16(
        weights=None,
        weights_backbone=None,
        num_classes=len(CLASSES) + 1
    )

    checkpoint = torch.load(SSD_MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def get_selected_model(model_choice):
    if model_choice == "YOLO":
        return load_yolo()
    if model_choice == "Faster R-CNN":
        return load_frcnn()
    return load_ssd()

# =========================
# DETECTION FUNCTIONS
# =========================
def detect_with_yolo(model, image_np, min_confidence):
    results = model(image_np, conf=min_confidence, iou=0.50)
    result = results[0]

    rendered = result.plot()
    detected_items = []
    rows = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = model.names[cls_id]

        rows.append(
            {
                "Detected Item": label,
                "Confidence": round(conf, 4),
                "Billable": "Yes" if label in PRICE_LIST else "No",
            }
        )

        if label in PRICE_LIST:
            detected_items.append(label)

    df = pd.DataFrame(rows)
    return rendered, detected_items, df


def detect_with_frcnn(model, image_np, min_confidence):
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    labels_tensor = outputs["labels"].detach().cpu().numpy()

    detected_items = []
    rows = []
    draw_boxes = []
    draw_labels = []
    draw_scores = []

    for box, score, label_id in zip(boxes, scores, labels_tensor):
        if float(score) < min_confidence:
            continue

        label_index = int(label_id) - 1
        if label_index < 0 or label_index >= len(CLASSES):
            continue

        label = CLASSES[label_index]

        rows.append(
            {
                "Detected Item": label,
                "Confidence": round(float(score), 4),
                "Billable": "Yes" if label in PRICE_LIST else "No",
            }
        )

        if label in PRICE_LIST:
            detected_items.append(label)

        draw_boxes.append(box)
        draw_labels.append(label)
        draw_scores.append(float(score))

    rendered = draw_boxes_pil(image_np, draw_boxes, draw_labels, draw_scores)
    df = pd.DataFrame(rows)
    return rendered, detected_items, df


def detect_with_ssd(model, image_np, min_confidence):
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    labels_tensor = outputs["labels"].detach().cpu().numpy()

    detected_items = []
    rows = []
    draw_boxes = []
    draw_labels = []
    draw_scores = []

    if len(scores) > 0:
        indices = np.argsort(scores)[::-1]
        boxes = boxes[indices]
        scores = scores[indices]
        labels_tensor = labels_tensor[indices]

    for box, score, label_id in zip(boxes, scores, labels_tensor):
        score = float(score)
        label_id = int(label_id)

        if score < min_confidence:
            continue

        if label_id <= 0 or label_id > len(CLASSES):
            continue

        label = CLASSES[label_id - 1]

        rows.append(
            {
                "Detected Item": label,
                "Confidence": round(score, 4),
                "Billable": "Yes" if label in PRICE_LIST else "No",
            }
        )

        if label in PRICE_LIST:
            detected_items.append(label)

        draw_boxes.append(box)
        draw_labels.append(label)
        draw_scores.append(score)

    rendered = draw_boxes_pil(image_np, draw_boxes, draw_labels, draw_scores)
    df = pd.DataFrame(rows)
    return rendered, detected_items, df


def detect_objects(image_np, model_choice, min_confidence):
    model = get_selected_model(model_choice)

    if model_choice == "YOLO":
        return detect_with_yolo(model, image_np, min_confidence)
    if model_choice == "Faster R-CNN":
        return detect_with_frcnn(model, image_np, min_confidence)
    return detect_with_ssd(model, image_np, min_confidence)


def summarize_detection_result(model_name, df, detected_items, total_price, elapsed_time):
    if df is None or df.empty:
        return {
            "Model": model_name,
            "Detected Objects": 0,
            "Billable Items": 0,
            "Avg Confidence": 0.0,
            "Max Confidence": 0.0,
            "Inference Time (s)": round(elapsed_time, 4),
            "Estimated Total (RM)": round(total_price, 2),
        }

    avg_conf = float(df["Confidence"].mean()) if "Confidence" in df.columns else 0.0
    max_conf = float(df["Confidence"].max()) if "Confidence" in df.columns else 0.0

    return {
        "Model": model_name,
        "Detected Objects": int(len(df)),
        "Billable Items": int(len(detected_items)),
        "Avg Confidence": round(avg_conf, 4),
        "Max Confidence": round(max_conf, 4),
        "Inference Time (s)": round(elapsed_time, 4),
        "Estimated Total (RM)": round(total_price, 2),
    }


def plot_bar_chart(df, x_col, y_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)


def compare_all_models(image_source, min_confidence):
    image = Image.open(image_source).convert("RGB")
    image_np = np.array(image)

    model_names = ["YOLO", "Faster R-CNN", "SSD"]
    comparison_rows = []
    rendered_results = {}

    for model_name in model_names:
        start_time = time.perf_counter()

        rendered_img, detected_items, df = detect_objects(
            image_np, model_name, min_confidence
        )

        elapsed_time = time.perf_counter() - start_time
        bill_rows, total_price = calculate_bill(detected_items)

        summary = summarize_detection_result(
            model_name=model_name,
            df=df,
            detected_items=detected_items,
            total_price=total_price,
            elapsed_time=elapsed_time,
        )

        comparison_rows.append(summary)
        rendered_results[model_name] = {
            "image": rendered_img,
            "df": df,
            "bill_rows": bill_rows,
            "total_price": total_price,
            "detected_items": detected_items,
        }

    comparison_df = pd.DataFrame(comparison_rows)

    st.markdown("## 📊 Object Detection Model Comparison")
    st.caption(
        "Compare YOLO, Faster R-CNN, and SSD on the same uploaded image using detected object count, confidence score, inference time, and billing result."
    )

    best_conf_model = comparison_df.loc[comparison_df["Avg Confidence"].idxmax(), "Model"]
    fastest_model = comparison_df.loc[comparison_df["Inference Time (s)"].idxmin(), "Model"]
    most_detected_model = comparison_df.loc[comparison_df["Detected Objects"].idxmax(), "Model"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Avg Confidence", best_conf_model)
    c2.metric("Fastest Model", fastest_model)
    c3.metric("Most Detected Objects", most_detected_model)

    st.markdown("### 🖼️ Detection Result Comparison")
    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.image(rendered_results["YOLO"]["image"], caption="YOLO", use_container_width=True)
    with img_col2:
        st.image(rendered_results["Faster R-CNN"]["image"], caption="Faster R-CNN", use_container_width=True)
    with img_col3:
        st.image(rendered_results["SSD"]["image"], caption="SSD", use_container_width=True)

    st.markdown("### 📋 Comparison Table")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("### 📈 Visual Comparison")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        plot_bar_chart(
            comparison_df,
            "Model",
            "Avg Confidence",
            "Average Confidence by Model",
            "Avg Confidence"
        )

    with chart_col2:
        plot_bar_chart(
            comparison_df,
            "Model",
            "Inference Time (s)",
            "Inference Time by Model",
            "Seconds"
        )

    with chart_col3:
        plot_bar_chart(
            comparison_df,
            "Model",
            "Detected Objects",
            "Detected Objects by Model",
            "Object Count"
        )

    st.markdown("### 🏆 Quick Comparison Summary")
    st.success(
        f"{best_conf_model} achieved the highest average confidence. "
        f"{fastest_model} was the fastest model. "
        f"{most_detected_model} detected the most objects."
    )

    detail_tab1, detail_tab2, detail_tab3 = st.tabs(
        ["YOLO Details", "Faster R-CNN Details", "SSD Details"]
    )

    with detail_tab1:
        st.markdown("### YOLO Detection Data")
        if not rendered_results["YOLO"]["df"].empty:
            st.dataframe(rendered_results["YOLO"]["df"], use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected by YOLO.")
        render_bill_section(
            rendered_results["YOLO"]["bill_rows"],
            rendered_results["YOLO"]["total_price"]
        )

    with detail_tab2:
        st.markdown("### Faster R-CNN Detection Data")
        if not rendered_results["Faster R-CNN"]["df"].empty:
            st.dataframe(rendered_results["Faster R-CNN"]["df"], use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected by Faster R-CNN.")
        render_bill_section(
            rendered_results["Faster R-CNN"]["bill_rows"],
            rendered_results["Faster R-CNN"]["total_price"]
        )

    with detail_tab3:
        st.markdown("### SSD Detection Data")
        if not rendered_results["SSD"]["df"].empty:
            st.dataframe(rendered_results["SSD"]["df"], use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected by SSD.")
        render_bill_section(
            rendered_results["SSD"]["bill_rows"],
            rendered_results["SSD"]["total_price"]
        )


def process_image(image_source, model_choice, min_confidence, source_label):
    image = Image.open(image_source).convert("RGB")
    image_np = np.array(image)

    rendered_img, detected_items, df = detect_objects(
        image_np, model_choice, min_confidence
    )
    bill_rows, total_price = calculate_bill(detected_items)

    render_summary_cards(total_price, bill_rows, df)

    tab1, tab2, tab3 = st.tabs(["🖼️ Image Preview", "🧾 Billing", "📊 Detection Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {source_label}")
            st.image(image, use_container_width=True)
        with col2:
            st.markdown("### Detection Result")
            st.image(rendered_img, use_container_width=True)

    with tab2:
        render_bill_section(bill_rows, total_price)

    with tab3:
        st.markdown("### Detection Table")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No objects were detected.")


def process_multiple_images(image_files, model_choice, min_confidence):
    total_detected_objects = 0
    total_billable_items = 0
    grand_total_price = 0.0

    st.markdown("## 🖼️ Multi-Image Detection Results")

    for idx, image_file in enumerate(image_files, start=1):
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)

        rendered_img, detected_items, df = detect_objects(
            image_np, model_choice, min_confidence
        )
        bill_rows, total_price = calculate_bill(detected_items)

        total_detected_objects += len(df) if not df.empty else 0
        total_billable_items += sum(row["Quantity"] for row in bill_rows) if bill_rows else 0
        grand_total_price += total_price

        with st.expander(f"Image {idx}: {image_file.name}", expanded=(idx == 1)):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("### Detection Result")
                st.image(rendered_img, use_container_width=True)

            tab1, tab2 = st.tabs(["Detection Data", "Billing"])

            with tab1:
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No objects were detected.")

            with tab2:
                render_bill_section(bill_rows, total_price)

    st.markdown("## 📦 Overall Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Uploaded Images", len(image_files))
    c2.metric("Total Detected Objects", total_detected_objects)
    c3.metric("Grand Total", f"RM {grand_total_price:.2f}")

    st.caption(f"Total billable items across all images: {total_billable_items}")


def compare_all_models_multiple_images(image_files, min_confidence):
    model_names = ["YOLO", "Faster R-CNN", "SSD"]
    comparison_rows = []

    # ===== overall metrics across all uploaded images =====
    for model_name in model_names:
        total_detected_objects = 0
        total_billable_items = 0
        total_price = 0.0
        confidence_values = []
        total_elapsed = 0.0

        for image_file in image_files:
            image = Image.open(image_file).convert("RGB")
            image_np = np.array(image)

            start_time = time.perf_counter()
            rendered_img, detected_items, df = detect_objects(
                image_np, model_name, min_confidence
            )
            elapsed_time = time.perf_counter() - start_time

            bill_rows, image_total_price = calculate_bill(detected_items)

            total_elapsed += elapsed_time
            total_detected_objects += len(df) if not df.empty else 0
            total_billable_items += sum(row["Quantity"] for row in bill_rows) if bill_rows else 0
            total_price += image_total_price

            if df is not None and not df.empty and "Confidence" in df.columns:
                confidence_values.extend(df["Confidence"].tolist())

        avg_conf = float(np.mean(confidence_values)) if confidence_values else 0.0
        max_conf = float(np.max(confidence_values)) if confidence_values else 0.0

        comparison_rows.append({
            "Model": model_name,
            "Images Tested": len(image_files),
            "Detected Objects": total_detected_objects,
            "Billable Items": total_billable_items,
            "Avg Confidence": round(avg_conf, 4),
            "Max Confidence": round(max_conf, 4),
            "Inference Time (s)": round(total_elapsed, 4),
            "Estimated Total (RM)": round(total_price, 2),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.markdown("## 📊 Multi-Image Model Comparison")
    st.caption(
        "Compare YOLO, Faster R-CNN, and SSD across multiple uploaded images using confidence, detection count, inference time, billing result, and visual detection quality."
    )

    best_conf_model = comparison_df.loc[comparison_df["Avg Confidence"].idxmax(), "Model"]
    fastest_model = comparison_df.loc[comparison_df["Inference Time (s)"].idxmin(), "Model"]
    most_detected_model = comparison_df.loc[comparison_df["Detected Objects"].idxmax(), "Model"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Avg Confidence", best_conf_model)
    c2.metric("Fastest Model", fastest_model)
    c3.metric("Most Detected Objects", most_detected_model)

    # ===== keep visual comparison feature =====
    st.markdown("### 🖼️ Detection Result Comparison")

    selected_image_name = st.selectbox(
        "Choose one uploaded image for side-by-side model comparison",
        [file.name for file in image_files],
        key="compare_image_selector"
    )

    selected_file = next(file for file in image_files if file.name == selected_image_name)
    selected_image = Image.open(selected_file).convert("RGB")
    selected_image_np = np.array(selected_image)

    visual_results = {}
    for model_name in model_names:
        rendered_img, detected_items, df = detect_objects(
            selected_image_np, model_name, min_confidence
        )
        visual_results[model_name] = rendered_img

    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.image(visual_results["YOLO"], caption="YOLO", use_container_width=True)
    with img_col2:
        st.image(visual_results["Faster R-CNN"], caption="Faster R-CNN", use_container_width=True)
    with img_col3:
        st.image(visual_results["SSD"], caption="SSD", use_container_width=True)

    st.markdown("### 📋 Comparison Table")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("### 📈 Visual Comparison")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        plot_bar_chart(
            comparison_df,
            "Model",
            "Avg Confidence",
            "Average Confidence by Model",
            "Avg Confidence"
        )

    with chart_col2:
        plot_bar_chart(
            comparison_df,
            "Model",
            "Inference Time (s)",
            "Inference Time by Model",
            "Seconds"
        )

    with chart_col3:
        plot_bar_chart(
            comparison_df,
            "Model",
            "Detected Objects",
            "Detected Objects by Model",
            "Object Count"
        )

    st.markdown("### 🏆 Quick Comparison Summary")
    st.success(
        f"{best_conf_model} achieved the highest average confidence. "
        f"{fastest_model} was the fastest model. "
        f"{most_detected_model} detected the most objects."
    )

# =========================
# UI RENDER
# =========================
def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="status-pill">AI-powered Smart Retail Demo</div>
            <div class="hero-title">Smart Retail Checkout System</div>
            <div class="hero-subtitle">
                Upload product images or use your webcam to detect retail items, compare deep learning models,
                and generate an automatic checkout summary with total pricing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_price_list():
    st.sidebar.markdown("## 🧾 Product Price List")
    price_df = pd.DataFrame(
        [{"Item": k.title(), "Price (RM)": f"{v:.2f}"} for k, v in PRICE_LIST.items()]
    )
    st.sidebar.dataframe(price_df, use_container_width=True, hide_index=True)


def render_sidebar_controls():
    st.sidebar.markdown("## ⚙️ Input Settings")

    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["YOLO", "Faster R-CNN", "SSD"]
    )

    input_mode = st.sidebar.radio(
        "Choose image source",
        ["Upload Image", "Webcam Snapshot"],
        label_visibility="visible",
    )

    if model_choice == "YOLO":
        default_conf = 0.25
    elif model_choice == "Faster R-CNN":
        default_conf = 0.30
    else:
        default_conf = 0.10

    min_confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.90,
        value=float(default_conf),
        step=0.05,
    )

    st.sidebar.info(
        "Tip: use clear images with good lighting so the detection boxes and billing results look more accurate."
    )
    return model_choice, input_mode, min_confidence


def render_summary_cards(total_price, bill_rows, df):
    detected_count = len(df) if not df.empty else 0
    billable_count = sum(row["Quantity"] for row in bill_rows) if bill_rows else 0
    unique_billable = len(bill_rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Detected Objects", detected_count)
    c2.metric("Billable Items", billable_count)
    c3.metric("Estimated Total", f"RM {total_price:.2f}")

    if unique_billable:
        st.caption(f"Unique billable products: {unique_billable}")


def render_bill_section(bill_rows, total_price):
    st.markdown("### 🧾 Checkout Summary")

    if bill_rows:
        bill_df = pd.DataFrame(bill_rows)
        for col in ["Unit Price (RM)", "Subtotal (RM)"]:
            bill_df[col] = bill_df[col].map(lambda x: f"{x:.2f}")
        st.dataframe(bill_df, use_container_width=True, hide_index=True)
        st.success(f"Total Price: RM {total_price:.2f}")
    else:
        st.warning("No billable items detected in the current image.")

# =========================
# APP
# =========================
render_header()
model_choice, input_mode, min_confidence = render_sidebar_controls()
render_price_list()

info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.markdown(
        f'<div class="mini-card"><div class="label-text">Model</div><div class="value-text">{model_choice}</div></div>',
        unsafe_allow_html=True,
    )
with info_col2:
    st.markdown(
        '<div class="mini-card"><div class="label-text">Function</div><div class="value-text">Auto Billing</div></div>',
        unsafe_allow_html=True,
    )
with info_col3:
    st.markdown(
        f'<div class="mini-card"><div class="label-text">Device</div><div class="value-text">{str(DEVICE).upper()}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-card'>", unsafe_allow_html=True)

try:
    get_selected_model(model_choice)

    if input_mode == "Upload Image":
        uploaded_files = st.file_uploader(
            "Upload at least 3 product images",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            accept_multiple_files=True,
        )

        if uploaded_files and len(uploaded_files) > 0:
            st.success(f"Uploaded {len(uploaded_files)} file(s).")

            for file in uploaded_files:
                st.write(f"• {file.name}")

            if len(uploaded_files) < 3:
                st.warning("Please upload at least 3 images to continue.")
            else:
                btn_col1, btn_col2 = st.columns(2)

                with btn_col1:
                    run_single = st.button("Run Selected Model", use_container_width=True)

                with btn_col2:
                    run_compare = st.button("Compare All 3 Models", use_container_width=True)

                if run_single:
                    process_multiple_images(
                        uploaded_files,
                        model_choice,
                        min_confidence,
                    )

                if run_compare:
                    compare_all_models_multiple_images(
                        uploaded_files,
                        min_confidence,
                    )

        else:
            st.info("Upload at least 3 images to start the smart checkout demo.")
    else:
        camera_image = st.camera_input("Take a picture for smart checkout")
        if camera_image is not None:
            process_image(
                camera_image,
                model_choice,
                min_confidence,
                "Captured Image",
            )
        else:
            st.info("Use your webcam to capture an image and preview the checkout result.")

except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.info("Make sure your model paths are correct for YOLO, Faster R-CNN, and SSD.")
except RuntimeError as e:
    st.error(f"RuntimeError while loading model: {e}")
    st.info("This usually means your saved .pth structure or num_classes does not match the current model.")
except Exception as e:
    st.error(f"Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
