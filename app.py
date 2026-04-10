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

        div[data-baseweb="select"] span {
            color: #0f172a !important;
        }

        ul[role="listbox"] {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        ul[role="listbox"] li,
        ul[role="listbox"] li * {
            color: #0f172a !important;
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
        }

        [data-testid="stFileUploaderFile"] {
            background: #0f172a !important;
            border-radius: 10px !important;
        }

        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploaderFile"] * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploaderFile"] button,
        [data-testid="stFileUploaderFile"] button * {
            color: #ffffff !important;
            fill: #ffffff !important;
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
YOLO_DRIVE_URL = "https://drive.google.com/uc?id=1onDCGuzge1fYdVtJifxHwUDk4Zhpgncg"
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
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        text = f"{label} {score:.2f}"

        # draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        # text size
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except:
            text_w = len(text) * 7
            text_h = 15

        # label background box
        text_x1 = x1
        text_y1 = max(0, y1 - text_h - 8)
        text_x2 = x1 + text_w + 8
        text_y2 = text_y1 + text_h + 6

        draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill="red")
        draw.text((text_x1 + 4, text_y1 + 3), text, fill="white", font=font)

    return np.array(image_pil)

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

    checkpoint = torch.load(FRCNN_MODEL_PATH, map_location=DEVICE)
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

    checkpoint = torch.load(SSD_MODEL_PATH, map_location=DEVICE)
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

def detect_with_frcnn(model, image_np, min_confidence, debug_mode=False):
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    labels_tensor = outputs["labels"].detach().cpu().numpy()

    threshold = min_confidence

    if debug_mode:
        st.write("Faster R-CNN raw scores:", scores[:10])
        st.write("Faster R-CNN raw labels:", labels_tensor[:10])

    detected_items = []
    rows = []
    draw_boxes = []
    draw_labels = []
    draw_scores = []

    for box, score, label_id in zip(boxes, scores, labels_tensor):
        if float(score) < threshold:
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

def detect_with_ssd(model, image_np, min_confidence, debug_mode=False):
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    labels_tensor = outputs["labels"].detach().cpu().numpy()

    if debug_mode:
        st.write("SSD raw scores:", scores[:20])
        st.write("SSD raw labels:", labels_tensor[:20])

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

def detect_objects(image_np, model_choice, min_confidence, debug_mode=False):
    model = get_selected_model(model_choice)

    if model_choice == "YOLO":
        return detect_with_yolo(model, image_np, min_confidence)
    if model_choice == "Faster R-CNN":
        return detect_with_frcnn(model, image_np, min_confidence, debug_mode)
    return detect_with_ssd(model, image_np, min_confidence, debug_mode)

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
                Upload an image or use your webcam to detect retail items, compare deep learning models,
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

    debug_mode = st.sidebar.checkbox("Show debug output", value=False)

    st.sidebar.info(
        "Tip: use a clear image with good lighting so the detection boxes and bill summary look more accurate."
    )
    return model_choice, input_mode, min_confidence, debug_mode

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

def process_image(image_source, model_choice, min_confidence, source_label, debug_mode=False):
    image = Image.open(image_source).convert("RGB")
    image_np = np.array(image)

    rendered_img, detected_items, df = detect_objects(
        image_np, model_choice, min_confidence, debug_mode
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

# =========================
# APP
# =========================
render_header()
model_choice, input_mode, min_confidence, debug_mode = render_sidebar_controls()
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
        uploaded_file = st.file_uploader(
            "Upload a product image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )
        if uploaded_file is not None:
            process_image(
                uploaded_file,
                model_choice,
                min_confidence,
                "Original Image",
                debug_mode,
            )
        else:
            st.info("Upload an image to start the smart checkout demo.")
    else:
        camera_image = st.camera_input("Take a picture for smart checkout")
        if camera_image is not None:
            process_image(
                camera_image,
                model_choice,
                min_confidence,
                "Captured Image",
                debug_mode,
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
