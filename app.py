import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from ultralytics import YOLO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="FruitScan AI",
    page_icon="🍎",
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
        /* ========= App Background ========= */
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* ========= Global Text ========= */
        h1, h2, h3, h4, h5, h6,
        p, span, label, div, small {
            color: #0f172a;
        }

        /* ========= Hero ========= */
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
            padding: 30px 32px;
            border-radius: 24px;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.16);
            margin-bottom: 1.2rem;
        }

        .hero-card * {
            color: white !important;
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.95;
        }

        .status-pill {
            display: inline-block;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.28);
            color: white !important;
            margin-bottom: 0.8rem;
        }

        /* ========= Cards ========= */
        .section-card {
            background: rgba(255,255,255,0.82);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 24px;
            padding: 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
        }

        .mini-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-radius: 20px;
            padding: 18px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }

        .label-text {
            font-size: 0.85rem;
            color: #64748b !important;
        }

        .value-text {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0f172a !important;
        }

        /* ========= Sidebar ========= */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.05);
        }

        section[data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        /* ========= Metrics ========= */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 18px;
            padding: 14px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetric"] * {
            color: #0f172a !important;
        }

        /* ========= Tabs ========= */
        .stTabs [data-baseweb="tab"] {
            background: #e8eefc;
            border-radius: 12px;
            padding: 0.55rem 1rem;
            color: #0f172a !important;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: #2563eb !important;
            color: white !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
        }

        .stTabs [aria-selected="true"] * {
            color: white !important;
        }

        /* ========= Upload Image ========= */
        [data-testid="stFileUploader"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
            border: 1px solid #dbe7ff !important;
            border-radius: 22px !important;
            padding: 16px !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }

        [data-testid="stFileUploader"] section {
            background: transparent !important;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small {
            color: #0f172a !important;
            font-weight: 600;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%) !important;
            border: 2px dashed #bfd4ff !important;
            border-radius: 18px !important;
            padding: 18px !important;
            transition: all 0.2s ease-in-out;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3b82f6 !important;
            background: #f4f8ff !important;
        }

        [data-testid="stFileUploaderFile"] {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
            padding: 6px 10px !important;
        }

        [data-testid="stFileUploaderFile"] * {
            color: white !important;
        }

        [data-testid="stFileUploaderFile"] button {
            background: rgba(255,255,255,0.14) !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            border-radius: 999px !important;
            width: 28px !important;
            height: 28px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            color: white !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        [data-testid="stFileUploaderFile"] button:hover {
            background: #ef4444 !important;
            border-color: #ef4444 !important;
            transform: scale(1.05);
        }

        [data-testid="stFileUploaderFile"] button svg {
            fill: white !important;
            color: white !important;
            width: 16px !important;
            height: 16px !important;
        }

        [data-testid="stFileUploader"] section button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.55rem 1rem !important;
            font-weight: 700 !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.25);
        }

        [data-testid="stFileUploader"] section button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        }

        [data-testid="stFileUploader"] svg {
            color: #2563eb !important;
            fill: #2563eb !important;
        }

        /* ========= Camera ========= */
        [data-testid="stCameraInput"] {
            background: #ffffff !important;
            border-radius: 16px;
            padding: 10px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stCameraInput"] * {
            color: #0f172a !important;
        }

        [data-testid="stCameraInput"] button {
            background: #0f172a !important;
            color: white !important;
            border-radius: 12px !important;
        }

        [data-testid="stCameraInput"] span {
            color: white !important;
        }

        /* ========= Dataframe ========= */
        [data-testid="stDataFrame"] {
            border-radius: 16px !important;
            overflow: hidden !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stDataFrame"] div {
            color: #0f172a !important;
        }

        /* ========= Alerts ========= */
        .stAlert, .stInfo, .stSuccess, .stWarning {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_custom_css()

# =========================
# PRICE LIST
# =========================
PRICE_LIST = {
    "apple": 2.50,
    "banana": 1.50,
    "orange": 2.00,
    "mango": 4.00,
    "pineapple": 6.50,
    "watermelon": 12.00,
}

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


model = load_model()

# =========================
# FUNCTIONS
# =========================
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


def detect_objects(image, min_confidence=0.25):
    results = model(image, conf=min_confidence)
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
                "Detected Item": label.title(),
                "Confidence": round(conf, 4),
                "Billable": "Yes" if label in PRICE_LIST else "No",
            }
        )

        if label in PRICE_LIST:
            detected_items.append(label)

    df = pd.DataFrame(rows)
    return rendered, detected_items, df


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="status-pill">AI-powered Fruit Checkout Demo</div>
            <div class="hero-title">FruitScan AI</div>
            <div class="hero-subtitle">
                Upload at least 3 fruit images or use your webcam to detect fruits,
                preview results, and generate one combined billing summary automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_price_list():
    st.sidebar.markdown("## 🧾 Fruit Price List")
    price_df = pd.DataFrame(
        [{"Item": k.title(), "Price (RM)": f"{v:.2f}"} for k, v in PRICE_LIST.items()]
    )
    st.sidebar.dataframe(price_df, use_container_width=True, hide_index=True)


def render_sidebar_controls():
    st.sidebar.markdown("## ⚙️ Input Settings")
    input_mode = st.sidebar.radio(
        "Choose image source",
        ["Upload Images", "Webcam Snapshot"],
    )
    min_confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    st.sidebar.info(
        "Tip: use clear images with good lighting so the fruit detection and billing result are more accurate."
    )
    return input_mode, min_confidence


def render_summary_cards(total_price, bill_rows, full_df):
    detected_count = len(full_df) if not full_df.empty else 0
    billable_count = sum(row["Quantity"] for row in bill_rows) if bill_rows else 0
    unique_billable = len(bill_rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Detected Objects", detected_count)
    c2.metric("Billable Fruits", billable_count)
    c3.metric("Estimated Total", f"RM {total_price:.2f}")

    if unique_billable:
        st.caption(f"Unique billable fruit types: {unique_billable}")


def render_bill_section(bill_rows, total_price):
    st.markdown("### 🧾 Combined Checkout Summary")

    if bill_rows:
        bill_df = pd.DataFrame(bill_rows)
        for col in ["Unit Price (RM)", "Subtotal (RM)"]:
            bill_df[col] = bill_df[col].map(lambda x: f"{x:.2f}")
        st.dataframe(bill_df, use_container_width=True, hide_index=True)
        st.success(f"Total Price: RM {total_price:.2f}")
    else:
        st.warning("No billable fruits detected in the current input.")


def process_multiple_images(uploaded_files, min_confidence):
    all_detected_items = []
    all_detection_rows = []
    image_results = []

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        rendered_img, detected_items, df = detect_objects(image_np, min_confidence)

        all_detected_items.extend(detected_items)

        if not df.empty:
            df["Source Image"] = f"Image {i}"
            all_detection_rows.append(df)

        image_results.append(
            {
                "index": i,
                "original_image": image,
                "rendered_image": rendered_img,
            }
        )

    full_df = pd.concat(all_detection_rows, ignore_index=True) if all_detection_rows else pd.DataFrame()
    bill_rows, total_price = calculate_bill(all_detected_items)

    render_summary_cards(total_price, bill_rows, full_df)

    tab1, tab2, tab3 = st.tabs(["🖼️ Image Preview", "🧾 Billing", "📊 Detection Data"])

    with tab1:
        for result in image_results:
            st.markdown(f"## Fruit Image {result['index']}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Original Image")
                st.image(result["original_image"], use_container_width=True)
            with col2:
                st.markdown("### Detection Result")
                st.image(result["rendered_image"], use_container_width=True)

    with tab2:
        render_bill_section(bill_rows, total_price)

    with tab3:
        st.markdown("### Combined Detection Table")
        if not full_df.empty:
            st.dataframe(full_df, use_container_width=True, hide_index=True)
        else:
            st.info("No objects were detected.")


def process_single_camera_image(image_source, min_confidence, source_label):
    image = Image.open(image_source).convert("RGB")
    image_np = np.array(image)

    rendered_img, detected_items, df = detect_objects(image_np, min_confidence)
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
# APP LAYOUT
# =========================
render_header()
input_mode, min_confidence = render_sidebar_controls()
render_price_list()

info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.markdown(
        '<div class="mini-card"><div class="label-text">Model</div><div class="value-text">YOLOv8</div></div>',
        unsafe_allow_html=True,
    )
with info_col2:
    st.markdown(
        '<div class="mini-card"><div class="label-text">Function</div><div class="value-text">Fruit Billing</div></div>',
        unsafe_allow_html=True,
    )
with info_col3:
    st.markdown(
        '<div class="mini-card"><div class="label-text">Input</div><div class="value-text">Multi Image / Camera</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-card'>", unsafe_allow_html=True)

if input_mode == "Upload Images":
    uploaded_files = st.file_uploader(
        "Upload at least 3 fruit images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_files:
        if len(uploaded_files) < 3:
            st.warning("Please upload at least 3 fruit images.")
        else:
            st.success(f"{len(uploaded_files)} images uploaded successfully.")
            process_multiple_images(uploaded_files, min_confidence)
    else:
        st.info("Upload at least 3 fruit images to start the fruit checkout demo.")

else:
    camera_image = st.camera_input("Take a fruit picture")
    if camera_image is not None:
        process_single_camera_image(camera_image, min_confidence, "Captured Image")
    else:
        st.info("Use your webcam to capture a fruit image and preview the billing result.")

st.markdown("</div>", unsafe_allow_html=True)
