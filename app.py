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

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
            color: white;
            padding: 28px 30px;
            border-radius: 22px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.95;
            line-height: 1.6;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid rgba(37, 99, 235, 0.12);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .mini-card {
            background: white;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }

        .label-text {
            font-size: 0.88rem;
            color: #475569;
            margin-bottom: 0.3rem;
        }

        .value-text {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0f172a;
        }

        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.28);
            font-size: 0.86rem;
            margin-bottom: 0.8rem;
        }

        .bill-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 14px;
            border-radius: 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.55rem 1rem;
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
    "bottle": 3.50,
    "cup": 2.00,
    "banana": 1.50,
}

# =========================
# LOAD MODEL
# =========================
@st.cache_resource

def load_model():
    return YOLO("yolov8n.pt")


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
                "Detected Item": label,
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
            <div class="status-pill">AI-powered Smart Retail Demo</div>
            <div class="hero-title">Smart Retail Checkout System</div>
            <div class="hero-subtitle">
                Upload an image or use your webcam to detect retail items, preview object detection,
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
    input_mode = st.sidebar.radio(
        "Choose image source",
        ["Upload Image", "Webcam Snapshot"],
        label_visibility="visible",
    )
    min_confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    st.sidebar.info(
        "Tip: use a clear image with good lighting so the detection boxes and bill summary look more accurate."
    )
    return input_mode, min_confidence


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
    st.markdown('<div class="bill-title">🧾 Checkout Summary</div>', unsafe_allow_html=True)

    if bill_rows:
        bill_df = pd.DataFrame(bill_rows)
        for col in ["Unit Price (RM)", "Subtotal (RM)"]:
            bill_df[col] = bill_df[col].map(lambda x: f"{x:.2f}")
        st.dataframe(bill_df, use_container_width=True, hide_index=True)
        st.success(f"Total Price: RM {total_price:.2f}")
    else:
        st.warning("No billable items detected in the current image.")


def process_image(image_source, min_confidence, source_label):
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
        '<div class="mini-card"><div class="label-text">Function</div><div class="value-text">Auto Billing</div></div>',
        unsafe_allow_html=True,
    )
with info_col3:
    st.markdown(
        '<div class="mini-card"><div class="label-text">Input</div><div class="value-text">Image / Camera</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-card'>", unsafe_allow_html=True)

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )
    if uploaded_file is not None:
        process_image(uploaded_file, min_confidence, "Original Image")
    else:
        st.info("Upload an image to start the smart checkout demo.")

else:
    camera_image = st.camera_input("Take a picture for smart checkout")
    if camera_image is not None:
        process_image(camera_image, min_confidence, "Captured Image")
    else:
        st.info("Use your webcam to capture an image and preview the checkout result.")

st.markdown("</div>", unsafe_allow_html=True)
