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
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
/* ===== Global text visibility fix ===== */
html, body, [class*="css"], [data-testid="stAppViewContainer"], 
[data-testid="stSidebar"], [data-testid="stHeader"] {
    color: #111827 !important;
}

/* Main app background */
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    color: #111827 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stSlider label {
    color: #f8fafc !important;
}

/* Block container spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hero section */
.hero-box {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #38bdf8 100%);
    padding: 28px 30px;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.20);
    margin-bottom: 1.25rem;
}

.hero-title {
    color: white !important;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    color: rgba(255,255,255,0.92) !important;
    font-size: 1rem;
    margin-top: 8px;
    margin-bottom: 0;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 22px 20px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    text-align: center;
    min-height: 130px;
}

.metric-label {
    font-size: 0.95rem;
    color: #475569 !important;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 2rem;
    color: #0f172a !important;
    font-weight: 800;
    margin: 0;
}

/* Section card */
.section-card {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

/* Receipt box */
.receipt-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
}

.receipt-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a !important;
    margin-bottom: 12px;
}

.receipt-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px dashed #cbd5e1;
    color: #1e293b !important;
    font-size: 0.98rem;
}

.receipt-total {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 2px solid #0f172a;
    font-size: 1.12rem;
    font-weight: 800;
    color: #0f172a !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: #e2e8f0;
    color: #0f172a !important;
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: white !important;
}

/* Streamlit alerts */
.stSuccess, .stInfo, .stWarning, .stAlert {
    border-radius: 14px;
}

/* Dataframe container */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

/* File uploader / camera box */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"] {
    background: rgba(255,255,255,0.75);
    border: 1px solid #dbeafe;
    border-radius: 16px;
    padding: 8px;
}

/* Subheaders */
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)

# =========================
# PRICE LIST
# =========================
PRICE_LIST = {
    "bottle": 3.50,
    "cup": 2.00,
    "banana": 1.50,
    # "apple": 2.00,
    # "orange": 3.00,
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
        bill_rows.append({
            "item": item,
            "qty": qty,
            "unit_price": unit_price,
            "subtotal": subtotal
        })

    return bill_rows, total_price


def detect_objects(image, conf_threshold=0.50):
    results = model(image, conf=conf_threshold)
    result = results[0]

    rendered = result.plot()
    detected_items = []
    rows = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = model.names[cls_id]

        rows.append({
            "name": label,
            "confidence": round(conf, 4)
        })

        if label in PRICE_LIST:
            detected_items.append(label)

    df = pd.DataFrame(rows)
    return rendered, detected_items, df


def show_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_receipt(bill_rows, total_price):
    if not bill_rows:
        st.warning("No billable items detected.")
        return

    receipt_html = """
    <div class="receipt-box">
        <div class="receipt-title">🧾 Billing Summary</div>
    """

    for row in bill_rows:
        receipt_html += f"""
        <div class="receipt-row">
            <div><strong>{row['item'].title()}</strong> × {row['qty']}</div>
            <div>RM {row['subtotal']:.2f}</div>
        </div>
        """

    receipt_html += f"""
        <div class="receipt-total">
            <div>Total Price</div>
            <div>RM {total_price:.2f}</div>
        </div>
    </div>
    """

    st.markdown(receipt_html, unsafe_allow_html=True)


# =========================
# HERO HEADER
# =========================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🛒 Smart Retail Checkout System</div>
    <div class="hero-subtitle">
        Detect retail objects automatically and calculate the estimated bill in real time.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.write("Configure input method and detection settings.")

option = st.sidebar.radio(
    "Choose input type",
    ["Upload Image", "Webcam Snapshot"]
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("### 💰 Price List")
price_df = pd.DataFrame(
    [{"Item": k.title(), "Price (RM)": f"{v:.2f}"} for k, v in PRICE_LIST.items()]
)
st.sidebar.table(price_df)

st.sidebar.markdown("---")
st.sidebar.info(
    "Tip: Increase the confidence threshold to reduce false detections."
)

# =========================
# INPUT
# =========================
image = None
image_label = "Original Image"

input_col1, input_col2 = st.columns([1.25, 1])

with input_col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Input Section")

    if option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_label = "Original Image"

    elif option == "Webcam Snapshot":
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")
            image_label = "Captured Image"

    st.markdown('</div>', unsafe_allow_html=True)

with input_col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ System Overview")
    st.write("This prototype uses YOLO object detection to identify products and estimate the total price automatically.")
    st.write("Supported billable items in this demo:")
    for item, price in PRICE_LIST.items():
        st.write(f"- **{item.title()}** — RM {price:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MAIN PROCESS
# =========================
if image is not None:
    image_np = np.array(image)

    rendered_img, detected_items, df = detect_objects(image_np, conf_threshold)
    bill_rows, total_price = calculate_bill(detected_items)

    total_detected_objects = len(df)
    total_billable_quantity = sum(row["qty"] for row in bill_rows) if bill_rows else 0
    unique_billable_items = len(bill_rows)

    # =========================
    # METRICS
    # =========================
    st.markdown("### 📊 Detection Overview")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        show_metric_card("Total Detected Objects", total_detected_objects)
    with m2:
        show_metric_card("Billable Quantity", total_billable_quantity)
    with m3:
        show_metric_card("Unique Billable Items", unique_billable_items)
    with m4:
        show_metric_card("Estimated Total", f"RM {total_price:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["🖼 Image Preview", "🧾 Billing Summary", "📋 Detection Table"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader(image_label)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Detection Result")
            st.image(rendered_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        render_receipt(bill_rows, total_price)

        if bill_rows:
            st.success(f"Checkout completed successfully. Total payable amount: RM {total_price:.2f}")

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Detection Data")

        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No objects detected.")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload an image or take a picture to start the smart checkout process.")
