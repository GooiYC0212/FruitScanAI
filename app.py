import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from ultralytics import YOLO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Retail Checkout", layout="wide")

# =========================
# CSS (FIX WHITE TEXT + BEAUTIFY)
# =========================
st.markdown("""
<style>

/* 全局字体 */
html, body, [class*="css"] {
    color: #111 !important;
}

/* 卡片样式 */
.metric-card {
    background: #f5f5f5;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}

/* 标题 */
h1, h2, h3 {
    color: #111 !important;
}

/* tabs */
.stTabs [data-baseweb="tab"] {
    color: #111 !important;
    font-weight: 600;
}

/* sidebar */
section[data-testid="stSidebar"] {
    color: #111 !important;
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


def detect_objects(image, conf_threshold):
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


# =========================
# HEADER
# =========================
st.title("🛒 Smart Retail Checkout System")
st.caption("AI-powered object detection with automatic billing")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

option = st.sidebar.radio("Input Type", ["Upload Image", "Webcam"])

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6)

st.sidebar.markdown("### 💰 Price List")
price_df = pd.DataFrame(list(PRICE_LIST.items()), columns=["Item", "Price (RM)"])
st.sidebar.table(price_df)

# =========================
# MAIN
# =========================
image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# =========================
# PROCESS
# =========================
if image is not None:
    image_np = np.array(image)

    rendered_img, detected_items, df = detect_objects(image_np, conf_threshold)
    bill_rows, total_price = calculate_bill(detected_items)

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="metric-card">
        <h4>Total Detected Objects</h4>
        <h2>{len(df)}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
        <h4>Billable Items</h4>
        <h2>{len(bill_rows)}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
        <h4>Estimated Total</h4>
        <h2>RM {total_price:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["🖼 Image", "🧾 Billing", "📊 Detection Data"])

    with tab1:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with colB:
            st.subheader("Detected")
            st.image(rendered_img, use_container_width=True)

    with tab2:
        st.subheader("Billing Summary")

        if bill_rows:
            for row in bill_rows:
                st.write(
                    f"{row['item']} x{row['qty']} → RM {row['subtotal']:.2f}"
                )
            st.success(f"Total: RM {total_price:.2f}")
        else:
            st.warning("No billable items detected")

    with tab3:
        st.subheader("Detection Table")
        st.dataframe(df, use_container_width=True)

else:
    st.info("Please upload an image or take a picture to start.")
