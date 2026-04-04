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
    model = YOLO("yolov8n.pt")
    return model

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


def detect_objects(image):
    results = model(image)
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
# TITLE
# =========================
st.title("Smart Retail Object Detection and Price Calculation System")
st.write("Detect objects and automatically calculate total price.")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose input type:", ["Upload Image", "Webcam Snapshot"])

st.sidebar.subheader("Price List")
for item, price in PRICE_LIST.items():
    st.sidebar.write(f"{item}: RM{price:.2f}")

# =========================
# MAIN
# =========================
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        rendered_img, detected_items, df = detect_objects(image_np)
        bill_rows, total_price = calculate_bill(detected_items)

        with col2:
            st.subheader("Detection Result")
            st.image(rendered_img, use_container_width=True)

        st.subheader("Bill Summary")
        if bill_rows:
            for row in bill_rows:
                st.write(
                    f"**{row['item']}** x{row['qty']} @ RM{row['unit_price']:.2f} = RM{row['subtotal']:.2f}"
                )
            st.success(f"Total Price = RM{total_price:.2f}")
        else:
            st.warning("No billable items detected.")

        with st.expander("Show Detection Table"):
            st.dataframe(df, use_container_width=True)

elif option == "Webcam Snapshot":
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Captured Image")
            st.image(image, use_container_width=True)

        rendered_img, detected_items, df = detect_objects(image_np)
        bill_rows, total_price = calculate_bill(detected_items)

        with col2:
            st.subheader("Detection Result")
            st.image(rendered_img, use_container_width=True)

        st.subheader("Bill Summary")
        if bill_rows:
            for row in bill_rows:
                st.write(
                    f"**{row['item']}** x{row['qty']} @ RM{row['unit_price']:.2f} = RM{row['subtotal']:.2f}"
                )
            st.success(f"Total Price = RM{total_price:.2f}")
        else:
            st.warning("No billable items detected.")

        with st.expander("Show Detection Table"):
            st.dataframe(df, use_container_width=True)
