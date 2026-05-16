import os
import tempfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "best.pt")
CONFIDENCE_THRESHOLD = 0.70


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def get_detected_labels(result):
    labels = []

    if result.boxes is None:
        return labels

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        label_name = f"{result.names[class_id]} ({confidence:.2f})"
        if label_name not in labels:
            labels.append(label_name)

    return labels


def has_valid_detection(result):
    if result.boxes is None:
        return False

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence >= CONFIDENCE_THRESHOLD:
            return True

    return False


def predict_image(model, uploaded_file):
    file_suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        results = model.predict(source=temp_path, conf=0.25)
        return results[0]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


st.set_page_config(page_title="Fresh or Stale Detector", layout="wide")

st.title("Fresh or Stale Fruit and Vegetable Detector")
st.write("Upload an image to check whether the fruit or vegetable is fresh or stale.")
st.caption(
    "Supported classes: apple, banana, capsicum, tomato, orange, and bitter gourd."
)

model = load_model()

uploaded_file = st.file_uploader(
    "Add an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(input_image, use_column_width=True)

    with st.spinner("Detecting result..."):
        result = predict_image(model, uploaded_file)
        detected_items = get_detected_labels(result)
        valid_detection = has_valid_detection(result)
        output_image = Image.fromarray(result.plot()) if valid_detection else None

    with col2:
        st.subheader("Detection Result")
        if output_image is not None:
            st.image(output_image, use_column_width=True)
        else:
            st.info("No supported fruit or vegetable detected in this image.")

    if detected_items:
        st.success("Detected result: " + ", ".join(detected_items))
    else:
        st.warning(
            "Out of model scope or low-confidence result. "
            "Please upload apple, banana, capsicum, tomato, orange, or bitter gourd."
        )
else:
    st.info("Please upload an image to start detection.")
