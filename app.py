import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="UAV Detection System", layout="wide")

# =========================================================
# USER LOGIN
# =========================================================
USERS = {
    "admin": "admin123",
    "uav": "uav123"
}

# =========================================================
# CLASS NAME MAP (RUSSIAN → ENGLISH)
# =========================================================
CLASS_MAP = {
    0: "Artillery",
    1: "BMP (Infantry Fighting Vehicle)",
    2: "UAV (Drone)",
    3: "Armored Vehicle",
    4: "BTR (APC)",
    5: "Infantry (Soldier)",
    6: "MLRS",
    7: "Tank"
}

WAR_CLASSES = list(CLASS_MAP.values())

# =========================================================
# SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

# =========================================================
# LOGIN PAGE WITH ALERT
# =========================================================
if not st.session_state.logged_in:
    st.title("🚁 UAV Login System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(username) == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Incorrect username or password")

    st.stop()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.write(f"👤 Logged in as: **{st.session_state.user}**")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.25, 0.05
)

mode = st.sidebar.radio(
    "Detection Mode",
    ["Upload Image", "Live Camera"]
)

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

model = load_model()

# =========================================================
# IMAGE UPLOAD DETECTION
# =========================================================
if mode == "Upload Image":

    st.title("🖼️ UAV Image Detection")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        results = model(img_np, conf=confidence)[0]
        draw = ImageDraw.Draw(image)

        detected_classes = []
        conf_scores = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            label = CLASS_MAP.get(cls_id, f"Class {cls_id}")
            detected_classes.append(label)
            conf_scores.append(conf)

            text = f"{label} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), text, fill="yellow")

        st.image(image, use_column_width=True)

        # =================================================
        # WAR FILTER MESSAGE
        # =================================================
        if not any(cls in WAR_CLASSES for cls in detected_classes):
            st.warning("⚠️ This is not a war-related item.")

        # =================================================
        # CONFIDENCE HISTOGRAM
        # =================================================
        if conf_scores:
            st.subheader("📊 Confidence Histogram")

            fig, ax = plt.subplots()
            ax.hist(conf_scores, bins=10)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # =================================================
        # LIME EXPLANATION
        # =================================================
        if st.button("Generate LIME Explanation"):
            st.info("Generating explanation...")

            def predict_fn(images):
                preds = []
                for img in images:
                    r = model(img, conf=confidence)[0]
                    if len(r.boxes) > 0:
                        conf = float(r.boxes.conf[0])
                        preds.append([1 - conf, conf])
                    else:
                        preds.append([1, 0])
                return np.array(preds)

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_np.astype("double"),
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=100
            )

            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False
            )

            lime_img = mark_boundaries(temp / 255.0, mask)
            st.image(lime_img, caption="LIME Explanation")

# =========================================================
# LIVE CAMERA DETECTION (LOCAL ONLY)
# =========================================================
elif mode == "Live Camera":

    st.title("📡 Live Camera Detection")

    start_cam = st.checkbox("Start Camera")
    frame_slot = st.empty()

    if start_cam:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Camera not accessible")
        else:
            while cap.isOpened() and start_cam:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb, conf=confidence)[0]

                img = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img)

                for box in results.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    label = CLASS_MAP.get(cls_id, f"Class {cls_id}")
                    text = f"{label} {conf:.2f}"

                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, max(0, y1 - 15)), text, fill="yellow")

                frame_slot.image(img, use_column_width=True)

            cap.release()

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("🚁 UAV Detection System | Image Upload + Live Camera")
