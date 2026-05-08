import streamlit as st
import cv2
import numpy as np

from src.edge_detection import detect_edges
from src.morphology import apply_morphology
from src.plate_localization import localize_plate
from src.preprocessing import preprocess_image, apply_frequency_highpass

st.title("License Plate Detection System")

st.sidebar.header("Pipeline Settings")

edge_method = st.sidebar.selectbox(
    "Edge Detection Method",
    ["canny", "sobel"],
    help="Compare Canny vs Sobel to see how each performs."
)

morph_method = st.sidebar.selectbox(
    "Morphological Operation",
    ["closing", "opening", "dilation", "erosion"]
)

use_freq_filter = st.sidebar.checkbox(
    "Apply Frequency-Domain High-Pass Filter",
    value=False,
    help="Runs a DFT-based high-pass filter before edge detection to suppress background noise."
)

freq_radius = st.sidebar.slider(
    "High-Pass Filter Radius", min_value=5, max_value=80, value=30,
    help="Larger radius → more low frequencies removed → stronger edge enhancement."
)
# ─────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    # ── preprocessing ────────────────────────────────────────────────────────
    resized, gray, blur = preprocess_image(external_image=image)

    # ADDED: optional frequency-domain step shown as its own stage
    if use_freq_filter:
        freq_filtered = apply_frequency_highpass(blur, radius=freq_radius)
        edge_input = freq_filtered

        st.subheader("Frequency-Domain High-Pass Filter")
        st.image(freq_filtered, caption="Low frequencies suppressed via DFT")
    else:
        edge_input = blur

    # ── edge detection ───────────────────────────────────────────────────────
    edges = detect_edges(edge_input, method=edge_method)

    st.subheader(f"Edge Detection — {edge_method.upper()}")
    st.image(edges, caption=f"Edges detected with {edge_method}")

    # ── morphology ───────────────────────────────────────────────────────────
    morph = apply_morphology(edges, method=morph_method)

    st.subheader(f"Morphological Operation — {morph_method.upper()}")
    st.image(morph, caption=f"After {morph_method}")

    # ── plate localisation ───────────────────────────────────────────────────
    plate, bbox = localize_plate(resized, morph)

    if plate is not None:
        # BUG FIX: convert (x,y,w,h) → (x1,y1,x2,y2) for display overlay
        x, y, bw, bh = bbox
        overlay = resized.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        st.subheader("Detected Plate Location")
        st.image(overlay, channels="BGR", caption="Green box = detected plate")

        st.subheader("Cropped Plate Region")
        st.image(plate, channels="BGR")
    else:
        st.warning("No license plate detected. Try adjusting the settings in the sidebar.")

    # ── ADDED: side-by-side technique comparison ──────────────────────────────
    st.markdown("---")
    st.subheader("Technique Comparison: Canny vs Sobel")

    col1, col2 = st.columns(2)

    canny_edges = detect_edges(edge_input, method="canny")
    sobel_edges = detect_edges(edge_input, method="sobel")

    canny_morph = apply_morphology(canny_edges, method=morph_method)
    sobel_morph = apply_morphology(sobel_edges, method=morph_method)

    _, canny_bbox = localize_plate(resized, canny_morph)
    _, sobel_bbox = localize_plate(resized, sobel_morph)

    with col1:
        st.markdown("**Canny**")
        st.image(canny_edges, caption="Canny edges")
        if canny_bbox:
            cx, cy, cw, ch = canny_bbox
            st.success(f"Plate found — W:{cw}px  H:{ch}px  AR:{cw/ch:.2f}")
        else:
            st.error("No plate detected")

    with col2:
        st.markdown("**Sobel**")
        st.image(sobel_edges, caption="Sobel edges")
        if sobel_bbox:
            sx, sy, sw, sh = sobel_bbox
            st.success(f"Plate found — W:{sw}px  H:{sh}px  AR:{sw/sh:.2f}")
        else:
            st.error("No plate detected")
