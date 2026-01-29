import streamlit as st
import cv2
import numpy as np
import zipfile
import io
from PIL import Image

# ─────────────────────────────
# Page config
# ─────────────────────────────
st.set_page_config(
    page_title="Shealth Masking App",
    layout="centered"
)

st.title("🖼️ ShealthAI Image Masking")
st.write("**Masking1** and **Masking2** polygon configurations.")

# ─────────────────────────────
# Polygon References
# ─────────────────────────────
POLYGONS_MASKING_1 = [
    np.array([[1494, 11], [1494, 78], [281, 83], [281, 3]], dtype=np.int32),
    np.array([[1768, 21], [1773, 93], [2984, 88], [2984, 10]], dtype=np.int32)
]

POLYGONS_MASKING_2 = [
    np.array([[391, 27], [2143, 23], [2143, 111], [399, 107]], dtype=np.int32),
    np.array([[2526, 23], [2530, 107], [4273, 115], [4265, 6]], dtype=np.int32)
]

# ─────────────────────────────
# Session State Init
# ─────────────────────────────
st.session_state.setdefault("mask_reference", "Masking1 (Reference 1)")
st.session_state.setdefault("zip_ready", False)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("zip_buffer", None)

# ─────────────────────────────
# Mask Reference Dropdown
# ─────────────────────────────
mask_type = st.selectbox(
    "Select Mask Reference",
    ["Masking1 (Reference 1)", "Masking2 (Reference 2)"],
    index=0 if st.session_state.mask_reference == "Masking1 (Reference 1)" else 1
)
st.session_state.mask_reference = mask_type

st.divider()

# ─────────────────────────────
# Reference Tabs
# ─────────────────────────────
tab1, tab2 = st.tabs(["Reference 1", "Reference 2"])

with tab1:
    st.subheader("Reference 1 – Polygon Masking Example")
    st.image("masking1.png", use_column_width=True)

with tab2:
    st.subheader("Reference 2 – Polygon Masking Example")
    st.image("masking2.png", use_column_width=True)

st.divider()

# ─────────────────────────────
# Upload Images (Dynamic Key)
# ─────────────────────────────
uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

# ─────────────────────────────
# Remove Uploaded Images
# ─────────────────────────────
if uploaded_files:
    if st.button("🗑️ Remove Uploaded Images"):
        st.session_state.uploader_key += 1
        st.session_state.zip_ready = False
        st.session_state.zip_buffer = None

# ─────────────────────────────
# Apply Mask (IN-MEMORY ONLY)
# ─────────────────────────────
if uploaded_files and st.button("🚀 Apply Mask & Prepare ZIP"):

    polygons = (
        POLYGONS_MASKING_1
        if "Masking1" in st.session_state.mask_reference
        else POLYGONS_MASKING_2
    )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                st.warning(f"❌ Could not read {file.name}")
                continue

            # Apply polygon mask
            cv2.fillPoly(img, polygons, color=(0, 0, 255))

            # Encode image directly to memory
            success, encoded_img = cv2.imencode(".png", img)
            if not success:
                st.warning(f"❌ Encoding failed for {file.name}")
                continue

            zipf.writestr(
                f"masked_{file.name}",
                encoded_img.tobytes()
            )

    zip_buffer.seek(0)
    st.session_state.zip_ready = True
    st.session_state.zip_buffer = zip_buffer

# ─────────────────────────────
# Download ZIP + Reset
# ─────────────────────────────
if st.session_state.zip_ready and st.session_state.zip_buffer is not None:
    st.success(f"✅ Masking completed using **{st.session_state.mask_reference}**")

    if st.download_button(
        "⬇️ Download ZIP",
        data=st.session_state.zip_buffer,
        file_name="masked_images.zip",
        mime="application/zip"
    ):
        # Reset uploader cleanly
        st.session_state.uploader_key += 1
        st.session_state.zip_ready = False
        st.session_state.zip_buffer = None