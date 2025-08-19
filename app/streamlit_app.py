import streamlit as st
from PIL import Image
import io
from typing import Any
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports like 'src.*'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcq_extractor import extract_mcq

st.set_page_config(page_title="MCQ Extractor", layout="wide")

st.title("Image → MCQ Extractor")
st.caption("Upload a question image. The app will OCR and split Question + Options (A/B/C/D). Math is heuristically wrapped for MathJax.")

with st.sidebar:
    st.header("Settings")
    show_bboxes = st.checkbox("Show debug lines (OCR)", value=False, help="Future: visualize detected regions")
    st.info("This MVP uses Tesseract OCR. For best results, use clear, printed images.")

uploaded = st.file_uploader("Upload question image", type=["jpg", "jpeg", "png", "bmp", "tiff"]) 

if uploaded is not None:
    try:
        image_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(pil_img, caption="Input Image", use_column_width=True)

        with st.spinner("Extracting MCQ ..."):
            result = extract_mcq(pil_img)

        st.subheader("Question")
        q = result.question
        if q.get("mathjax"):
            st.latex(q["mathjax"])  # render math if any
        st.write(q.get("text", ""))

        st.subheader("Options")
        for opt in result.options:
            col1, col2 = st.columns([1, 9])
            with col1:
                st.markdown(f"**{opt['label']}**")
            with col2:
                if opt.get("mathjax"):
                    st.latex(opt["mathjax"])  # render math if any
                st.write(opt.get("text", ""))

        if result.diagrams:
            st.subheader("Diagram(s)")
            img_w, img_h = pil_img.size
            adjusted_diagrams = []
            for i, d in enumerate(result.diagrams):
                st.markdown(f"**Diagram {i+1}** — original bbox: {d.get('bbox')}")

                # Original bbox (x, y, w, h)
                try:
                    x, y, bw, bh = d.get("bbox", [0, 0, 0, 0])
                except Exception:
                    x, y, bw, bh = 0, 0, 0, 0

                with st.container(border=True):
                    cols = st.columns(4)
                    with cols[0]:
                        left_adj = st.slider(
                            "Left (+ expand)", min_value=-200, max_value=200, value=0, step=1,
                            key=f"left_adj_{i}"
                        )
                    with cols[1]:
                        top_adj = st.slider(
                            "Top (+ expand)", min_value=-200, max_value=200, value=0, step=1,
                            key=f"top_adj_{i}"
                        )
                    with cols[2]:
                        right_adj = st.slider(
                            "Right (+ expand)", min_value=-200, max_value=200, value=0, step=1,
                            key=f"right_adj_{i}"
                        )
                    with cols[3]:
                        bottom_adj = st.slider(
                            "Bottom (+ expand)", min_value=-200, max_value=200, value=0, step=1,
                            key=f"bottom_adj_{i}"
                        )

                    # Compute adjusted crop within bounds
                    x0 = max(0, x - max(0, left_adj)) if left_adj > 0 else min(img_w - 1, x - left_adj)
                    y0 = max(0, y - max(0, top_adj)) if top_adj > 0 else min(img_h - 1, y - top_adj)
                    x1 = min(img_w, x + bw + max(0, right_adj)) if right_adj > 0 else max(0, x + bw + right_adj)
                    y1 = min(img_h, y + bh + max(0, bottom_adj)) if bottom_adj > 0 else max(0, y + bh + bottom_adj)

                    # Ensure valid box
                    if x1 <= x0: x1 = min(img_w, x0 + 1)
                    if y1 <= y0: y1 = min(img_h, y0 + 1)

                    adj_bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                    st.caption(f"Adjusted bbox: {adj_bbox}")

                    # Show adjusted preview
                    try:
                        crop = pil_img.crop((int(x0), int(y0), int(x1), int(y1)))
                        st.image(crop, caption="Adjusted preview", width=280)
                    except Exception as _e:
                        st.warning("Could not generate adjusted preview.")

                    # Record adjusted bbox for JSON output
                    adjusted_diagrams.append({
                        "bbox": adj_bbox,
                        "image_path": d.get("image_path"),
                        "original_bbox": d.get("bbox"),
                        "adjusted": True,
                        "index": i + 1,
                    })

                    # Dummy save button (no actual file write)
                    if st.button("Save adjusted (dummy)", key=f"save_btn_{i}"):
                        st.success("Pretend-saved adjusted diagram (no file written).")

        with st.expander("Raw JSON"):
            import json
            payload = {
                "question": result.question,
                "options": result.options,
                "diagrams": adjusted_diagrams if result.diagrams else result.diagrams,
                "meta": {**result.meta, "ui_adjusted": bool(result.diagrams)},
            }
            st.code(json.dumps(payload, indent=2), language="json")

    except Exception as e:
        st.error(f"Failed to process image: {e}")
else:
    st.info("Upload an image to begin.")
