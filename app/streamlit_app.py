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

st.title("Image → MCQ Extractor (MVP)")
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
            for i, d in enumerate(result.diagrams):
                st.write(f"Diagram {i+1} bbox: {d.get('bbox')}")
                if d.get("image_path"):
                    st.image(d["image_path"], use_column_width=True)  # path if saved

        with st.expander("Raw JSON"):
            import json
            st.code(json.dumps({
                "question": result.question,
                "options": result.options,
                "diagrams": result.diagrams,
                "meta": result.meta,
            }, indent=2), language="json")

    except Exception as e:
        st.error(f"Failed to process image: {e}")
else:
    st.info("Upload an image to begin.")
