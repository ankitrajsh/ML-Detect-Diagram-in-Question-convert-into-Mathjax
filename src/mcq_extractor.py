import re
import io
import base64
import os
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import cv2
import pytesseract
from config import Config
try:
    from openai import AzureOpenAI
    _HAS_OPENAI = True
except Exception:
    AzureOpenAI = None
    _HAS_OPENAI = False


@dataclass
class MCQOption:
    label: str
    text: str
    mathjax: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h in original image coords


@dataclass
class MCQResult:
    question: Dict[str, Any]
    options: List[Dict[str, Any]]
    diagrams: List[Dict[str, Any]]
    meta: Dict[str, Any]


def _read_image(image_input: Any) -> Image.Image:
    """
    Accepts: file path, PIL Image, numpy array, or bytes-like. Returns PIL Image (RGB).
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img = Image.fromarray(image_input)
        else:
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    elif isinstance(image_input, (bytes, bytearray, io.BytesIO)):
        if isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            img = Image.open(image_input)
        else:
            img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        raise ValueError("Unsupported image input type")

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Basic preprocessing to improve OCR quality
    np_img = np.array(img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    # Adaptive threshold to separate text
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 10)
    # Slight dilation to connect characters
    kernel = np.ones((1, 1), np.uint8)
    proc = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(proc)


def _ocr_with_bboxes(img: Image.Image) -> List[Dict[str, Any]]:
    """Run Tesseract and return list of tokens with text, conf, and bbox."""
    # Use a config that preserves inter-word spaces and line structure
    config = "--oem 3 --psm 6 -l eng preserve_interword_spaces=1"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    results = []
    n = len(data.get('text', []))
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = -1.0
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        line_num = data.get('line_num', [None]*n)[i]
        block_num = data.get('block_num', [None]*n)[i]
        par_num = data.get('par_num', [None]*n)[i]
        results.append({
            'text': text,
            'conf': conf,
            'bbox': (x, y, w, h),
            'line_num': line_num,
            'block_num': block_num,
            'par_num': par_num,
        })
    return results


def _group_tokens_by_line(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lines: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}
    for t in tokens:
        key = (t.get('block_num'), t.get('par_num'), t.get('line_num'))
        lines.setdefault(key, []).append(t)
    grouped = []
    for key, ts in lines.items():
        ts_sorted = sorted(ts, key=lambda x: x['bbox'][0])
        text = " ".join([x['text'] for x in ts_sorted]).strip()
        xs = [x['bbox'][0] for x in ts_sorted]
        ys = [x['bbox'][1] for x in ts_sorted]
        ws = [x['bbox'][2] for x in ts_sorted]
        hs = [x['bbox'][3] for x in ts_sorted]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max([xs[i] + ws[i] for i in range(len(xs))]), max([ys[i] + hs[i] for i in range(len(ys))])
        bbox = (x0, y0, x1 - x0, y1 - y0)
        conf = float(np.mean([x['conf'] for x in ts_sorted if x['conf'] >= 0])) if ts_sorted else -1.0
        grouped.append({'text': text, 'conf': conf, 'bbox': bbox})
    # sort by y then x
    grouped = sorted(grouped, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    return grouped


OPTION_LABEL_START = re.compile(r"\(?([A-Da-d])\)?\s*[\.:\)]\s*")
OPTION_LABEL_RE = re.compile(r"^\s*\(?([A-Da-d])\)?\s*[\.:\)]\s*(.*)$")

def _split_line_into_options(line_text: str) -> List[Tuple[str, str]]:
    """
    If a single line contains multiple options (e.g., "A. foo B. bar C. baz"),
    split into [(label, text), ...]. If no labels, return empty list.
    """
    parts = []
    matches = list(OPTION_LABEL_START.finditer(line_text))
    if not matches:
        return parts
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        start = m.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(line_text)
        txt = line_text[start:end].strip()
        parts.append((label, txt))
    return parts


def _split_question_and_options(lines: List[Dict[str, Any]]):
    question_lines = []
    options: List[MCQOption] = []
    current_opt: Optional[MCQOption] = None
    options_started = False

    for ln in lines:
        text = ln['text']
        # First, try multi-option split within one line
        multi_opts = _split_line_into_options(text)
        if multi_opts:
            options_started = True
            # Flush any existing option
            if current_opt:
                options.append(current_opt)
                current_opt = None
            # Create options for all in this line
            for idx, (label, opt_text) in enumerate(multi_opts):
                bbox = ln['bbox'] if idx == 0 else None  # bbox for first; others unknown at line granularity
                options.append(MCQOption(label=label, text=opt_text, mathjax=None, bbox=bbox))
            continue

        # Else, check if this line begins a single option
        m = OPTION_LABEL_RE.match(text)
        if m:
            options_started = True
            if current_opt:
                options.append(current_opt)
            label = m.group(1).upper()
            opt_text = m.group(2).strip()
            current_opt = MCQOption(label=label, text=opt_text, mathjax=None, bbox=ln['bbox'])
        else:
            # Continuation of current option or question part
            if current_opt is not None:
                # Heuristic: indented or similar y appends to option
                current_opt.text = (current_opt.text + " " + text).strip()
                # expand bbox
                x, y, w, h = current_opt.bbox
                x2, y2, w2, h2 = ln['bbox']
                nx0, ny0 = min(x, x2), min(y, y2)
                nx1 = max(x + w, x2 + w2)
                ny1 = max(y + h, y2 + h2)
                current_opt.bbox = (nx0, ny0, nx1 - nx0, ny1 - ny0)
            else:
                # If options started, and no current option, treat as stray; else question
                if not options_started:
                    question_lines.append(ln)

    if current_opt:
        options.append(current_opt)

    # Question text is all lines before first option
    if question_lines:
        q_text = " ".join([l['text'] for l in question_lines]).strip()
        # Merge bbox of all question lines
        xs = [l['bbox'][0] for l in question_lines]
        ys = [l['bbox'][1] for l in question_lines]
        x1s = [l['bbox'][0] + l['bbox'][2] for l in question_lines]
        y1s = [l['bbox'][1] + l['bbox'][3] for l in question_lines]
        q_bbox = (min(xs), min(ys), max(x1s) - min(xs), max(y1s) - min(ys))
    else:
        q_text, q_bbox = "", None

    return q_text, q_bbox, options


MATH_HEURISTIC_CHARS = set(list("=+−-*/^_{}()[]|<>∑∫∞≈≃≠≤≥πθλμσΔΩαβγηϕφψω″′··×÷√％％%"))

SCI_NOTATION_RE = re.compile(r"(?P<coef>\d+(?:\.\d+)?)\s*[x×\*]\s*10\s*(?:\^|\s*\^?\s*)?\s*(?P<sign>[-−–])?\s*(?P<exp>\d+)")

def _to_latex_scientific(match: re.Match) -> str:
    coef = match.group('coef')
    sign = match.group('sign')
    exp = match.group('exp')
    if sign:
        exp = f"-{exp}"
    return f"{coef} \\times 10^{{{exp}}}"


def _inject_mathjax(text: str) -> str:
    """Wrap segments that look mathy into $...$ as a naive placeholder."""
    if not text:
        return text
    # Normalize unicode multiply to LaTeX form inside scientific notation
    replaced = SCI_NOTATION_RE.sub(lambda m: _to_latex_scientific(m), text)
    # If we changed anything, wrap minimally around the changed fragments by inserting $...$
    if replaced != text:
        return replaced.replace("×", "\\times")
    # fallback: if many math symbols, wrap entire string
    count = sum(ch in MATH_HEURISTIC_CHARS for ch in text)
    if count >= max(2, len(text) // 20):
        return f"$ {text} $"
    return text


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def _find_diagram_regions(img_rgb: np.ndarray, text_line_bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Simple contour-based detector for diagram-like regions:
    - Edge detect via Canny
    - Find external contours and their bounding boxes
    - Filter by size/aspect and low overlap with text lines
    Returns list of bboxes (x, y, w, h).
    """
    h, w, _ = img_rgb.shape
    # Edges
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    # Dilate to connect lines
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[int, int, int, int]] = []
    img_area = w * h
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < img_area * 0.01:  # too small
            continue
        if bw < 40 or bh < 40:
            continue
        aspect = bw / max(1, bh)
        if aspect < 0.2 or aspect > 5.0:
            continue
        # Discard if overlaps strongly with text line boxes
        max_iou = 0.0
        for tb in text_line_bboxes:
            max_iou = max(max_iou, _iou((x, y, bw, bh), tb))
            if max_iou > 0.4:
                break
        if max_iou > 0.4:
            continue
        candidates.append((x, y, bw, bh))

    # Prefer regions in the upper 2/3 of the image (where question+diagram typically are)
    filtered = [b for b in candidates if b[1] + b[3] / 2 < h * 0.75]
    if not filtered:
        filtered = candidates

    # Merge overlapping candidates (non-maximum suppression by IoU)
    filtered = sorted(filtered, key=lambda b: b[2] * b[3], reverse=True)
    chosen: List[Tuple[int, int, int, int]] = []
    for b in filtered:
        if all(_iou(b, c) < 0.3 for c in chosen):
            chosen.append(b)
        if len(chosen) >= 3:
            break
    return chosen


def _detect_diagrams_gpt(pil_img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """Call Azure OpenAI (GPT-4o) to propose diagram bounding boxes.
    Returns list of (x, y, w, h). Empty list on failure or if disabled.
    """
    if not (Config.USE_GPT_DIAGRAM and _HAS_OPENAI and Config.AZURE_OPENAI_ENDPOINT and Config.AZURE_OPENAI_API_KEY):
        return []
    try:
        w, h = pil_img.size
        # Downscale to reduce tokens while preserving coords ratio
        max_side = 1024
        scale = min(1.0, max_side / max(w, h))
        img_for_api = pil_img
        if scale < 1.0:
            img_for_api = pil_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img_for_api.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        prompt = (
            "Return JSON with key 'diagrams' as a list of boxes for diagram regions only (exclude plain text).\n"
            f"Use coordinates for the ORIGINAL image size width={w}, height={h}.\n"
            "Each box fields: {\"x\":int, \"y\":int, \"w\":int, \"h\":int, \"confidence\":float}.\n"
            "Only return JSON, no extra text."
        )

        # Provide scale hint if resized
        if scale < 1.0:
            prompt += f" The attached image is downscaled by factor {scale:.4f}; scale your returned coordinates up to original size."

        resp = client.responses.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            response_format={"type": "json_object"},
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url}
                ]
            }]
        )

        # The SDK exposes output_text for JSON content
        raw = getattr(resp, "output_text", None)
        if not raw:
            return []
        import json
        payload = json.loads(raw)
        boxes = []
        for d in payload.get("diagrams", []) or []:
            try:
                x, y, bw, bh = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]) 
                # clamp
                x = max(0, min(x, w-1)); y = max(0, min(y, h-1))
                bw = max(1, min(bw, w - x)); bh = max(1, min(bh, h - y))
                boxes.append((x, y, bw, bh))
            except Exception:
                continue
        return boxes
    except Exception:
        return []


def extract_mcq(image_input: Any) -> MCQResult:
    """
    Minimal MCQ extractor using pytesseract.
    Returns structured result with placeholders for mathjax.
    """
    pil_img = _read_image(image_input)
    w, h = pil_img.size

    pre = _preprocess_for_ocr(pil_img)
    tokens = _ocr_with_bboxes(pre)
    lines = _group_tokens_by_line(tokens)

    q_text, q_bbox, options = _split_question_and_options(lines)

    # Inject MathJax heuristically
    q_math = _inject_mathjax(q_text) if q_text else None
    for opt in options:
        opt.mathjax = _inject_mathjax(opt.text) if opt.text else None

    # Collect text line boxes to avoid counting them as diagrams
    text_line_bboxes = [l['bbox'] for l in lines]

    # Detect diagram-like regions on the original image (GPT-4o first if enabled, else local)
    np_rgb = np.array(pil_img)
    diag_boxes = []
    gpt_boxes = _detect_diagrams_gpt(pil_img)
    if gpt_boxes:
        # remove boxes that overlap heavily with text lines
        for b in gpt_boxes:
            if max((_iou(b, tb) for tb in text_line_bboxes), default=0.0) <= 0.4:
                diag_boxes.append(b)
    if not diag_boxes:
        diag_boxes = _find_diagram_regions(np_rgb, text_line_bboxes)

    # Save crops
    diagrams: List[Dict[str, Any]] = []
    out_dir = os.path.join("data", "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    for b in diag_boxes:
        x, y, bw, bh = b
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        crop = pil_img.crop((x0, y0, x1, y1))
        fname = f"diagram_{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join(out_dir, fname)
        try:
            crop.save(fpath)
            diagrams.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "image_path": fpath})
        except Exception:
            diagrams.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "image_path": None})

    result = MCQResult(
        question={"text": q_text, "mathjax": q_math, "bbox": q_bbox},
        options=[asdict(o) for o in options],
        diagrams=diagrams,
        meta={
            "image_size": [w, h],
            "engine": "pytesseract",
            "notes": "MVP extractor with simple diagram detection.",
        },
    )
    return result


__all__ = ["extract_mcq", "MCQResult", "MCQOption"]
