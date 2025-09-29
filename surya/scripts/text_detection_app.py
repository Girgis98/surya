"""Streamlit app for exploring Surya text detection and OCR."""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - pandas is an optional dependency at runtime
    pd = None

import pypdfium2
import streamlit as st
from PIL import Image

from surya.common.surya.schema import TaskNames
from surya.debug.draw import draw_polys_on_image
from surya.debug.text import draw_text_on_image
from surya.models import load_predictors
from surya.recognition.schema import OCRResult
from surya.settings import settings

# Ensure predictors know we're running inside a Streamlit experience.
settings.IN_STREAMLIT = True


@dataclass
class DetectionVisuals:
    overlay: Image.Image
    heatmap: Image.Image | None
    affinity_map: Image.Image | None


@st.cache_resource(show_spinner=True)
def get_predictors():
    """Load all Surya predictors and cache them across reruns."""
    return load_predictors()


@st.cache_data(show_spinner=False)
def load_pdf_page(pdf_bytes: bytes, page_number: int, dpi: int) -> Image.Image:
    """Render the requested page of a PDF as a PIL image."""
    with io.BytesIO(pdf_bytes) as buffer:
        document = pypdfium2.PdfDocument(buffer)
        try:
            renderer = document.render(
                pypdfium2.PdfBitmap.to_pil,
                page_indices=[page_number - 1],
                scale=dpi / 72,
            )
            page_image = list(renderer)[0]
            return page_image.convert("RGB")
        finally:
            document.close()


@st.cache_data(show_spinner=False)
def get_pdf_page_count(pdf_bytes: bytes) -> int:
    with io.BytesIO(pdf_bytes) as buffer:
        document = pypdfium2.PdfDocument(buffer)
        try:
            return len(document)
        finally:
            document.close()


def build_detection_overlay(image: Image.Image, boxes):
    polygons = []
    labels = []
    for idx, box in enumerate(boxes, start=1):
        polygons.append(box.polygon)
        label = f"#{idx}"
        if box.confidence is not None:
            label = f"{label} · {box.confidence:.2f}"
        labels.append(label)
    return draw_polys_on_image(polygons, image.copy(), labels=labels, label_font_size=16)


def summarize_detection(prediction) -> List[dict]:
    rows: List[dict] = []
    for idx, bbox in enumerate(prediction.bboxes, start=1):
        rows.append(
            {
                "region": idx,
                "confidence": None if bbox.confidence is None else round(bbox.confidence, 4),
                "bbox": [int(value) for value in bbox.bbox],
                "polygon": [[int(point) for point in corner] for corner in bbox.polygon],
                "width": round(bbox.width, 2),
                "height": round(bbox.height, 2),
                "area": round(bbox.area, 2),
            }
        )
    return rows


def detection_coverage(prediction, image: Image.Image) -> float:
    image_area = image.width * image.height
    if image_area == 0:
        return 0.0
    total_area = sum(bbox.area for bbox in prediction.bboxes)
    return min(total_area / image_area, 1.0)


def run_detection(image: Image.Image, predictors) -> tuple:
    detection_predictor = predictors["detection"]
    prediction = detection_predictor([image], include_maps=True)[0]
    overlay = build_detection_overlay(image, prediction.bboxes)
    visuals = DetectionVisuals(
        overlay=overlay,
        heatmap=prediction.heatmap,
        affinity_map=prediction.affinity_map,
    )
    return prediction, visuals


def run_ocr_from_detection(
    image: Image.Image,
    prediction,
    predictors,
    recognize_math: bool,
) -> tuple[OCRResult, Image.Image, Image.Image]:
    recognition_predictor = predictors["recognition"]
    polygons = [[corner for corner in bbox.polygon] for bbox in prediction.bboxes]
    ocr_prediction = recognition_predictor(
        [image],
        task_names=[TaskNames.ocr_with_boxes],
        polygons=[polygons],
        math_mode=recognize_math,
        return_words=True,
    )[0]
    line_bboxes = [line.bbox for line in ocr_prediction.text_lines]
    line_polygons = [line.polygon for line in ocr_prediction.text_lines]
    line_texts = [line.text for line in ocr_prediction.text_lines]

    text_overlay = draw_text_on_image(line_bboxes, line_texts, image.size)
    box_overlay = draw_polys_on_image(
        line_polygons,
        image.copy(),
        labels=[
            f"#{idx} · {line.confidence:.2f}" if line.confidence is not None else f"#{idx}"
            for idx, line in enumerate(ocr_prediction.text_lines, start=1)
        ],
        label_font_size=16,
    )
    return ocr_prediction, text_overlay, box_overlay


def detection_details_section(prediction, visuals, image: Image.Image):
    st.subheader("Detection details")
    coverage = detection_coverage(prediction, image) * 100
    metrics_col, bbox_col = st.columns([1, 1])
    metrics_col.metric("Detected regions", len(prediction.bboxes))
    metrics_col.metric("Coverage (approx.)", f"{coverage:.2f}%")
    bbox_col.metric("Image width", image.width)
    bbox_col.metric("Image height", image.height)

    tabs = st.tabs(["Overlay", "Heatmap", "Affinity map"])
    with tabs[0]:
        st.image(visuals.overlay, caption="Detected polygons", use_container_width=True)
    with tabs[1]:
        if visuals.heatmap is not None:
            st.image(visuals.heatmap, caption="Detector heatmap", use_container_width=True)
        else:
            st.info("Heatmap not available for this detection run.")
    with tabs[2]:
        if visuals.affinity_map is not None:
            st.image(visuals.affinity_map, caption="Detector affinity map", use_container_width=True)
        else:
            st.info("Affinity map not available for this detection run.")

    rows = summarize_detection(prediction)
    with st.expander("Detected region data", expanded=True):
        if pd is not None:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:  # pragma: no cover - fallback for environments without pandas
            st.write(rows)

    with st.expander("Raw detection payload"):
        st.json(
            prediction.model_dump(exclude=["heatmap", "affinity_map"]),
            expanded=False,
        )


def ocr_details_section(ocr_result: OCRResult, text_overlay: Image.Image, box_overlay: Image.Image):
    st.subheader("OCR from detected regions")
    visual_tabs = st.tabs(["Text overlay", "Boxes on source", "Recognized text"])
    with visual_tabs[0]:
        st.image(text_overlay, caption="Text rendered by OCR", use_container_width=True)
    with visual_tabs[1]:
        st.image(box_overlay, caption="OCR bounding boxes", use_container_width=True)
    with visual_tabs[2]:
        st.text("\n".join(line.text for line in ocr_result.text_lines))

    line_rows: List[dict] = []
    word_rows: List[dict] = []
    for line_idx, line in enumerate(ocr_result.text_lines, start=1):
        line_rows.append(
            {
                "line": line_idx,
                "text": line.text,
                "confidence": None if line.confidence is None else round(line.confidence, 4),
                "bbox": [int(value) for value in line.bbox],
                "polygon": [[int(point) for point in corner] for corner in line.polygon],
                "chars": len(line.chars),
                "words": len(line.words or []),
            }
        )
        if line.words:
            for word_idx, word in enumerate(line.words, start=1):
                word_rows.append(
                    {
                        "line": line_idx,
                        "word": word_idx,
                        "text": word.text,
                        "confidence": None
                        if word.confidence is None
                        else round(word.confidence, 4),
                        "bbox": [int(value) for value in word.bbox],
                    }
                )

    with st.expander("OCR line details", expanded=True):
        if pd is not None:
            st.dataframe(pd.DataFrame(line_rows), use_container_width=True)
        else:  # pragma: no cover
            st.write(line_rows)

    if word_rows:
        with st.expander("OCR word details"):
            if pd is not None:
                st.dataframe(pd.DataFrame(word_rows), use_container_width=True)
            else:  # pragma: no cover
                st.write(word_rows)

    with st.expander("Raw OCR payload"):
        st.json(ocr_result.model_dump(), expanded=False)


def main():
    st.set_page_config(
        page_title="Surya Text Detection Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Surya Text Detection Explorer")
    st.write(
        "Inspect every artifact produced by Surya's text detector and immediately run OCR on the detected regions."
    )

    predictors = get_predictors()

    st.sidebar.header("Input")
    uploaded = st.sidebar.file_uploader(
        "Upload a PDF page or image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    )
    recognize_math = st.sidebar.toggle(
        "Recognize math",
        value=True,
        help="When enabled, OCR keeps mathematical markup in the output.",
    )

    if uploaded is None:
        st.info("Upload a PDF or image to begin.")
        return

    file_bytes = uploaded.getvalue()
    is_pdf = uploaded.type == "application/pdf"

    if is_pdf:
        page_count = get_pdf_page_count(file_bytes)
        page_number = st.sidebar.number_input(
            "Page number",
            min_value=1,
            max_value=page_count,
            value=1,
        )
        base_image = load_pdf_page(file_bytes, int(page_number), settings.IMAGE_DPI)
    else:
        base_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    st.sidebar.image(base_image, caption="Input preview", use_container_width=True)

    detection_placeholder = st.empty()
    action_col, _ = st.columns([1, 3])
    run_detection_clicked = action_col.button("Run text detection", type="primary")

    if run_detection_clicked:
        with st.spinner("Running text detection..."):
            prediction, visuals = run_detection(base_image, predictors)
        st.session_state["last_detection"] = prediction
        st.session_state["last_detection_visuals"] = visuals
        st.session_state.pop("last_ocr", None)
        st.session_state.pop("last_ocr_text_overlay", None)
        st.session_state.pop("last_ocr_box_overlay", None)

    prediction = st.session_state.get("last_detection")
    visuals = st.session_state.get("last_detection_visuals")

    if prediction is None or visuals is None:
        detection_placeholder.info("Run text detection to see detailed results.")
        return

    with detection_placeholder.container():
        detection_details_section(prediction, visuals, base_image)

    st.divider()
    st.subheader("Next step: OCR")
    ocr_button = st.button("Run OCR on detected regions", type="secondary")

    if ocr_button:
        with st.spinner("Running OCR on detected regions..."):
            ocr_result, text_overlay, box_overlay = run_ocr_from_detection(
                base_image, prediction, predictors, recognize_math
            )
        st.session_state["last_ocr"] = ocr_result
        st.session_state["last_ocr_text_overlay"] = text_overlay
        st.session_state["last_ocr_box_overlay"] = box_overlay

    ocr_result = st.session_state.get("last_ocr")
    text_overlay = st.session_state.get("last_ocr_text_overlay")
    box_overlay = st.session_state.get("last_ocr_box_overlay")

    if ocr_result is not None and text_overlay is not None and box_overlay is not None:
        ocr_details_section(ocr_result, text_overlay, box_overlay)
    else:
        st.info("Run OCR to inspect recognition results for the detected text.")


if __name__ == "__main__":
    main()
