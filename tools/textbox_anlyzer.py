"""
extract_part_b.py
-----------------
Extract and OCR the handwritten text lines from a Part-B answer box,
using the QR code position and the PDF layout formula to locate each line.

Layout (from draw_part_b):
  - box_h      = 45 mm
  - qr_size    = 20 mm  (QR is placed 1pt below y_box_top)
  - First line : y_box_top − 1 mm − 2×FONT_SIZE_LARGE (PDF coords)
  - Line step  : 2×FONT_SIZE_LARGE points
  - Lines overlapping the QR row start after the QR (left_pad);
    lines below the QR span the full box width.

For each line the function returns:
  • code_name  – QR payload (e.g. "B.1")
  • bbox       – (x1, y1, x2, y2) in original image pixels
  • roi        – np.ndarray crop at original scale
  • text       – OCR result (Polish, PSM 7 single-line)

Dependencies:
    pip install opencv-python-headless pytesseract pyzbar
    apt-get install tesseract-ocr tesseract-ocr-pol
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass
from typing import Optional
from pyzbar.pyzbar import decode as pyzbar_decode


# ── PDF layout constants ──────────────────────────────────────────────────────
_BOX_H_MM       = 45.0
_QR_SIZE_MM     = 20.0
_FONT_LARGE_PT  = 12.0          # FONT_SIZE_LARGE
_LINE_STEP_PT   = 2 * _FONT_LARGE_PT   # 24 pt between lines
_FIRST_LINE_MM  = 1.0           # gap from box top before first line
_LEFT_PAD_PT    = 2.0           # pt gap between QR right edge and line start
_LEFT_MARGIN_MM = 2.0           # mm from box left for lines below QR
_RIGHT_MARGIN_MM = 3.0          # mm from box right


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class LineResult:
    """
    One extracted text line from a Part-B answer box.

    Attributes
    ----------
    code_name : str
        QR code payload used as the section identifier (e.g. ``"B.1"``).
    line_index : int
        1-based line number within the box.
    bbox : tuple[int, int, int, int]
        (x1, y1, x2, y2) of the strip in original image pixel coordinates.
    roi : np.ndarray
        Cropped image of the strip at original scale (BGR).
    text : str
        OCR result; empty string when the line is blank.
    """
    code_name: str
    line_index: int
    bbox: tuple[int, int, int, int]
    roi: np.ndarray
    text: str

    def __repr__(self) -> str:
        return (f"LineResult(code={self.code_name!r}, line={self.line_index}, "
                f"bbox={self.bbox}, text={self.text!r})")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_qr(image: np.ndarray) -> Optional[tuple[str, tuple[int, int, int, int]]]:
    """
    Detect the first QR code in *image* and return (payload, xyxy_bbox).
    Uses pyzbar; returns None when nothing is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hits = pyzbar_decode(gray)
    if not hits:
        return None
    obj = hits[0]
    r = obj.rect
    bbox = (r.left, r.top, r.left + r.width, r.top + r.height)
    return obj.data.decode("utf-8", errors="replace"), bbox


def _find_outer_box(image: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the largest rectangle (outer answer box) in the image.
    Returns (x1, y1, x2, y2).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return 0, 0, w, h
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    return x, y, x + bw, y + bh


def _preprocess_for_ocr(roi_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Upscale, sharpen, and binarise a colour ROI for Tesseract.
    Returns a single-channel binary image (dark text = 0, background = 255).
    """
    up = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _ocr_line(roi_bgr: np.ndarray, lang: str = "pol") -> str:
    """
    Run Tesseract on a single line ROI.
    Tries PSM 7 (single line) and PSM 8 (single word) and returns the
    longer result, which tends to be more complete for handwriting.
    """
    binary = _preprocess_for_ocr(roi_bgr)
    cfg7 = f"--oem 3 --psm 7 -l {lang}"
    cfg8 = f"--oem 3 --psm 8 -l {lang}"
    t7 = pytesseract.image_to_string(binary, config=cfg7).strip()
    t8 = pytesseract.image_to_string(binary, config=cfg8).strip()
    return t7 if len(t7) >= len(t8) else t8


# ── Public function ───────────────────────────────────────────────────────────

def extract_part_b_lines(
    image: np.ndarray,
    qr_bbox: Optional[tuple[int, int, int, int]] = None,
    qr_payload: Optional[str] = None,
    *,
    avg_qr_size_px: Optional[int] = None,
    n_lines: int = 5,
    lang: str = "pol",
    strip_pad_above: float = 0.12,
    strip_pad_below: float = 0.80,
) -> list[LineResult]:
    """
    Extract and OCR the ``n_lines`` handwritten text lines from a Part-B
    answer box, using the QR code position as a geometric anchor.

    Parameters
    ----------
    image : np.ndarray
        Full scanned page image (BGR).
    qr_bbox : (x1, y1, x2, y2) | None
        QR code bounding box in image pixels.  When None the function
        auto-detects the first QR code in the image using pyzbar.
    qr_payload : str | None
        QR code payload used as ``code_name`` in results.  Auto-detected
        when ``qr_bbox`` is None; must be supplied when ``qr_bbox`` is given
        manually.
    avg_qr_size_px : int | None
        Override the QR side length used for geometry (pixels).  Useful when
        the detected bbox is noisy.  When None the side is taken from qr_bbox.
    n_lines : int
        Number of text lines to extract (default 5, matching the PDF layout).
    lang : str
        Tesseract language string (default ``"pol"`` for Polish).
    strip_pad_above : float
        Fraction of line_spacing added above the ruled line as top-of-strip
        margin (default 0.12).
    strip_pad_below : float
        Fraction of line_spacing added below the ruled line as bottom-of-strip
        margin (default 0.80).

    Returns
    -------
    list[LineResult]
        One entry per line in order; each carries ``code_name``, ``line_index``,
        ``bbox``, ``roi`` (ndarray), and ``text``.

    Raises
    ------
    RuntimeError
        When ``qr_bbox`` is None and no QR code can be found in the image.
    """
    # ── 1. Resolve QR position ────────────────────────────────────────────────
    if qr_bbox is None:
        hit = _detect_qr(image)
        if hit is None:
            raise RuntimeError("No QR code detected. Supply qr_bbox manually.")
        qr_payload, qr_bbox = hit
    if qr_payload is None:
        qr_payload = "unknown"

    qx1, qy1, qx2, qy2 = qr_bbox
    qr_side_px = avg_qr_size_px if avg_qr_size_px is not None else (qx2 - qx1)

    # ── 2. Derive DPI and unit converters from QR size ────────────────────────
    dpi    = qr_side_px / _QR_SIZE_MM * 25.4
    mm2px  = dpi / 25.4
    pt2px  = dpi / 72.0

    # ── 3. Locate outer answer box ────────────────────────────────────────────
    box_x1, box_y1, box_x2, box_y2 = _find_outer_box(image)
    img_h, img_w = image.shape[:2]

    # ── 4. Compute line y-positions using the PDF formula ─────────────────────
    # PDF: qr_y = y_box_top - qr_size - 1pt  →  y_box_top(img) = qr_y1 - 1pt
    y_box_top_img = qy1 - int(round(1.0 * pt2px))
    box_h_px      = int(round(_BOX_H_MM * mm2px))
    y_box_bot_img = y_box_top_img + box_h_px

    line_spacing_px = _LINE_STEP_PT * pt2px
    first_line_y    = (y_box_top_img
                       + int(round(_FIRST_LINE_MM * mm2px))
                       + int(round(_LINE_STEP_PT * pt2px)))

    line_ys: list[int] = []
    ly = first_line_y
    while len(line_ys) < n_lines and ly < y_box_bot_img:
        line_ys.append(int(round(ly)))
        ly += line_spacing_px

    # ── 5. Horizontal margins ─────────────────────────────────────────────────
    left_pad_x    = qx2 + int(round(_LEFT_PAD_PT * pt2px))
    left_margin_x = box_x1 + int(round(_LEFT_MARGIN_MM * mm2px))
    right_margin_x = box_x2 - int(round(_RIGHT_MARGIN_MM * mm2px))

    # ── 6. Extract and OCR each strip ─────────────────────────────────────────
    results: list[LineResult] = []
    pad_above_px = int(line_spacing_px * strip_pad_above)
    pad_below_px = int(line_spacing_px * strip_pad_below)

    for i, ruled_y in enumerate(line_ys):
        y_top = max(box_y1, ruled_y - pad_above_px)
        y_bot = min(box_y2, ruled_y + pad_below_px)

        # Lines that vertically overlap the QR code start after it
        x_left = left_pad_x if ruled_y <= qy2 + 5 else left_margin_x
        x_left  = max(0, x_left)
        x_right = min(img_w, right_margin_x)

        roi  = image[y_top:y_bot, x_left:x_right]
        text = _ocr_line(roi, lang=lang) if roi.size > 0 else ""

        results.append(LineResult(
            code_name=qr_payload,
            line_index=i + 1,
            bbox=(x_left, y_top, x_right, y_bot),
            roi=roi,
            text=text,
        ))

    return results


# ── Debug visualisation ───────────────────────────────────────────────────────

def draw_line_debug(
    image: np.ndarray,
    lines: list[LineResult],
    qr_bbox: tuple[int, int, int, int],
    *,
    colour_strip: tuple[int, int, int] = (0, 200, 0),
    colour_qr:    tuple[int, int, int] = (0, 120, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Return annotated copy of *image* showing QR and strip bboxes."""
    out = image.copy()
    qx1, qy1, qx2, qy2 = qr_bbox
    cv2.rectangle(out, (qx1, qy1), (qx2, qy2), colour_qr, 2)
    cv2.putText(out, lines[0].code_name if lines else "QR",
                (qx1, qy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_qr, 1)
    for ln in lines:
        x1, y1, x2, y2 = ln.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colour_strip, thickness)
        label = f"L{ln.line_index}: {ln.text[:40]}" if ln.text else f"L{ln.line_index}"
        cv2.putText(out, label, (x1 + 4, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour_strip, 1)
    return out


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pathlib

    if len(sys.argv) < 2:
        print("Usage: python extract_part_b.py <image_path> [avg_qr_size_px]")
        sys.exit(1)

    path         = sys.argv[1]
    avg_qr_size  = int(sys.argv[2]) if len(sys.argv) > 2 else None

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read: {path}")

    lines = extract_part_b_lines(img, avg_qr_size_px=avg_qr_size)

    print(f"Code : {lines[0].code_name if lines else '?'}")
    print()
    for ln in lines:
        print(f"  Line {ln.line_index}  bbox={ln.bbox}")
        print(f"    text : {ln.text!r}")
        roi_path = pathlib.Path(path).with_stem(
            pathlib.Path(path).stem + f"_line{ln.line_index}"
        )
        cv2.imwrite(str(roi_path), ln.roi)

    # Debug overlay
    qr_hit = _detect_qr(img)
    if qr_hit:
        debug = draw_line_debug(img, lines, qr_hit[1])
        dbg_path = pathlib.Path(path).with_stem(pathlib.Path(path).stem + "_debug")
        cv2.imwrite(str(dbg_path), debug)
        print(f"\nDebug image → {dbg_path}")