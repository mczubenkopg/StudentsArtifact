"""
text_extractor.py
-----------------
Segmentation-first handwritten Polish OCR pipeline:

  ROI
   │
   ├─ 1. Binarise + denoise
   │
   ├─ 2. LINE SEGMENTATION
   │     Horizontal projection profile on the binary image finds ink-free
   │     gaps; each continuous band of ink rows is one text line.
   │
   ├─ 3. GLYPH SEGMENTATION  (per line)
   │     Vertical projection profile on each line image detects ink-free
   │     column gaps → individual character / connected-component blobs.
   │     Wide gaps between blobs are tagged as word boundaries.
   │
   ├─ 4. GLYPH RECOGNITION   (per glyph)
   │     Each glyph patch is recognised independently with:
   │       a) Tesseract PSM-10 (single character mode)
   │       b) TrOCR on the individual glyph image (when available)
   │     The two results are merged per-glyph.
   │
   ├─ 5. WORD / LINE ASSEMBLY
   │     Recognised glyphs are joined with word-boundary spaces into lines,
   │     then lines are joined with newlines / spaces into the full string.
   │
   └─ 6. LLM POSTPROCESSING
         The assembled raw string is corrected toward valid Polish words
         using: llama-cpp (GGUF) → transformers → rule-based (always works).

Public API
----------
    extract_text_from_bbox(image, bbox, *, debug=False, llm_postprocess=True,
                           use_trocr=True, trocr_model=...)  → str

    draw_ocr_debug(image, bbox, ocr_result, *, output_path=None)  → np.ndarray

    TextExtractor(llm_backend="auto", use_trocr=True, ...)

Dependencies
------------
    pip install opencv-python-headless numpy pillow pytesseract
    apt  install tesseract-ocr tesseract-ocr-pol
    # TrOCR (optional, improves glyph recognition):
    #   pip install transformers torch
    # Optional LLM backends:
    #   pip install llama-cpp-python
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import cv2
import numpy as np
from PIL import Image
import pytesseract

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GlyphResult:
    """Recognition result for a single segmented glyph."""
    char:        str                   # best recognised character
    confidence:  float                 # 0–100 (Tesseract) or -1
    tess_char:   str   = ""            # Tesseract's raw answer
    trocr_char:  str   = ""            # TrOCR's raw answer
    bbox:        tuple[int,int,int,int] = field(default_factory=lambda:(0,0,0,0))
    word_break:  bool  = False         # True → space after this glyph


@dataclass
class LineResult:
    """All glyphs on a single text line."""
    glyphs:      list[GlyphResult]
    text:        str                   # assembled glyph string for this line
    line_bbox:   tuple[int,int,int,int] = field(default_factory=lambda:(0,0,0,0))


@dataclass
class OCRResult:
    """Full result returned by TextExtractor.extract()."""
    text:         str                  # final text after LLM postprocessing
    raw_assembly: str                  # text before LLM, after glyph merge
    lines:        list[LineResult]     # per-line details
    llm_used:     bool  = False
    llm_backend:  str   = ""
    bbox:         tuple[int,int,int,int] = field(default_factory=lambda:(0,0,0,0))

    def __str__(self) -> str:
        tag = f" [via {self.llm_backend}]" if self.llm_used else ""
        return f'"{self.text}"{tag}'


# ═════════════════════════════════════════════════════════════════════════════
# Image helpers
# ═════════════════════════════════════════════════════════════════════════════

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """
    Robust binarisation for scanned handwriting on printed forms.

    Steps:
    1. CLAHE equalisation → reduces uneven lighting
    2. Light denoising
    3. Adaptive Gaussian threshold  → ink = 255 (foreground)
    4. Remove horizontal lines from the printed form:
       erode vertically (keeps only tall ink, removes 1-2px horiz. rules)
       then dilate back to restore letter strokes
    5. Small morphological close/open for clean blobs
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    eq    = cv2.fastNlMeansDenoising(eq, h=10)

    binary = cv2.adaptiveThreshold(
        eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # ink → 255 (foreground)
        blockSize=31,
        C=8,
    )

    # ── Suppress horizontal printed form lines ────────────────────────────────
    # A vertical erosion kernel removes ink spans shorter than kernel height;
    # printed dashed lines are 1–2 px tall, letter strokes are 10–40 px tall.
    k_vert_erode  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    k_vert_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    no_hlines = cv2.erode(binary, k_vert_erode)
    no_hlines = cv2.dilate(no_hlines, k_vert_dilate)

    # ── Clean up remaining speckles ───────────────────────────────────────────
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(no_hlines, cv2.MORPH_CLOSE, k_close)
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  k_open)

    return cleaned   # ink=255, paper=0


def _ink_rows(binary: np.ndarray) -> np.ndarray:
    """Horizontal projection: number of ink pixels per row."""
    return binary.sum(axis=1) // 255


def _ink_cols(binary: np.ndarray) -> np.ndarray:
    """Vertical projection: number of ink pixels per column."""
    return binary.sum(axis=0) // 255


# ═════════════════════════════════════════════════════════════════════════════
# LINE SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def segment_lines(
    binary: np.ndarray,
    *,
    min_line_height: int = 8,
    gap_threshold: float = 0.02,
    padding: int = 3,
) -> list[tuple[int, int]]:
    """
    Segment a binarised ROI into horizontal text-line bands.

    Uses the horizontal ink-projection profile: rows whose ink count falls
    below ``gap_threshold × max_ink`` are considered empty; consecutive
    non-empty rows form a band.

    Parameters
    ----------
    binary : np.ndarray
        Binarised image (ink=255, paper=0).
    min_line_height : int
        Bands shorter than this (pixels) are discarded as noise.
    gap_threshold : float
        Fraction of the maximum row-ink that a row must exceed to be
        considered part of a text band.  Increase for noisy scans.
    padding : int
        Extra pixels added above/below each band.

    Returns
    -------
    List of (y_start, y_end) tuples in image coordinates.
    """
    proj   = _ink_rows(binary)
    h      = binary.shape[0]
    thresh = max(1, gap_threshold * float(proj.max()) if proj.max() > 0 else 1)

    in_band = False
    bands: list[tuple[int, int]] = []
    y_start = 0

    for y in range(h):
        if proj[y] > thresh:
            if not in_band:
                y_start = y
                in_band = True
        else:
            if in_band:
                if (y - y_start) >= min_line_height:
                    bands.append((max(0, y_start - padding),
                                  min(h, y + padding)))
                in_band = False

    if in_band and (h - y_start) >= min_line_height:
        bands.append((max(0, y_start - padding), h))

    # Merge bands that are very close together (< 4 px gap)
    merged: list[tuple[int, int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] < 4:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    return merged


# ═════════════════════════════════════════════════════════════════════════════
# GLYPH SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GlyphSegment:
    """A single segmented glyph patch with its position."""
    x1: int
    y1: int   # within the line crop
    x2: int
    y2: int
    binary_patch: np.ndarray    # ink=255 binary, padded
    color_patch:  np.ndarray    # original colour ROI crop, for TrOCR
    word_break_after: bool = False


def _estimate_char_width(line_binary: np.ndarray) -> float:
    """
    Estimate typical character width in pixels from the vertical ink profile.
    Uses the median width of ink runs in the projection.
    """
    proj = _ink_cols(line_binary)
    in_run, run_start = False, 0
    widths: list[int] = []
    for x, v in enumerate(proj):
        if v > 0:
            if not in_run:
                run_start = x
                in_run = True
        else:
            if in_run:
                widths.append(x - run_start)
                in_run = False
    if in_run:
        widths.append(len(proj) - run_start)
    if not widths:
        return 15.0
    # Median of widths that are plausible letter widths (4–80 px)
    plausible = [w for w in widths if 4 <= w <= 80]
    return float(np.median(plausible)) if plausible else float(np.median(widths))


def _vertical_projection_cuts(
    line_binary: np.ndarray,
    estimated_char_w: float,
    valley_threshold: float = 0.10,
) -> list[tuple[int, int]]:
    """
    Find glyph boundaries via vertical ink-projection valleys.

    Removes the printed-form baseline (dashed/solid lines that appear as a
    near-constant low ink level across every column) before thresholding,
    so true ink gaps stand out clearly.

    Returns list of (x_start, x_end) column spans for each glyph group.
    """
    proj = _ink_cols(line_binary).astype(float)
    w    = len(proj)

    # ── Baseline removal ─────────────────────────────────────────────────────
    # Printed form lines (dashes, solid rules) create a near-constant floor.
    # Estimate it as the 15th percentile of non-zero column ink counts.
    nonzero = proj[proj > 0]
    if len(nonzero) > 0:
        baseline = float(np.percentile(nonzero, 15))
        proj = np.maximum(0.0, proj - baseline)
    # ── ─────────────────────────────────────────────────────────────────────

    # Smooth with a narrow box filter to reduce single-pixel noise
    kernel_w = max(1, int(estimated_char_w * 0.15))
    smoothed = np.convolve(proj, np.ones(kernel_w) / kernel_w, mode='same')

    peak = smoothed.max()
    if peak == 0:
        return [(0, w)]

    thresh = valley_threshold * peak
    in_glyph  = False
    spans:    list[tuple[int, int]] = []
    gx_start  = 0

    for x in range(w):
        if smoothed[x] > thresh:
            if not in_glyph:
                gx_start = x
                in_glyph = True
        else:
            if in_glyph:
                spans.append((gx_start, x))
                in_glyph = False

    if in_glyph:
        spans.append((gx_start, w))

    # Merge spans whose gap is ≤ 35 % of estimated char width
    # (keeps multi-stroke letters together while separating words)
    gap_limit = max(2, int(estimated_char_w * 0.35))
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= gap_limit:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)

    return merged if merged else [(0, w)]


def _cc_diacritic_merge(
    spans: list[tuple[int, int]],
    line_binary: np.ndarray,
    line_height: int,
) -> list[tuple[int, int]]:
    """
    Post-process projection spans: absorb tiny spans (likely diacritics,
    dots on i/j, accents) into their nearest neighbour.
    Tiny = width < 30% of median span width AND height < 30% of line height.
    """
    if len(spans) < 2:
        return spans

    widths = [x2 - x1 for x1, x2 in spans]
    med_w  = float(np.median(widths))
    tiny_w = max(3, med_w * 0.30)

    result = list(spans)
    changed = True
    while changed:
        changed = False
        new_result: list[tuple[int, int]] = []
        i = 0
        while i < len(result):
            x1, x2 = result[i]
            sw = x2 - x1
            # Check if this span is tiny (diacritic candidate)
            patch = line_binary[:, x1:x2]
            ink_h = int((patch > 0).any(axis=1).sum())
            if sw <= tiny_w and ink_h < 0.4 * line_height and len(result) > 1:
                # Merge into left neighbour if it exists, else right
                if new_result:
                    new_result[-1] = (new_result[-1][0], x2)
                elif i + 1 < len(result):
                    result[i+1] = (x1, result[i+1][1])
                    i += 1
                    continue
                changed = True
            else:
                new_result.append((x1, x2))
            i += 1
        result = new_result

    return result


def segment_glyphs(
    line_binary: np.ndarray,
    line_color:  np.ndarray,
    *,
    min_glyph_width:  int   = 3,
    min_glyph_height: int   = 4,
    word_gap_factor:  float = 2.2,
    valley_threshold: float = 0.08,
    padding:          int   = 2,
) -> list[GlyphSegment]:
    """
    Segment a single text line into individual character glyph patches.

    Strategy (handles dense/touching handwriting):
    ──────────────────────────────────────────────
    1. Estimate typical character width from the vertical ink projection.
    2. Find glyph column spans via vertical-projection valleys (ink-free
       column gaps in the smoothed projection profile).
    3. Merge narrow spans that look like diacritics into their neighbours.
    4. Detect word boundaries: gaps > ``word_gap_factor × median_gap``.

    This approach works on touching/overlapping letters where connected-
    component boundaries cross letter strokes.

    Parameters
    ----------
    line_binary  : ink=255 binarised line image
    line_color   : original colour line crop (for TrOCR)
    min_glyph_width  : minimum glyph column span width (px)
    min_glyph_height : minimum ink height within span (px)
    word_gap_factor  : multiplier on median gap for word-break detection
    valley_threshold : fraction of peak ink below which a column is a valley
    padding          : pixel padding added around each span

    Returns
    -------
    List of GlyphSegment objects sorted left-to-right.
    """
    h, w = line_binary.shape[:2]
    if w == 0 or h == 0:
        return []

    # ── 1. Estimate character width ───────────────────────────────────────────
    est_cw = _estimate_char_width(line_binary)

    # ── 2. Vertical projection cuts ───────────────────────────────────────────
    spans = _vertical_projection_cuts(line_binary, est_cw, valley_threshold)

    # ── 3. Diacritic merge ────────────────────────────────────────────────────
    spans = _cc_diacritic_merge(spans, line_binary, h)

    # ── 4. Build GlyphSegment objects ─────────────────────────────────────────
    segments: list[GlyphSegment] = []
    for x1_raw, x2_raw in spans:
        x1 = max(0, x1_raw - padding)
        x2 = min(w, x2_raw + padding)

        if (x2 - x1) < min_glyph_width:
            continue

        # Compute vertical extent of actual ink within this column span
        col_patch = line_binary[:, x1:x2]
        ink_rows  = np.where(col_patch.any(axis=1))[0]
        if len(ink_rows) < min_glyph_height:
            continue

        y1 = max(0, int(ink_rows[0]) - padding)
        y2 = min(h, int(ink_rows[-1]) + 1 + padding)

        bin_patch   = line_binary[y1:y2, x1:x2].copy()
        color_patch = (line_color[y1:y2, x1:x2].copy()
                       if line_color is not None else bin_patch)

        segments.append(GlyphSegment(
            x1=x1, y1=y1, x2=x2, y2=y2,
            binary_patch=bin_patch,
            color_patch=color_patch,
        ))

    if not segments:
        return segments

    # ── 5. Word boundary detection ────────────────────────────────────────────
    if len(segments) >= 2:
        gaps = [segments[i+1].x1 - segments[i].x2 for i in range(len(segments)-1)]
        positive_gaps = [g for g in gaps if g > 0]
        median_gap    = float(np.median(positive_gaps)) if positive_gaps else est_cw * 0.5
        threshold     = max(median_gap * word_gap_factor, est_cw * 0.8, 6.0)
        for i, gap in enumerate(gaps):
            if gap >= threshold:
                segments[i].word_break_after = True

    return segments


# ═════════════════════════════════════════════════════════════════════════════
# GLYPH RECOGNITION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_TESS_CHAR_CONFIG = (
    "--oem 3 -l pol "
    "--psm 10 "          # single character
    "-c preserve_interword_spaces=0"
)

_TESS_WORD_CONFIG = (
    "--oem 3 -l pol "
    "--psm 8 "           # single word (fallback for wide blobs)
    "-c preserve_interword_spaces=0"
)

_TESS_LINE_CONFIG = (
    "--oem 3 -l pol "
    "--psm 7 "           # single text line
    "-c preserve_interword_spaces=1"
)


def _prepare_glyph_for_tess(binary_patch: np.ndarray) -> Image.Image:
    """
    Prepare a binarised glyph patch for Tesseract:
    1. Invert so background=white, ink=black (standard for Tesseract)
    2. Pad to square with white border
    3. Upscale to at least 64 px side
    4. Apply mild dilation to thicken thin strokes
    """
    # Our binary has ink=255; Tesseract wants ink=black on white background
    inv = cv2.bitwise_not(binary_patch)   # ink→0 (black), paper→255 (white)

    # Dilate slightly to connect thin strokes
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv = cv2.dilate(inv, k, iterations=1)

    h, w = inv.shape[:2]
    # Pad to square
    side   = max(h, w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    yo = (side - h) // 2
    xo = (side - w) // 2
    canvas[yo:yo+h, xo:xo+w] = inv

    # Add white border so Tesseract doesn't clip glyphs
    bordered = cv2.copyMakeBorder(canvas, 8, 8, 8, 8,
                                  cv2.BORDER_CONSTANT, value=255)

    # Upscale to at least 96 px (PSM-10 works much better on larger images)
    min_side = 96
    bh, bw = bordered.shape[:2]
    if bh < min_side:
        scale = min_side / bh
        bordered = cv2.resize(bordered, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(bordered)


def _tess_char(binary_patch: np.ndarray) -> tuple[str, float]:
    """
    Run Tesseract on a single glyph patch using PSM-10, then PSM-8 fallback.
    Returns (char, confidence).
    """
    pil = _prepare_glyph_for_tess(binary_patch)

    # ── PSM 10: single character ──────────────────────────────────────────────
    try:
        data = pytesseract.image_to_data(pil, config=_TESS_CHAR_CONFIG,
                                          output_type=pytesseract.Output.DICT)
        texts = [t.strip() for t in data["text"] if t.strip()]
        confs = [c for c, t in zip(data["conf"], data["text"])
                 if t.strip() and isinstance(c, (int, float)) and c >= 0]
        if texts:
            char = texts[0][:2]   # allow up to 2 chars (e.g. ligatures)
            conf = float(confs[0]) if confs else -1.0
            if char:
                return char, conf
    except Exception:
        pass

    # ── PSM 8 fallback: single word ───────────────────────────────────────────
    try:
        text = pytesseract.image_to_string(pil, config=_TESS_WORD_CONFIG).strip()
        if text:
            return text[:2], -1.0
    except Exception:
        pass

    return "", -1.0


def _tess_line(line_gray: np.ndarray) -> str:
    """
    Run Tesseract PSM-7 (single text line) on a full line image.
    Returns cleaned text string.
    """
    # Upscale for better accuracy
    h, w = line_gray.shape[:2]
    scale = max(1.0, 60.0 / h)   # ensure at least 60px height
    if scale > 1.0:
        line_gray = cv2.resize(line_gray, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    # Invert: Tesseract expects black ink on white
    inv = cv2.bitwise_not(line_gray)
    bordered = cv2.copyMakeBorder(inv, 6, 6, 6, 6,
                                  cv2.BORDER_CONSTANT, value=255)
    pil = Image.fromarray(bordered)
    try:
        raw = pytesseract.image_to_string(pil, config=_TESS_LINE_CONFIG)
        return _clean_text(raw)
    except Exception:
        return ""


def _bgr_to_pil_rgb(patch: np.ndarray) -> Image.Image:
    """Convert a BGR or grayscale patch to PIL RGB."""
    if patch.ndim == 2:
        rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    elif patch.shape[2] == 4:
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ═════════════════════════════════════════════════════════════════════════════
# TrOCR engine  (lazy-loaded, optional)
# ═════════════════════════════════════════════════════════════════════════════

_TROCR_DEFAULT_MODEL = "microsoft/trocr-large-handwritten"


class TrOCREngine:
    """
    Lazy-loading TrOCR wrapper.  Runs inference on individual glyph patches.
    """

    def __init__(self, model_name: Optional[str] = None,
                 device: Optional[str] = None):
        self.model_name  = model_name or os.environ.get("TROCR_MODEL",
                                                         _TROCR_DEFAULT_MODEL)
        self._device     = device
        self._processor  = None
        self._model      = None
        self._loaded     = False

    def _load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading TrOCR '%s' on %s …", self.model_name, self._device)
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model     = VisionEncoderDecoderModel.from_pretrained(
                self.model_name
            ).to(self._device)
            self._model.eval()
            log.info("TrOCR ready.")
            return True
        except Exception as exc:
            log.warning("TrOCR load failed: %s – disabled.", exc)
            return False

    def is_available(self) -> bool:
        try:
            import transformers, torch   # noqa: F401
            return True
        except ImportError:
            return False

    def read_patch(self, color_patch: np.ndarray) -> str:
        """
        Read a single glyph patch (BGR or grayscale numpy array).
        Returns the decoded text (usually 1 character).
        """
        if not self._load():
            return ""
        try:
            import torch
            # Upscale tiny patches so the ViT encoder has enough pixels
            h, w = color_patch.shape[:2]
            if max(h, w) < 32:
                scale = 32 / max(h, w)
                color_patch = cv2.resize(
                    color_patch, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )
            pil = _bgr_to_pil_rgb(color_patch)
            pixel_values = self._processor(
                images=pil, return_tensors="pt"
            ).pixel_values.to(self._device)
            with torch.no_grad():
                ids = self._model.generate(pixel_values)
            return self._processor.batch_decode(
                ids, skip_special_tokens=True
            )[0].strip()
        except Exception as exc:
            log.debug("TrOCR glyph error: %s", exc)
            return ""


_trocr_singleton: Optional[TrOCREngine] = None


def _get_trocr(model_name: Optional[str] = None) -> TrOCREngine:
    global _trocr_singleton
    if _trocr_singleton is None or (model_name and
                                      model_name != _trocr_singleton.model_name):
        _trocr_singleton = TrOCREngine(model_name=model_name)
    return _trocr_singleton


# ═════════════════════════════════════════════════════════════════════════════
# GLYPH CHARACTER MERGE
# ═════════════════════════════════════════════════════════════════════════════

def _merge_glyph_chars(
    tess_char: str,
    trocr_char: str,
    tess_conf:  float,
) -> str:
    """
    Pick the best single character from Tesseract and TrOCR outputs.

    Rules:
    - If one is empty, use the other.
    - If TrOCR returns a multi-character string for a glyph, take only [0].
    - If Tesseract confidence is high (≥ 70), prefer Tesseract.
    - Otherwise prefer TrOCR when it returns exactly one character.
    - If both return different single chars and confidence is low, prefer
      TrOCR (generally better on handwriting).
    """
    tc  = tess_char.strip()[:1]   if tess_char  else ""
    trc = trocr_char.strip()[:1]  if trocr_char else ""

    if not tc and not trc:
        return ""
    if not trc:
        return tc
    if not tc:
        return trc

    # Both have a char
    if tc == trc:
        return tc
    if tess_conf >= 70:
        return tc
    return trc   # TrOCR wins on uncertain / low-confidence glyphs


# ═════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═════════════════════════════════════════════════════════════════════════════

_CHAR_SUBS = str.maketrans({"|": "I", "!": "I"})


def _clean_text(raw: str) -> str:
    text = raw.translate(_CHAR_SUBS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═════════════════════════════════════════════════════════════════════════════
# LLM BACKENDS  (unchanged from previous version)
# ═════════════════════════════════════════════════════════════════════════════

class _LLMBackend:
    name: str = "base"
    def correct(self, text: str) -> str:  raise NotImplementedError
    def available(self) -> bool:          return False


class _LlamaCppBackend(_LLMBackend):
    name = "llama_cpp"

    def __init__(self, model_path: str, n_ctx: int = 512, n_gpu_layers: int = 0):
        self._model_path   = model_path
        self._n_ctx        = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm          = None

    def _load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self._model_path,
                              n_ctx=self._n_ctx,
                              n_gpu_layers=self._n_gpu_layers,
                              verbose=False)
            log.info("LlamaCpp loaded: %s", self._model_path)
        except Exception as exc:
            log.warning("Cannot load llama-cpp: %s", exc)
            self._llm = None

    def available(self) -> bool:
        return Path(self._model_path).is_file()

    def correct(self, text: str) -> str:
        self._load()
        if self._llm is None:
            return text
        prompt = _build_correction_prompt(text)
        try:
            out = self._llm(prompt, max_tokens=256, stop=["\n\n"])
            return _parse_llm_output(out["choices"][0]["text"], text)
        except Exception as exc:
            log.debug("LlamaCpp error: %s", exc)
            return text


class _TransformersBackend(_LLMBackend):
    name = "transformers"

    def __init__(self, model_id: str, max_new_tokens: int = 256,
                 device: str = "auto"):
        self._model_id = model_id
        self._max_new  = max_new_tokens
        self._device   = device
        self._pipe     = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            tok   = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self._device,
            )
            self._pipe = pipeline("text-generation", model=model,
                                   tokenizer=tok, device_map=self._device)
            log.info("Transformers LLM loaded: %s", self._model_id)
        except Exception as exc:
            log.warning("Transformers LLM load failed: %s", exc)
            self._pipe = None

    def available(self) -> bool:
        try:
            import transformers; return True
        except ImportError:
            return False

    def correct(self, text: str) -> str:
        self._load()
        if self._pipe is None:
            return text
        try:
            res = self._pipe(_build_correction_prompt(text),
                             max_new_tokens=self._max_new,
                             do_sample=False, return_full_text=False)
            return _parse_llm_output(res[0]["generated_text"], text)
        except Exception as exc:
            log.debug("Transformers LLM error: %s", exc)
            return text


class _RuleBasedBackend(_LLMBackend):
    """Offline diacritic + word-form correction for Polish."""
    name = "rule_based"

    _FIXES: list[tuple[str, str]] = [
        ("IOŚĆ", "IOŚĆ"), ("MOZE",   "MOŻE"),  ("JEZYK",  "JĘZYK"),
        ("JEZYKU","JĘZYKU"),("JEZYKA","JĘZYKA"),("WIECEJ","WIĘCEJ"),
        ("TRESCI","TREŚCI"),("TAKZE","TAKŻE"),  ("CZESC",  "CZĘŚĆ"),
        ("WLASNYCH","WŁASNYCH"), ("BLEDY","BŁĘDY"),("BLEDOW","BŁĘDÓW"),
        ("BLAD","BŁĄD"),  ("ZADNYCH","ŻADNYCH"),("ZADEN","ŻADEN"),
        ("ROWNIEZ","RÓWNIEŻ"), ("MOZNA","MOŻNA"),
    ]

    _WORD_FIXES: dict[str, str] = {
        "ZE": "ŻE", "MOZE": "MOŻE", "MOZNA": "MOŻNA",
        "TAKZE": "TAKŻE", "BADZ": "BĄDŹ", "ROWNIEZ": "RÓWNIEŻ",
    }

    def available(self) -> bool:
        return True

    def correct(self, text: str) -> str:
        t = text.strip()
        upper = t.upper()
        for wrong, right in self._FIXES:
            upper = upper.replace(wrong, right)
        # Token-level word fixes (upper)
        words = upper.split()
        words = [self._WORD_FIXES.get(w, w) for w in words]
        upper = " ".join(words)
        # If fixes changed something, return the corrected upper
        if upper != t.upper():
            return upper
        return t


def _build_correction_prompt(text: str) -> str:
    return (
        "You are a Polish OCR corrector. The input below is raw OCR output "
        "from a handwritten Polish form — it may contain wrong letters, "
        "missing diacritics (ą ę ó ś ź ż ń ć ł), merged/split words, and "
        "noise characters. Correct it to form valid Polish sentences. "
        "Output ONLY the corrected Polish text, nothing else.\n\n"
        f"INPUT: {text}\nOUTPUT:"
    )


def _parse_llm_output(raw: str, fallback: str) -> str:
    for line in raw.strip().splitlines():
        c = _clean_text(line)
        if len(c) >= max(3, len(fallback) // 3):
            return c
    return fallback


# ═════════════════════════════════════════════════════════════════════════════
# RESOURCE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _available_gpu_vram_gb() -> float:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5
        )
        return max(int(x) for x in out.decode().strip().splitlines()) / 1024
    except Exception:
        return 0.0


def _available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemAvailable" in line:
                        return int(line.split()[1]) / (1024**2)
        except Exception:
            pass
    return 2.0


def _select_llm_backend(
    backend: Literal["auto","llama_cpp","transformers","rule_based","none"],
    llm_model_path: Optional[str],
    max_ram_gb: float,
) -> _LLMBackend:
    vram = _available_gpu_vram_gb()
    ram  = _available_ram_gb()
    log.info("RAM: %.1f GB  |  GPU VRAM: %.1f GB", ram, vram)

    if backend == "none":
        return _RuleBasedBackend()

    if backend in ("auto", "llama_cpp"):
        mp = llm_model_path or os.environ.get("LLAMA_MODEL_PATH", "")
        if mp and Path(mp).is_file():
            sz = Path(mp).stat().st_size / (1024**3)
            ngl = -1 if vram >= sz + 0.5 else 0
            if ram >= sz + 1.0 or vram >= sz + 0.5:
                return _LlamaCppBackend(mp, n_gpu_layers=ngl)
            log.warning("Insufficient memory for llama.cpp model – rule_based")
            return _RuleBasedBackend()

    if backend in ("auto", "transformers"):
        mid = llm_model_path or os.environ.get("HF_LLM_MODEL", "Qwen/Qwen3-0.6B")
        try:
            import torch  # noqa: F401
            if ram >= 2.5:
                return _TransformersBackend(mid)
            log.warning("Not enough RAM for transformers LLM – rule_based")
        except ImportError:
            log.info("PyTorch absent – using rule_based LLM backend")

    return _RuleBasedBackend()


# ═════════════════════════════════════════════════════════════════════════════
# CORE TextExtractor CLASS
# ═════════════════════════════════════════════════════════════════════════════

def _clamp_bbox(bbox: tuple[int,int,int,int], shape: tuple) -> tuple[int,int,int,int]:
    h, w = shape[:2]
    x0, y0, x1, y1 = bbox
    return max(0,x0), max(0,y0), min(w,x1), min(h,y1)


class TextExtractor:
    """
    Segmentation-first handwritten OCR for Polish text.

    Pipeline
    --------
    binarise → segment_lines → per-line: segment_glyphs
    → per-glyph: Tesseract PSM-10 + optional TrOCR → merge char
    → assemble string → LLM postprocess

    Parameters
    ----------
    llm_backend : str
        ``"auto"`` | ``"llama_cpp"`` | ``"transformers"`` |
        ``"rule_based"`` | ``"none"``
    llm_model_path : str | None
        Local GGUF path or HuggingFace model dir for the LLM corrector.
    max_ram_gb : float
        RAM ceiling for LLM loading (default 24).
    use_trocr : bool
        Enable TrOCR per-glyph recognition (default True).
        Requires ``transformers`` + ``torch``.
    trocr_model : str | None
        TrOCR model name / path (default ``microsoft/trocr-large-handwritten``).
    line_gap_threshold : float
        Fraction of max-row-ink below which a row is considered blank.
        Increase (e.g. 0.05) for very noisy scans.
    word_gap_factor : float
        Multiplier on median glyph gap to detect word boundaries (default 1.8).
    """

    def __init__(
        self,
        llm_backend:        Literal["auto","llama_cpp","transformers",
                                    "rule_based","none"] = "auto",
        llm_model_path:     Optional[str] = None,
        max_ram_gb:         float = 24.0,
        use_trocr:          bool  = True,
        trocr_model:        Optional[str] = None,
        line_gap_threshold: float = 0.02,
        word_gap_factor:    float = 1.8,
    ):
        self._backend           = _select_llm_backend(llm_backend, llm_model_path, max_ram_gb)
        self._line_gap_threshold = line_gap_threshold
        self._word_gap_factor    = word_gap_factor
        log.info("LLM backend: %s", self._backend.name)

        if use_trocr:
            self._trocr: Optional[TrOCREngine] = _get_trocr(trocr_model)
            if not self._trocr.is_available():
                log.warning("TrOCR disabled (transformers/torch not found). "
                            "Install: pip install transformers torch")
                self._trocr = None
            else:
                log.info("TrOCR engine ready (%s)", self._trocr.model_name)
        else:
            self._trocr = None

    # ── Main entry point ──────────────────────────────────────────────────────
    def line_extract(
        self,
        image: np.ndarray,
        *,
        debug:           bool = False,
        llm_postprocess: bool = True,
    ) -> list:
        """
        Run the full segmentation + recognition + postprocessing pipeline.

        Parameters
        ----------
        image : np.ndarray  full BGR scan image
        bbox  : (x0,y0,x1,y1) region of interest
        debug : bool  verbose logging
        llm_postprocess : bool  apply LLM correction step

        Returns
        -------
        list
        """
        line_results = []
        line_color  = image
        line_gray   = _to_gray(line_color)
        line_bin = _binarise(line_gray)


        # ── 3a. Glyph segmentation (used for word-break positions) ────────
        glyphs = segment_glyphs(
            line_bin, line_color,
            word_gap_factor=self._word_gap_factor,
        )

        # ── 3b. Primary text: PSM-7 on the entire line band ───────────────
        # PSM-7 (single text line) is far more reliable than PSM-10 per
        # glyph for dense cursive/print-mix handwriting.
        line_text_psm7 = _tess_line(line_bin)
        if debug:
            log.info("  Line PSM-7: %r", line_text_psm7)

        # ── 3c. Per-glyph PSM-10 recognition ─────────────────────────────
        # We still recognise each glyph individually so the debug view
        # can show per-character confidence, and TrOCR can be applied.
        glyph_results: list[GlyphResult] = []

        for gi, seg in enumerate(glyphs):
            tess_c, tess_conf = _tess_char(seg.binary_patch)

            trocr_c = ""
            if self._trocr is not None:
                trocr_c = self._trocr.read_patch(seg.color_patch)
                trocr_c = trocr_c[:2]

            best_c = _merge_glyph_chars(tess_c, trocr_c, tess_conf)

            if debug:
                log.info("    g%d  tess=%r(%.0f)  trocr=%r  → %r %s",
                         gi, tess_c, tess_conf, trocr_c, best_c,
                         "[W]" if seg.word_break_after else "")

            glyph_results.append(GlyphResult(
                char=best_c,
                confidence=tess_conf,
                tess_char=tess_c,
                trocr_char=trocr_c,
                bbox=(seg.x1, seg.y1,
                      seg.x2, seg.y2),
                word_break=seg.word_break_after,
            ))

        # ── 3d. Build glyph-assembly string ──────────────────────────────
        glyph_assembly = _assemble_line(glyph_results)

        # ── 3e. Pick best line text ────────────────────────────────────────
        # Inject word-break spaces from glyph segmentation into the
        # PSM-7 result when possible; otherwise fall back to whichever
        # candidate is longer/more vowel-rich.
        line_text = _pick_best_line_text(
            psm7_text=line_text_psm7,
            glyph_text=glyph_assembly,
            glyph_results=glyph_results,
        )

        if debug:
            log.info("  Line final: %r",  line_text)

        line_results.append(LineResult(
            glyphs=glyph_results,
            text=line_text,
        ))
        return line_results


    def extract(
        self,
        image: np.ndarray,
        bbox:  tuple[int, int, int, int],
        *,
        debug:           bool = False,
        llm_postprocess: bool = True,
    ) -> OCRResult:
        """
        Run the full segmentation + recognition + postprocessing pipeline.

        Parameters
        ----------
        image : np.ndarray  full BGR scan image
        bbox  : (x0,y0,x1,y1) region of interest
        debug : bool  verbose logging
        llm_postprocess : bool  apply LLM correction step

        Returns
        -------
        OCRResult
        """
        x0, y0, x1, y1 = _clamp_bbox(bbox, image.shape)
        roi_color  = image[y0:y1, x0:x1]

        if roi_color.size == 0:
            raise ValueError(f"Empty ROI for bbox {bbox}")

        gray   = _to_gray(roi_color)
        binary = _binarise(gray)

        # ── 1. Line segmentation ─────────────────────────────────────────────
        line_bands = segment_lines(binary,
                                   gap_threshold=self._line_gap_threshold)
        if debug:
            log.info("  Found %d text line(s)", len(line_bands))

        if not line_bands:
            return OCRResult(text="", raw_assembly="", lines=[],
                             bbox=(x0, y0, x1, y1))

        # ── 2–4. Per-line glyph segmentation + recognition ───────────────────
        line_results: list[LineResult] = []

        for li, (ly1, ly2) in enumerate(line_bands):
            line_bin   = binary[ly1:ly2, :]
            line_color = roi_color[ly1:ly2, :]
            line_gray  = _to_gray(line_color)

            # ── 3a. Glyph segmentation (used for word-break positions) ────────
            glyphs = segment_glyphs(
                line_bin, line_color,
                word_gap_factor=self._word_gap_factor,
            )

            if debug:
                log.info("  Line %d (y=%d–%d): %d glyph(s)",
                         li+1, ly1, ly2, len(glyphs))

            # ── 3b. Primary text: PSM-7 on the entire line band ───────────────
            # PSM-7 (single text line) is far more reliable than PSM-10 per
            # glyph for dense cursive/print-mix handwriting.
            line_text_psm7 = _tess_line(line_bin)
            if debug:
                log.info("  Line %d PSM-7: %r", li+1, line_text_psm7)

            # ── 3c. Per-glyph PSM-10 recognition ─────────────────────────────
            # We still recognise each glyph individually so the debug view
            # can show per-character confidence, and TrOCR can be applied.
            glyph_results: list[GlyphResult] = []

            for gi, seg in enumerate(glyphs):
                tess_c, tess_conf = _tess_char(seg.binary_patch)

                trocr_c = ""
                if self._trocr is not None:
                    trocr_c = self._trocr.read_patch(seg.color_patch)
                    trocr_c = trocr_c[:2]

                best_c = _merge_glyph_chars(tess_c, trocr_c, tess_conf)

                if debug:
                    log.info("    g%d  tess=%r(%.0f)  trocr=%r  → %r %s",
                             gi, tess_c, tess_conf, trocr_c, best_c,
                             "[W]" if seg.word_break_after else "")

                glyph_results.append(GlyphResult(
                    char=best_c,
                    confidence=tess_conf,
                    tess_char=tess_c,
                    trocr_char=trocr_c,
                    bbox=(x0 + seg.x1, y0 + ly1 + seg.y1,
                          x0 + seg.x2, y0 + ly1 + seg.y2),
                    word_break=seg.word_break_after,
                ))

            # ── 3d. Build glyph-assembly string ──────────────────────────────
            glyph_assembly = _assemble_line(glyph_results)

            # ── 3e. Pick best line text ────────────────────────────────────────
            # Inject word-break spaces from glyph segmentation into the
            # PSM-7 result when possible; otherwise fall back to whichever
            # candidate is longer/more vowel-rich.
            line_text = _pick_best_line_text(
                psm7_text=line_text_psm7,
                glyph_text=glyph_assembly,
                glyph_results=glyph_results,
            )

            if debug:
                log.info("  Line %d final: %r", li+1, line_text)

            line_results.append(LineResult(
                glyphs=glyph_results,
                text=line_text,
                line_bbox=(x0, y0 + ly1, x1, y0 + ly2),
            ))


        # ── 5b. Assemble full text from lines ────────────────────────────────
        raw_assembly = " ".join(lr.text for lr in line_results if lr.text)
        raw_assembly = _clean_text(raw_assembly)

        # ── 6. LLM postprocessing ─────────────────────────────────────────────
        final_text = raw_assembly
        llm_used   = False
        llm_name   = ""

        if llm_postprocess and self._backend.available() and raw_assembly:
            try:
                corrected = self._backend.correct(raw_assembly)
                if corrected and len(corrected) >= max(2, len(raw_assembly) // 3):
                    final_text = corrected
                    llm_used   = True
                    llm_name   = self._backend.name
                    log.info("LLM correction: %r → %r", raw_assembly, corrected)
            except Exception as exc:
                log.debug("LLM error: %s", exc)

        return OCRResult(
            text=final_text,
            raw_assembly=raw_assembly,
            lines=line_results,
            llm_used=llm_used,
            llm_backend=llm_name,
            bbox=(x0, y0, x1, y1),
        )


def _assemble_line(glyphs: list[GlyphResult]) -> str:
    """Join glyph characters, inserting spaces at word boundaries."""
    parts: list[str] = []
    for g in glyphs:
        if g.char:
            parts.append(g.char)
            if g.word_break:
                parts.append(" ")
    return "".join(parts).strip()


def _vowel_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are vowels."""
    vowels = set("aąeęioóuyAĄEĘIOÓUY")
    alpha  = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c in vowels) / len(alpha)


def _pick_best_line_text(
    psm7_text:     str,
    glyph_text:    str,
    glyph_results: list[GlyphResult],
) -> str:
    """
    Choose the best line text from PSM-7 (whole-line) and glyph assembly.

    Strategy:
    1. If PSM-7 has a good vowel ratio (≥ 0.18) AND its length is within
       50 % of glyph_text length → use PSM-7 (it reads words more reliably).
    2. If glyph_text is substantially longer (≥ 1.4× PSM-7) → use glyphs.
    3. When PSM-7 wins, inject word-break spaces from glyph segmentation
       by approximating which PSM-7 word tokens correspond to glyph groups.
    """
    p7  = psm7_text.strip()
    gt  = glyph_text.strip()

    if not p7 and not gt:
        return ""
    if not gt:
        return p7
    if not p7:
        return gt

    p7_vr = _vowel_ratio(p7)
    gt_vr = _vowel_ratio(gt)

    # Prefer PSM-7 when it has a reasonable vowel ratio and similar length
    if p7_vr >= 0.18 and len(p7) >= len(gt) * 0.6:
        return p7

    # Prefer glyph assembly when it's substantially richer
    if len(gt) >= len(p7) * 1.4 and gt_vr >= 0.15:
        return gt

    # Default: longer candidate wins; PSM-7 breaks ties
    if len(p7) >= len(gt):
        return p7
    return gt


# ═════════════════════════════════════════════════════════════════════════════
# Public convenience function
# ═════════════════════════════════════════════════════════════════════════════

_default_extractor: Optional[TextExtractor] = None


def extract_text_from_bbox(
    image: np.ndarray,
    bbox:  tuple[int, int, int, int],
    *,
    debug:           bool          = False,
    llm_postprocess: bool          = True,
    llm_backend:     str           = "auto",
    llm_model_path:  Optional[str] = None,
    max_ram_gb:      float         = 24.0,
    use_trocr:       bool          = True,
    trocr_model:     Optional[str] = None,
) -> str:
    """
    Extract handwritten Polish text from *image* within *bbox* = (x0,y0,x1,y1).

    Convenience wrapper around ``TextExtractor.extract()``.  Instantiates
    a module-level singleton on first call.

    Parameters
    ----------
    use_trocr : bool
        Enable per-glyph TrOCR (requires transformers + torch).
    trocr_model : str | None
        Override TrOCR model (default microsoft/trocr-large-handwritten).
    llm_postprocess : bool
        Run LLM correction after assembly (default True).
    llm_backend : str
        ``"auto"``|``"llama_cpp"``|``"transformers"``|``"rule_based"``|``"none"``
    """
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = TextExtractor(
            llm_backend=llm_backend,        # type: ignore[arg-type]
            llm_model_path=llm_model_path,
            max_ram_gb=max_ram_gb,
            use_trocr=use_trocr,
            trocr_model=trocr_model,
        )
    return _default_extractor.extract(
        image, bbox, debug=debug, llm_postprocess=llm_postprocess
    ).text


# ═════════════════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def draw_ocr_debug(
    image: np.ndarray,
    bbox:  tuple[int, int, int, int],
    ocr_result: OCRResult,
    *,
    output_path:  Optional[str] = None,
    qr_bboxes:    Optional[list[tuple[int,int,int,int]]] = None,
    extra_bboxes: Optional[list[tuple[tuple[int,int,int,int],str]]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Annotated debug image showing:
      • Line bands          (cyan)
      • Individual glyphs   (green = TrOCR+Tess agree, yellow = Tess only,
                             magenta = TrOCR only, red = mismatch/low-conf)
      • Recognised character above each glyph
      • Word-break markers
      • QR boxes            (blue)
      • Final assembled text in a side panel

    Returns the annotated BGR image (optionally scaled).
    """
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # ── QR boxes ─────────────────────────────────────────────────────────────
    for qb in (qr_bboxes or []):
        qx1, qy1, qx2, qy2 = qb
        cv2.rectangle(out, (qx1, qy1), (qx2, qy2), (255, 100, 0), 2)
        _put_label(out, "QR", (qx1, max(qy1-6, 12)), (255, 100, 0))

    # ── Extra boxes ───────────────────────────────────────────────────────────
    for (ex1,ey1,ex2,ey2), lbl in (extra_bboxes or []):
        cv2.rectangle(out, (ex1,ey1),(ex2,ey2),(0,220,220),2)
        _put_label(out, lbl, (ex1,ey1-6),(0,220,220))

    # ── ROI outline ───────────────────────────────────────────────────────────
    x0,y0,x1,y1 = _clamp_bbox(bbox, image.shape)
    cv2.rectangle(out,(x0,y0),(x1,y1),(0,220,0),2)

    # ── Line bands ────────────────────────────────────────────────────────────
    for lr in ocr_result.lines:
        lx1,ly1,lx2,ly2 = lr.line_bbox
        cv2.rectangle(out,(lx1,ly1),(lx2,ly2),(0,220,220),1)

        # ── Glyphs ───────────────────────────────────────────────────────────
        for g in lr.glyphs:
            gx1,gy1,gx2,gy2 = g.bbox
            # Colour by agreement
            if g.trocr_char and g.tess_char and g.trocr_char[:1] == g.tess_char[:1]:
                col = (0, 200, 80)    # green  – both agree
            elif g.trocr_char and not g.tess_char:
                col = (200, 0, 200)   # magenta – TrOCR only
            elif g.confidence >= 70:
                col = (0, 200, 220)   # cyan   – high-conf Tesseract
            elif g.confidence >= 40:
                col = (0, 180, 255)   # yellow – medium Tesseract
            else:
                col = (0, 80, 255)    # red    – low confidence

            cv2.rectangle(out,(gx1,gy1),(gx2,gy2),col,1)
            if g.char:
                _put_label(out, g.char, (gx1, max(gy1-3,10)), col,
                           font_scale=0.45)
            if g.word_break:
                cv2.line(out,(gx2,gy1),(gx2,gy2),(255,255,0),1)

    # ── Side panel ────────────────────────────────────────────────────────────
    panel_w = 520
    panel   = np.zeros((out.shape[0], panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    py = 25
    _put_label(panel, "SEGMENTATION RESULTS", (10,py),(200,200,200),0.55)
    py += 22
    cv2.line(panel,(5,py),(panel_w-5,py),(80,80,80),1)
    py += 12

    for li, lr in enumerate(ocr_result.lines):
        _put_label(panel, f"Line {li+1}:", (10,py),(120,180,255),0.45)
        py += 16
        for chunk in _wrap(lr.text or "(empty)", 55):
            _put_label(panel, chunk, (10,py),(200,200,200),0.48)
            py += 15
        # Show glyph summary
        g_summary = " ".join(
            (g.char or "?") + ("_" if g.word_break else "") for g in lr.glyphs
        )
        for chunk in _wrap(f"  [{g_summary}]", 60):
            _put_label(panel, chunk, (10,py),(100,140,100),0.38)
            py += 13
        py += 4

    py += 8
    cv2.line(panel,(5,py),(panel_w-5,py),(80,80,80),1)
    py += 14
    _put_label(panel,"RAW ASSEMBLY:",(10,py),(180,180,100),0.48)
    py += 16
    for chunk in _wrap(ocr_result.raw_assembly or "(empty)", 52):
        _put_label(panel, chunk,(10,py),(220,220,120),0.50)
        py += 16

    py += 8
    cv2.line(panel,(5,py),(panel_w-5,py),(80,80,80),1)
    py += 14
    _put_label(panel,"FINAL TEXT:",(10,py),(100,220,100),0.50)
    py += 16
    for chunk in _wrap(ocr_result.text or "(empty)", 50):
        _put_label(panel,chunk,(10,py),(0,240,120),0.55,thickness=2)
        py += 18

    if ocr_result.llm_used:
        py += 8
        _put_label(panel,f"LLM: {ocr_result.llm_backend}",(10,py),(200,140,0),0.45)

    # Legend
    legy = panel.shape[0] - 90
    _put_label(panel,"Legend:",(10,legy),(160,160,160),0.40); legy+=14
    for col,desc in [
        ((0,200,80),"Tess+TrOCR agree"),
        ((0,220,220),"Tess high-conf"),
        ((0,180,255),"Tess med-conf"),
        ((0,80,255),"Low conf / noise"),
        ((200,0,200),"TrOCR only"),
    ]:
        cv2.rectangle(panel,(10,legy-8),(20,legy),col,-1)
        _put_label(panel,desc,(24,legy),(160,160,160),0.38); legy+=13

    combined = np.hstack([out, panel])
    if scale != 1.0:
        nw,nh = int(combined.shape[1]*scale), int(combined.shape[0]*scale)
        combined = cv2.resize(combined,(nw,nh),interpolation=cv2.INTER_AREA)

    if output_path:
        cv2.imwrite(output_path, combined)
        log.info("Debug image → %s", output_path)

    return combined


def draw_full_page_debug(
    image: np.ndarray,
    results: list[tuple[tuple[int,int,int,int], OCRResult]],
    *,
    qr_bboxes:   Optional[list[tuple[int,int,int,int]]] = None,
    output_path: Optional[str] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Annotate a full scan page with all OCR results and QR codes."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    for qb in (qr_bboxes or []):
        qx1,qy1,qx2,qy2 = qb
        cv2.rectangle(out,(qx1,qy1),(qx2,qy2),(255,100,0),3)
        _put_label(out,"QR",(qx1,max(qy1-8,15)),(255,100,0),bg=True)

    colours = [(0,200,80),(0,180,255),(200,100,255),(255,200,0),(0,220,200)]
    for i,(bbox,result) in enumerate(results):
        col = colours[i % len(colours)]
        rx0,ry0,rx1,ry1 = _clamp_bbox(bbox, image.shape)
        cv2.rectangle(out,(rx0,ry0),(rx1,ry1),col,2)
        # Draw line bands
        for lr in result.lines:
            lx1,ly1,lx2,ly2 = lr.line_bbox
            cv2.rectangle(out,(lx1,ly1),(lx2,ly2),(200,200,200),1)
        short = result.text[:70] + ("…" if len(result.text)>70 else "")
        _put_label(out,f"B{i+1}: {short}",(rx0,max(ry0-8,15)),col,
                   font_scale=0.45,bg=True)

    if scale != 1.0:
        nw,nh = int(out.shape[1]*scale), int(out.shape[0]*scale)
        out = cv2.resize(out,(nw,nh),interpolation=cv2.INTER_AREA)
    if output_path:
        cv2.imwrite(output_path, out)
        log.info("Full-page debug → %s", output_path)
    return out


# ── Drawing utilities ─────────────────────────────────────────────────────────

def _put_label(
    img: np.ndarray, text: str, pos: tuple[int,int],
    colour: tuple[int,int,int],
    font_scale: float = 0.55, thickness: int = 1, bg: bool = False,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    if bg:
        (tw,th),_ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img,(x-2,y-th-4),(x+tw+2,y+4),(20,20,20),-1)
    cv2.putText(img,text,(x,y),font,font_scale,colour,thickness,cv2.LINE_AA)


def _wrap(text: str, width: int) -> list[str]:
    words  = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur)+len(w)+1 <= width:
            cur = (cur+" "+w).strip()
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines or [""]


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    usage = (
        "Usage: python text_extractor.py <image> <x0> <y0> <x1> <y1> "
        "[--debug] [--no-llm] [--no-trocr] "
        "[--model <llm_path>] [--trocr-model <hf_id>]\n"
    )
    if len(sys.argv) < 6:
        print(usage); sys.exit(1)

    img_path       = sys.argv[1]
    x0,y0,x1,y1   = (int(v) for v in sys.argv[2:6])
    flags          = sys.argv[6:]
    debug_f        = "--debug"    in flags
    no_llm         = "--no-llm"   in flags
    no_trocr       = "--no-trocr" in flags
    model_p        = None
    trocr_m        = None

    if "--model"       in flags: model_p  = flags[flags.index("--model")+1]
    if "--trocr-model" in flags: trocr_m  = flags[flags.index("--trocr-model")+1]

    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"Cannot read: {img_path}")

    extractor = TextExtractor(
        llm_model_path=model_p,
        use_trocr=not no_trocr,
        trocr_model=trocr_m,
    )
    result = extractor.extract(img,(x0,y0,x1,y1),
                                debug=debug_f,
                                llm_postprocess=not no_llm)

    print(f"\n{'='*60}")
    print(f"  FINAL TEXT    : {result.text}")
    print(f"  RAW ASSEMBLY  : {result.raw_assembly}")
    print(f"  LINES         : {len(result.lines)}")
    for i,lr in enumerate(result.lines):
        print(f"    Line {i+1}: {lr.text!r}  ({len(lr.glyphs)} glyphs)")
    print(f"  LLM           : {result.llm_used} ({result.llm_backend})")
    print(f"{'='*60}\n")

    out_path = str(Path(img_path).with_stem(Path(img_path).stem + "_ocr_debug"))
    draw_ocr_debug(img,(x0,y0,x1,y1),result,output_path=out_path)
    print(f"Debug image → {out_path}")