"""
analyse_checkboxes.py
---------------------
Locate and read 6 checkboxes arranged in a single horizontal line
immediately to the right of a QR code, then return the single most-marked
checkbox.

Layout (mirrors the PDF generator):
  QR  | gap |  1  | gap |  2  | gap |  3  | gap |  4  | gap |  5  | gap |  6

Exactly one checkbox is assumed to be marked.  The function scores every
box by its interior dark-pixel ratio and returns the one with the highest
score, provided it exceeds `mark_threshold` (default 0.50 = 50 %).

Dependencies:
    pip install opencv-python-headless numpy
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Public data classes ───────────────────────────────────────────────────────

@dataclass
class CheckboxResult:
    """
    Analysis result for a single checkbox.

    Attributes
    ----------
    index : int
        1-based checkbox number (1 … n).
    bbox : tuple[int, int, int, int]
        Bounding box (x1, y1, x2, y2) in image pixel coordinates.
    dark_ratio : float
        Fraction of interior pixels that are dark (0.0 – 1.0).
    marked : bool
        True only for the single highest-scoring box when it exceeds
        mark_threshold.
    """
    index: int
    bbox: tuple[int, int, int, int]
    dark_ratio: float
    marked: bool

    def __repr__(self) -> str:
        state = "MARKED" if self.marked else "empty"
        return (f"CheckboxResult(index={self.index}, bbox={self.bbox}, "
                f"dark_ratio={self.dark_ratio:.3f}, state={state})")


@dataclass
class CheckboxAnalysis:
    """
    Full result returned by ``analyse_checkboxes``.

    Attributes
    ----------
    checkboxes : list[CheckboxResult]
        One entry per checkbox in index order; exactly one has marked=True
        when a clear winner is found.
    marked_index : int | None
        1-based index of the marked checkbox, or None if no box exceeds
        mark_threshold.
    qr_bbox : tuple[int, int, int, int]
        The QR code bbox used as the layout anchor.
    qr_side_px : int
        The effective QR side length in pixels that was actually used
        (either derived from qr_bbox or supplied via avg_qr_size_px).
    """
    checkboxes: list[CheckboxResult]
    marked_index: Optional[int]
    qr_bbox: tuple[int, int, int, int]
    qr_side_px: int

    def __repr__(self) -> str:
        return (f"CheckboxAnalysis(marked_index={self.marked_index}, "
                f"qr_side_px={self.qr_side_px}, "
                f"checkboxes={self.checkboxes})")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Dark pixels → 255 (foreground), using global Otsu threshold."""
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _dark_ratio(binary_roi: np.ndarray, shrink: float = 0.15) -> float:
    """
    Fraction of foreground (dark) pixels in the *interior* of a
    pre-binarised ROI.  The border (``shrink`` × side) is stripped on
    every edge to exclude the checkbox stroke from the measurement.
    """
    h, w = binary_roi.shape
    pad_x = max(1, int(w * shrink))
    pad_y = max(1, int(h * shrink))
    interior = binary_roi[pad_y: h - pad_y, pad_x: w - pad_x]
    if interior.size == 0:
        return 0.0
    return float(np.count_nonzero(interior)) / interior.size


def _checkbox_bboxes(
    qr_bbox: tuple[int, int, int, int],
    qr_side_px: int,
    n: int,
    checkbox_size_ratio: float,
    gap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """
    Derive n checkbox bounding boxes from the QR anchor and effective QR side.

    The checkbox side and gap are computed from ``qr_side_px`` (which may
    differ from ``qr_bbox`` width when an external average size is supplied).
    Checkboxes are centred vertically within the QR row height.
    """
    qx1, qy1, qx2, qy2 = qr_bbox
    qr_h = qy2 - qy1

    cb_side = int(round(qr_side_px * checkbox_size_ratio))
    gap     = int(round(qr_side_px * gap_ratio))
    step    = cb_side + gap

    # Vertical centring within the QR row
    v_offset = (qr_h - cb_side) // 2
    cb_y1    = qy1 + v_offset
    cb_y2    = cb_y1 + cb_side

    # Horizontal start: one gap to the right of the QR right edge
    first_x = qx2 + gap

    return [
        (first_x + i * step, cb_y1, first_x + i * step + cb_side, cb_y2)
        for i in range(n)
    ]


# ── Public function ───────────────────────────────────────────────────────────

def analyse_checkboxes(
    image: np.ndarray,
    qr_bbox: tuple[int, int, int, int],
    *,
    n: int = 6,
    checkbox_size_ratio: float = 7 / 9,
    gap_ratio: float = 4.15 / 9,
    mark_threshold: float = 0.50,
    shrink: float = 0.15,
    avg_qr_size_px: Optional[int] = None,
) -> CheckboxAnalysis:
    """
    Find the single marked checkbox in a row of ``n`` boxes next to a QR code.

    The function scores every checkbox by its interior dark-pixel ratio and
    returns the one with the highest score.  Exactly one checkbox is assumed
    to be filled; ``marked_index`` is None only when the best score is still
    below ``mark_threshold``.

    Parameters
    ----------
    image : np.ndarray
        Full BGR (or grayscale) image of the scanned page.
    qr_bbox : tuple[int, int, int, int]
        Bounding box of the QR code (x1, y1, x2, y2) in image pixels.
        Comes from ``fix_qrcode.decode_qrcodes()`` → ``QRResult.bbox``.
    n : int
        Number of checkboxes to analyse (default 6).
    checkbox_size_ratio : float
        Checkbox side as a fraction of the *effective* QR side.
        Default 7/9 ≈ 0.778.
    gap_ratio : float
        Gap between consecutive boxes as a fraction of the *effective* QR side.
        Default 4.15/9 ≈ 0.461.
    mark_threshold : float
        Minimum dark-pixel ratio in the interior of a checkbox to consider it
        marked (default 0.50 = 50 %).  Only the highest-scoring box is ever
        returned as marked; this threshold rules out fully empty pages.
    shrink : float
        Border fraction stripped before measuring interior darkness, to exclude
        the checkbox stroke (default 0.15 = 15 % per side).
    avg_qr_size_px : int | None
        When provided, overrides the QR side derived from ``qr_bbox`` and uses
        this fixed pixel size instead.  Useful when the detected QR bbox is
        noisy and you have a reliable average QR size from a calibration step
        or from the rectified page geometry.

    Returns
    -------
    CheckboxAnalysis
        ``.checkboxes``   – all n CheckboxResult objects in index order
        ``.marked_index`` – 1-based index of the winning box, or None
        ``.qr_bbox``      – the anchor bbox that was passed in
        ``.qr_side_px``   – effective QR side that was used for geometry

    Raises
    ------
    ValueError
        If the QR bbox is degenerate (zero width or height).
    """
    qx1, qy1, qx2, qy2 = qr_bbox
    if qx2 <= qx1 or qy2 <= qy1:
        raise ValueError(f"Degenerate qr_bbox: {qr_bbox}")

    # Effective QR side: external average takes priority over detected width
    qr_side_px: int = avg_qr_size_px if avg_qr_size_px is not None else (qx2 - qx1)

    img_h, img_w = image.shape[:2]
    binary = _binarise(_to_gray(image))

    bboxes = _checkbox_bboxes(qr_bbox, qr_side_px, n, checkbox_size_ratio, gap_ratio)

    scores: list[tuple[int, float, tuple[int, int, int, int]]] = []  # (idx, ratio, bbox)
    for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(img_w, x2), min(img_h, y2)

        if cx2 <= cx1 or cy2 <= cy1:
            scores.append((idx, 0.0, (x1, y1, x2, y2)))
            continue

        roi   = binary[cy1:cy2, cx1:cx2]
        ratio = _dark_ratio(roi, shrink=shrink)
        scores.append((idx, ratio, (x1, y1, x2, y2)))

    # Winner = single highest-scoring box
    best_idx, best_ratio, _ = max(scores, key=lambda s: s[1])
    winner_found = best_ratio >= mark_threshold

    checkboxes = [
        CheckboxResult(
            index=idx,
            bbox=bbox,
            dark_ratio=ratio,
            marked=(winner_found and idx == best_idx),
        )
        for idx, ratio, bbox in scores
    ]

    return CheckboxAnalysis(
        checkboxes=checkboxes,
        marked_index=best_idx if winner_found else None,
        qr_bbox=qr_bbox,
        qr_side_px=qr_side_px,
    )


# ── Optional debug visualisation ──────────────────────────────────────────────

def draw_checkbox_debug(
    image: np.ndarray,
    analysis: CheckboxAnalysis,
    *,
    colour_marked: tuple[int, int, int] = (0, 200, 0),
    colour_empty:  tuple[int, int, int] = (180, 180, 180),
    colour_qr:     tuple[int, int, int] = (0, 120, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Return a copy of *image* with the QR bbox and all checkbox bboxes drawn.

    Green  = marked (winning) checkbox
    Grey   = empty checkbox
    Blue   = QR code anchor
    """
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    qx1, qy1, qx2, qy2 = analysis.qr_bbox
    cv2.rectangle(out, (qx1, qy1), (qx2, qy2), colour_qr, thickness)
    cv2.putText(out, f"QR({analysis.qr_side_px}px)", (qx1, qy1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour_qr, 1)

    for cb in analysis.checkboxes:
        colour = colour_marked if cb.marked else colour_empty
        x1, y1, x2, y2 = cb.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(out, str(cb.index), (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        cv2.putText(out, f"{cb.dark_ratio:.2f}", (x1 + 2, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)

    return out


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pathlib

    usage = (
        "Usage: python analyse_checkboxes.py <image> <x1> <y1> <x2> <y2> "
        "[avg_qr_size_px]\n"
        "  x1 y1 x2 y2      = QR code bounding box in image pixels\n"
        "  avg_qr_size_px    = override QR side length in pixels (optional)\n"
    )
    if len(sys.argv) < 6:
        print(usage)
        sys.exit(1)

    path          = sys.argv[1]
    qr_box        = tuple(int(v) for v in sys.argv[2:6])
    avg_qr_size   = int(sys.argv[6]) if len(sys.argv) > 6 else None

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read image: {path}")

    analysis = analyse_checkboxes(img, qr_box, avg_qr_size_px=avg_qr_size)

    print(f"QR bbox        : {analysis.qr_bbox}")
    print(f"QR side used   : {analysis.qr_side_px} px")
    print(f"Marked index   : {analysis.marked_index}")
    print()
    for cb in analysis.checkboxes:
        state = "✓ MARKED" if cb.marked else "  empty"
        print(f"  Checkbox {cb.index}: {state}  dark_ratio={cb.dark_ratio:.3f}  bbox={cb.bbox}")

    debug    = draw_checkbox_debug(img, analysis)
    out_path = pathlib.Path(path).with_stem(pathlib.Path(path).stem + "_checkboxes")
    cv2.imwrite(str(out_path), debug)
    print(f"\nDebug image saved → {out_path}")