"""
analyse_checkboxes.py
---------------------
Locate and read 6 checkboxes that are arranged in a single horizontal
line immediately to the right of a QR code.

Layout assumption (mirrors the PDF generator):
  • The QR code and every checkbox have the same square side length.
  • CHECKBOX_GAP between consecutive boxes equals half the box side.
  • The checkbox row shares the same top edge as the QR code.
  • Boxes are numbered 1-6 left-to-right, starting right after the QR code.

  QR  | gap |  1  | gap |  2  | gap |  3  | gap |  4  | gap |  5  | gap |  6

A checkbox is considered MARKED when its interior dark-pixel ratio
exceeds `mark_threshold` (default 0.10 = 10 %).

The QR code bounding box (x1, y1, x2, y2) is the only positional input
required; all checkbox positions are derived from it.

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
        1-based checkbox number (1 … 6).
    bbox : tuple[int, int, int, int]
        Bounding box (x1, y1, x2, y2) in image pixel coordinates.
    dark_ratio : float
        Fraction of interior pixels that are dark (0.0 – 1.0).
    marked : bool
        True when dark_ratio >= mark_threshold.
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
        One entry per checkbox, always 6 items in index order.
    marked_index : int | None
        1-based index of the marked checkbox, or None if none / ambiguous.
    marked_count : int
        Total number of checkboxes exceeding the mark threshold.
    qr_bbox : tuple[int, int, int, int]
        The QR code bbox that was used as the layout anchor.
    """
    checkboxes: list[CheckboxResult]
    marked_index: Optional[int]
    marked_count: int
    qr_bbox: tuple[int, int, int, int]

    def __repr__(self) -> str:
        return (f"CheckboxAnalysis(marked_index={self.marked_index}, "
                f"marked_count={self.marked_count}, "
                f"checkboxes={self.checkboxes})")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """
    Produce a binary image where dark pixels = 255 (foreground).
    Uses Otsu on the local region for robustness against lighting variation.
    """
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return binary


def _dark_ratio(binary_roi: np.ndarray, shrink: float = 0.15) -> float:
    """
    Compute the fraction of foreground (dark) pixels in the interior of
    a pre-binarised ROI.

    ``shrink`` removes a border of that fraction of the box width/height
    to exclude the checkbox border stroke itself from the measurement.
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
    n: int = 6,
    gap_ratio: float = 0.5,
    checkbox_size_ratio: float = 5 / 9,
) -> list[tuple[int, int, int, int]]:
    """
    Derive the n checkbox bounding boxes from the QR code bbox.

    Parameters
    ----------
    qr_bbox : (x1, y1, x2, y2)
        QR code position in image pixels.
    n : int
        Number of checkboxes (default 6).
    gap_ratio : float
        Gap between boxes expressed as a fraction of the *checkbox* side.
        Matches CHECKBOX_GAP = 0.5 × CHECKBOX_SIZE from the PDF generator.
    checkbox_size_ratio : float
        Checkbox side length as a fraction of the QR code side length.
        Default 5/9 ≈ 0.556 (CHECKBOX_SIZE = 5 mm, QR ≈ 9 mm).
        Checkboxes are centred vertically within the QR row.

    Returns
    -------
    list of (x1, y1, x2, y2) for checkboxes 1 … n.
    """
    qx1, qy1, qx2, qy2 = qr_bbox
    qr_side = qx2 - qx1
    qr_h    = qy2 - qy1

    cb_side = int(round(qr_side * checkbox_size_ratio))
    gap     = int(round(cb_side * gap_ratio))
    step    = cb_side + gap                      # left-edge advance per box

    # Vertical centre alignment within the QR row
    v_offset = (qr_h - cb_side) // 2
    cb_y1    = qy1 + v_offset
    cb_y2    = cb_y1 + cb_side

    # First checkbox starts one gap to the right of the QR code
    first_x = qx2 + gap

    bboxes = []
    for i in range(n):
        x1 = first_x + i * step
        x2 = x1 + cb_side
        bboxes.append((x1, cb_y1, x2, cb_y2))
    return bboxes


# ── Public function ───────────────────────────────────────────────────────────

def analyse_checkboxes(
    image: np.ndarray,
    qr_bbox: tuple[int, int, int, int],
    *,
    n: int = 6,
    checkbox_size_ratio: float = 5 / 9,
    gap_ratio: float = 0.5,
    mark_threshold: float = 0.10,
    shrink: float = 0.15,
) -> CheckboxAnalysis:
    """
    Detect which of the ``n`` checkboxes next to a QR code is marked.

    Parameters
    ----------
    image : np.ndarray
        Full BGR (or grayscale) image of the scanned page.
    qr_bbox : tuple[int, int, int, int]
        Bounding box of the QR code in image pixel coordinates (x1, y1, x2, y2).
        Typically comes from ``fix_qrcode.decode_qrcodes()``:
            ``result.bbox`` on the returned ``QRResult``.
    n : int
        Number of checkboxes to analyse (default 6).
    checkbox_size_ratio : float
        Checkbox side length as a fraction of the QR code side length.
        Default 5/9 ≈ 0.556, matching CHECKBOX_SIZE = 5 mm against a
        QR code of ≈ 9 mm as generated by the PDF layout code.
        Checkboxes are centred vertically within the QR row.
    gap_ratio : float
        Gap between boxes as a fraction of the *checkbox* side (default 0.5).
        Matches CHECKBOX_GAP = 2.5 mm, CHECKBOX_SIZE = 5 mm → ratio = 0.5.
    mark_threshold : float
        Minimum dark-pixel fraction inside a checkbox interior to consider
        it marked (default 0.10 = 10 %).
    shrink : float
        Border fraction stripped before measuring darkness, to exclude the
        checkbox stroke itself (default 0.15 = 15 % on each side).

    Returns
    -------
    CheckboxAnalysis
        ``.checkboxes``    – list of CheckboxResult, one per box (1-indexed)
        ``.marked_index``  – 1-based index of the single marked box, or None
        ``.marked_count``  – total number of marked boxes found
        ``.qr_bbox``       – the anchor bbox that was used

    Raises
    ------
    ValueError
        If the QR bbox is degenerate (zero width or height).
    """
    qx1, qy1, qx2, qy2 = qr_bbox
    if qx2 <= qx1 or qy2 <= qy1:
        raise ValueError(f"Degenerate qr_bbox: {qr_bbox}")

    img_h, img_w = image.shape[:2]
    gray   = _to_gray(image)

    # Build one global binary for consistent thresholding across all boxes
    binary = _binarise(gray)

    bboxes = _checkbox_bboxes(
        qr_bbox,
        n=n,
        gap_ratio=gap_ratio,
        checkbox_size_ratio=checkbox_size_ratio,
    )
    checkboxes: list[CheckboxResult] = []

    for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        # Clamp to image bounds
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(img_w, x2)
        cy2 = min(img_h, y2)

        if cx2 <= cx1 or cy2 <= cy1:
            # Checkbox is outside the image — treat as empty
            checkboxes.append(CheckboxResult(
                index=idx,
                bbox=(x1, y1, x2, y2),
                dark_ratio=0.0,
                marked=False,
            ))
            continue

        roi    = binary[cy1:cy2, cx1:cx2]
        ratio  = _dark_ratio(roi, shrink=shrink)
        marked = ratio >= mark_threshold

        checkboxes.append(CheckboxResult(
            index=idx,
            bbox=(x1, y1, x2, y2),
            dark_ratio=ratio,
            marked=marked,
        ))

    marked_boxes  = [cb for cb in checkboxes if cb.marked]
    marked_count  = len(marked_boxes)
    marked_index  = marked_boxes[0].index if marked_count == 1 else None

    return CheckboxAnalysis(
        checkboxes=checkboxes,
        marked_index=marked_index,
        marked_count=marked_count,
        qr_bbox=qr_bbox,
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
    Return a copy of *image* with QR bbox and all checkbox bboxes drawn.

    Green  = marked checkbox
    Grey   = empty checkbox
    Orange = QR code anchor
    """
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw QR anchor
    qx1, qy1, qx2, qy2 = analysis.qr_bbox
    cv2.rectangle(out, (qx1, qy1), (qx2, qy2), colour_qr, thickness)

    cv2.putText(out, f"{qx2-qx1}, {qy1-qy2}", (qx1, qy1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour_qr, 1)

    for cb in analysis.checkboxes:
        colour = colour_marked if cb.marked else colour_empty
        x1, y1, x2, y2 = cb.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

        # Index label above the box
        label = str(cb.index)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

        # Dark-ratio below the box
        ratio_label = f"{cb.dark_ratio:.2f}"
        cv2.putText(out, ratio_label, (x1 + 2, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)

    return out


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pathlib

    if len(sys.argv) < 6:
        print("Usage: python analyse_checkboxes.py <image> <x1> <y1> <x2> <y2> [cb_size_ratio]")
        print("  x1 y1 x2 y2      = QR code bounding box in image pixels")
        print("  cb_size_ratio     = checkbox side / QR side  (default 0.5556 = 5/9)")
        sys.exit(1)

    path               = sys.argv[1]
    qr_box             = tuple(int(v) for v in sys.argv[2:6])
    cb_size_ratio      = float(sys.argv[6]) if len(sys.argv) > 6 else 5 / 9

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read image: {path}")

    analysis = analyse_checkboxes(img, qr_box, checkbox_size_ratio=cb_size_ratio)

    print(f"QR bbox            : {analysis.qr_bbox}")
    print(f"Checkbox size ratio: {cb_size_ratio:.4f}")
    print(f"Marked count       : {analysis.marked_count}")
    print(f"Marked index       : {analysis.marked_index}")
    print()
    for cb in analysis.checkboxes:
        state = "✓ MARKED" if cb.marked else "  empty"
        print(f"  Checkbox {cb.index}: {state}  dark_ratio={cb.dark_ratio:.3f}  bbox={cb.bbox}")

    # Save debug image
    debug = draw_checkbox_debug(img, analysis)
    out_path = pathlib.Path(path).with_stem(pathlib.Path(path).stem + "_checkboxes")
    cv2.imwrite(str(out_path), debug)
    print(f"\nDebug image saved → {out_path}")