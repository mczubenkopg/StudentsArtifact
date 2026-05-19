import cv2
import numpy as np
from typing import Optional

# ── PDF layout constants (must match the generator) ──────────────────────────
MM = 2.834645669  # 1 mm in PDF points
A4_W_PT = 595.2755906  # A4 width  in points
A4_H_PT = 841.8897638  # A4 height in points

MARGIN_L_PT = 5 * MM
MARGIN_R_PT = 5 * MM
MARGIN_T_PT = 5 * MM
MARGIN_B_PT = 5 * MM
FIDUCIAL_SIZE_PT = 5 * MM  # the black square side length


def _pdf_fiducial_corners_normalised() -> np.ndarray:
    """
    Returns the four fiducial *centre* positions as (x, y) fractions
    of (page_width, page_height), in image coordinate order:
        [top-left, top-right, bottom-left, bottom-right]

    PDF origin is bottom-left; image origin is top-left, so Y is flipped.
    """
    W, H = A4_W_PT, A4_H_PT
    SZ = FIDUCIAL_SIZE_PT

    # PDF rect(x, y, w, h) draws from bottom-left corner of the rect.
    # Centre of each fiducial:
    # Top-left fiducial:     x=MARGIN_L,              y=H-MARGIN_T-SZ
    # Top-right fiducial:    x=W-MARGIN_R-SZ,         y=H-MARGIN_T-SZ
    # Bottom-left fiducial:  x=MARGIN_L,              y=MARGIN_B
    # Bottom-right fiducial: x=W-MARGIN_R-SZ,         y=MARGIN_B

    pdf_pts = {
        "tl": (MARGIN_L_PT + SZ / 2, H - MARGIN_T_PT - SZ / 2),
        "tr": (W - MARGIN_R_PT - SZ / 2, H - MARGIN_T_PT - SZ / 2),
        "bl": (MARGIN_L_PT + SZ / 2, MARGIN_B_PT + SZ / 2),
        "br": (W - MARGIN_R_PT - SZ / 2, MARGIN_B_PT + SZ / 2),
    }

    # Convert to normalised image coords (flip Y: img_y = 1 - pdf_y/H)
    def to_img(px, py):
        return px / W, 1.0 - py / H

    return np.array([
        to_img(*pdf_pts["tl"]),  # index 0 → image top-left
        to_img(*pdf_pts["tr"]),  # index 1 → image top-right
        to_img(*pdf_pts["bl"]),  # index 2 → image bottom-left
        to_img(*pdf_pts["br"]),  # index 3 → image bottom-right
    ], dtype=np.float64)


def _find_fiducial_centres(
        gray: np.ndarray,
        min_area_frac: float = 0.0001,
        max_area_frac: float = 0.005,
        search_radius_frac: float = 0.15,
        dark_threshold: int = 80,
) -> Optional[np.ndarray]:
    """
    Detect the four black fiducial squares.

    Strategy:
      1. Threshold to isolate dark blobs.
      2. Find contours and keep candidates by area and squareness.
      3. Among all candidates, pick the four that best match the expected
         corner positions (top-left, top-right, bottom-left, bottom-right).

    Returns
    -------
    np.ndarray of shape (4, 2) with pixel (x, y) centres in order
    [top-left, top-right, bottom-left, bottom-right], or None on failure.
    """
    h, w = gray.shape
    img_area = h * w

    # Binarise: dark pixels → white blobs
    _, thresh = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area_frac * img_area < area < max_area_frac * img_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if aspect < 0.5:  # must be roughly square
            continue
        cx, cy = x + bw / 2, y + bh / 2
        candidates.append((cx, cy, area))

    if len(candidates) < 4:
        return None

    # Expected normalised centres of the four corners
    expected_norm = _pdf_fiducial_corners_normalised()  # shape (4, 2)
    # Absolute expected positions
    expected_abs = expected_norm * np.array([w, h])

    # For each expected corner, pick the nearest candidate
    cand_xy = np.array([[c[0], c[1]] for c in candidates])
    chosen = []
    used = set()
    for ex, ey in expected_abs:
        dists = np.linalg.norm(cand_xy - np.array([ex, ey]), axis=1)
        for idx in np.argsort(dists):
            if idx not in used:
                # Reject if it is too far away (> search_radius_frac of image diagonal)
                diag = np.hypot(w, h)
                if dists[idx] > search_radius_frac * diag:
                    break  # no candidate close enough for this corner
                chosen.append(cand_xy[idx])
                used.add(idx)
                break

    if len(chosen) != 4:
        return None

    return np.array(chosen, dtype=np.float32)  # TL, TR, BL, BR


# ── Main rectification function ───────────────────────────────────────────────

def rectify_image(
        image: np.ndarray,
        output_size: Optional[tuple[int, int]] = None,
        dark_threshold: int = 80,
        search_radius_frac: float = 0.15,
        return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Rectify a scanned image using the four black fiducial squares produced by
    ``draw_fiducials()``.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale image (as returned by cv2.imread).
    output_size : (width, height), optional
        Pixel dimensions of the output.  Defaults to A4 at 150 dpi
        (1240 × 1754 px).
    dark_threshold : int
        Pixels darker than this value (0-255) are treated as black.
        Lower = stricter; raise if fiducials are not detected on low-contrast
        scans.
    search_radius_frac : float
        Fraction of the image diagonal used as a search radius around each
        expected corner position.  Increase if the scan is heavily skewed.
    return_debug : bool
        If True, also return an annotated copy of the input image showing
        detected fiducial centres.

    Returns
    -------
    rectified : np.ndarray
        The perspective-corrected image (BGR).
    debug_img : np.ndarray  (only when return_debug=True)
        Input image with detected fiducial centres drawn on it.

    Raises
    ------
    RuntimeError
        If fewer than four fiducial squares can be found.
    """
    if output_size is None:
        # A4 at 150 dpi
        output_size = (1240, 1754)

    out_w, out_h = output_size

    # Work in grayscale for detection
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    centres = _find_fiducial_centres(
        gray,
        dark_threshold=dark_threshold,
        search_radius_frac=search_radius_frac,
    )

    if centres is None:
        raise RuntimeError(
            "Could not locate four fiducial squares in the image. "
            "Try adjusting dark_threshold or search_radius_frac."
        )

    # centres order: [TL, TR, BL, BR]  (pixel coords, float32)
    src_pts = centres

    # Destination points: the fiducial centres mapped to the output canvas.
    # We use the *same* normalised positions as the PDF spec.
    norm = _pdf_fiducial_corners_normalised()  # shape (4,2), [TL,TR,BL,BR]
    dst_pts = (norm * np.array([out_w, out_h])).astype(np.float32)

    # Perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    src_bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rectified = cv2.warpPerspective(src_bgr, M, (out_w, out_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

    if return_debug:
        debug = src_bgr.copy()
        colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        labels = ["TL", "TR", "BL", "BR"]
        for (cx, cy), col, lbl in zip(centres, colours, labels):
            cv2.circle(debug, (int(cx), int(cy)), 18, col, 4)
            cv2.putText(debug, lbl, (int(cx) + 12, int(cy) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        return rectified, debug

    return rectified