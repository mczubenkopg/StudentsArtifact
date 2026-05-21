"""
fix_qrcode.py
-------------
Single function that exhausts every combination of image-enhancement
pipeline × QR decoder until it has collected the expected number of
unique QR codes from a colour image.

Each decoder returns per-code (payload, bbox_xyxy) pairs so bounding
boxes are always available in original image coordinates.

Decoders tried (whichever are installed):
  • cv2.QRCodeDetectorAruco         – fast, handles perspective distortion
  • cv2.QRCodeDetector              – OpenCV classic detector
  • pyzbar                          – zbar-based, excellent on clean binary images
  • zxingcpp / LocalAverage         – zxing-cpp 3.x, adaptive local binarizer (default)
  • zxingcpp / GlobalHistogram      – zxing-cpp 3.x, global histogram binarizer
  • zxingcpp / FixedThreshold       – zxing-cpp 3.x, fixed 127 threshold
  • zxingcpp / BoolCast             – zxing-cpp 3.x, raw bool cast (fastest)
  • qreader                         – deep-learning detector, handles damaged codes

Enhancement pipelines (applied in order, least → most destructive):
  raw_gray | global_otsu | adaptive | clahe_otsu | channel_best |
  saturation_mask | sharpened | morph_close | invert | upscale

Dependencies:
    pip install opencv-contrib-python numpy pyzbar pillow "zxingcpp>=3.0.0" qreader
    # pyzbar system lib:  apt-get install libzbar0  /  brew install zbar
    # qreader requires torch; see https://github.com/Eric-Canas/qreader
    # zxingcpp 3.x requires a C++20 toolchain when building from source;
    #   pre-built wheels are available on PyPI for all major platforms.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Optional decoder imports ──────────────────────────────────────────────────

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

try:
    import zxingcpp as _zxingcpp
    # Verify it is >= 3.x by checking the Binarizer enum exists
    _zxingcpp.Binarizer.LocalAverage  # raises AttributeError on older versions
    HAS_ZXINGCPP = True
except (ImportError, AttributeError):
    HAS_ZXINGCPP = False

try:
    from qreader import QReader as _QReader
    _qreader_instance: Optional[_QReader] = None   # lazy singleton
    HAS_QREADER = True
except ImportError:
    HAS_QREADER = False


# ── Public result class ───────────────────────────────────────────────────────

@dataclass
class QRResult:
    """
    A single decoded QR code.

    Attributes
    ----------
    payload : str
        Decoded text content of the QR code (UTF-8).
    bbox : tuple[int, int, int, int]
        Bounding box in original image coordinates as (x1, y1, x2, y2)
        where (x1, y1) is the top-left corner and (x2, y2) is the
        bottom-right corner.  None when the decoder did not return
        positional information.
    enhancement : str
        Name of the enhancement pipeline that produced the readable image.
    decoder : str
        Name of the decoder that found this code.
    """
    payload: str
    bbox: Optional[tuple[int, int, int, int]]   # (x1, y1, x2, y2) or None
    enhancement: str
    decoder: str

    def __repr__(self) -> str:
        bbox_s = f"({self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]})" \
                 if self.bbox else "None"
        return (f"QRResult(payload={self.payload!r}, bbox={bbox_s}, "
                f"enhancement={self.enhancement!r}, decoder={self.decoder!r})")


# ── Internal detection result ─────────────────────────────────────────────────

@dataclass
class _Detection:
    """Raw (payload, optional bbox) as returned by a decoder, before scaling."""
    payload: str
    bbox: Optional[tuple[int, int, int, int]]   # xyxy in enhanced-image coords


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _points_to_xyxy(points) -> Optional[tuple[int, int, int, int]]:
    """Convert an arbitrary point collection (polygon / corners) to xyxy."""
    try:
        arr = np.array(points, dtype=float).reshape(-1, 2)
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def _scale_bbox(
    bbox: Optional[tuple[int, int, int, int]],
    scale_x: float,
    scale_y: float,
) -> Optional[tuple[int, int, int, int]]:
    """Scale a bbox back from enhanced-image coordinates to original ones."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 / scale_x)),
        int(round(y1 / scale_y)),
        int(round(x2 / scale_x)),
        int(round(y2 / scale_y)),
    )


# ── Enhancement pipelines ─────────────────────────────────────────────────────
# Each returns (enhanced_image, scale_x, scale_y).
# scale_x / scale_y record how much the image was resized relative to the
# original so bboxes can be mapped back to original coordinates.

def _enh_raw_gray(img: np.ndarray):
    return _to_gray(img), 1.0, 1.0

def _enh_global_otsu(img: np.ndarray):
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, 1.0, 1.0

def _enh_adaptive(img: np.ndarray):
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    return binary, 1.0, 1.0

def _enh_clahe_otsu(img: np.ndarray):
    gray = _to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    binary = cv2.threshold(clahe.apply(gray), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary, 1.0, 1.0

def _enh_channel_best(img: np.ndarray):
    if img.ndim == 2:
        return _enh_global_otsu(img)
    best = max(cv2.split(img[:, :, :3]), key=lambda ch: float(ch.std()))
    binary = cv2.threshold(best, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary, 1.0, 1.0

def _enh_saturation_mask(img: np.ndarray):
    if img.ndim == 2:
        return _enh_global_otsu(img)
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].copy()
    v[hsv[:, :, 1] > 80] = 255
    binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary, 1.0, 1.0

def _enh_sharpened(img: np.ndarray):
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary, 1.0, 1.0

def _enh_morph_close(img: np.ndarray):
    gray = _to_gray(img)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return closed, 1.0, 1.0

def _enh_invert(img: np.ndarray):
    gray = _to_gray(img)
    binary = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return binary, 1.0, 1.0

def _enh_upscale(img: np.ndarray):
    """2-3× upscale; returns scale factors so bboxes can be mapped back."""
    gray = _to_gray(img)
    oh, ow = gray.shape
    scale = 3 if max(oh, ow) < 200 else 2
    big = cv2.resize(gray, (ow * scale, oh * scale), interpolation=cv2.INTER_CUBIC)
    binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary, float(scale), float(scale)   # bbox / scale → original coords


_ENHANCEMENTS: list[tuple[str, callable]] = [
    ("raw_gray",        _enh_raw_gray),
    ("global_otsu",     _enh_global_otsu),
    ("clahe_otsu",      _enh_clahe_otsu),
    ("adaptive",        _enh_adaptive),
    ("channel_best",    _enh_channel_best),
    ("saturation_mask", _enh_saturation_mask),
    ("sharpened",       _enh_sharpened),
    ("morph_close",     _enh_morph_close),
    ("invert",          _enh_invert),
    ("upscale",         _enh_upscale),
]


# ── Per-decoder functions ─────────────────────────────────────────────────────
# Each returns list[_Detection].  bbox is in enhanced-image pixel coordinates.

def _decode_cv2_aruco(img: np.ndarray) -> list[_Detection]:
    """cv2.QRCodeDetectorAruco – returns per-code corner polygons."""
    try:
        detector = cv2.QRCodeDetectorAruco()
        data, points, _ = detector.detectAndDecodeMulti(img)
        if not data:
            return []
        out = []
        for text, pts in zip(data, points if points is not None else [None] * len(data)):
            if not text:
                continue
            bbox = _points_to_xyxy(pts) if pts is not None else None
            out.append(_Detection(payload=text, bbox=bbox))
        return out
    except Exception:
        return []


def _decode_cv2_classic(img: np.ndarray) -> list[_Detection]:
    """cv2.QRCodeDetector – standard OpenCV decoder."""
    try:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecodeMulti(img)
        if not data:
            return []
        out = []
        for text, pts in zip(data, points if points is not None else [None] * len(data)):
            if not text:
                continue
            bbox = _points_to_xyxy(pts) if pts is not None else None
            out.append(_Detection(payload=text, bbox=bbox))
        return out
    except Exception:
        return []


def _decode_pyzbar(img: np.ndarray) -> list[_Detection]:
    """pyzbar – zbar library; rect attribute gives (left, top, w, h)."""
    if not HAS_PYZBAR:
        return []
    try:
        out = []
        for obj in _pyzbar_decode(img.astype(np.uint8)):
            text = obj.data.decode("utf-8", errors="replace")
            r = obj.rect                          # Rect(left, top, width, height)
            bbox = (r.left, r.top, r.left + r.width, r.top + r.height)
            out.append(_Detection(payload=text, bbox=bbox))
        return out
    except Exception:
        return []


def _make_zxingcpp_decoder(binarizer_name: str):
    """
    Factory: returns a decoder that calls zxingcpp.read_barcodes with the
    given Binarizer.  The position object exposes four named corners from
    which an xyxy bbox is derived.

    zxingcpp 3.x Binarizer options
    --------------------------------
    LocalAverage    – adaptive local threshold (default)
    GlobalHistogram – fast global histogram
    FixedThreshold  – fixed 127 threshold (for pre-binarised input)
    BoolCast        – raw bool cast (fastest, cleanest images)
    """
    def _decoder(img: np.ndarray) -> list[_Detection]:
        if not HAS_ZXINGCPP:
            return []
        try:
            binarizer = getattr(_zxingcpp.Binarizer, binarizer_name)
            results = _zxingcpp.read_barcodes(
                img,
                binarizer=binarizer,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
            )
            out = []
            for r in results:
                if not r.valid or not r.text:
                    continue
                # position has .top_left / .top_right / .bottom_left / .bottom_right
                # each with .x and .y attributes
                try:
                    pos = r.position
                    corners = [
                        (pos.top_left.x,     pos.top_left.y),
                        (pos.top_right.x,    pos.top_right.y),
                        (pos.bottom_right.x, pos.bottom_right.y),
                        (pos.bottom_left.x,  pos.bottom_left.y),
                    ]
                    bbox = _points_to_xyxy(corners)
                except Exception:
                    bbox = None
                out.append(_Detection(payload=r.text, bbox=bbox))
            return out
        except Exception:
            return []

    _decoder.__name__ = f"_decode_zxingcpp_{binarizer_name.lower()}"
    _decoder.__doc__ = f"zxingcpp 3.x – Binarizer.{binarizer_name}"
    return _decoder


def _decode_qreader(img: np.ndarray) -> list[_Detection]:
    """QReader – deep-learning detector; returns (text, bbox) pairs."""
    if not HAS_QREADER:
        return []
    global _qreader_instance
    try:
        if _qreader_instance is None:
            _qreader_instance = _QReader()
        bgr = _ensure_bgr(img)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # detect_and_decode returns texts; get_detail returns bbox per code
        texts, bboxes = _qreader_instance.detect_and_decode(
            image=rgb, return_detections=True
        )
        out = []
        for text, det in zip(texts or [], bboxes or []):
            if not text:
                continue
            # det is (x1, y1, x2, y2) or a dict with 'bbox_xyxy'
            try:
                if isinstance(det, dict):
                    b = det["bbox_xyxy"]
                    bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                else:
                    bbox = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
            except Exception:
                bbox = None
            out.append(_Detection(payload=text, bbox=bbox))
        return out
    except Exception:
        return []


_DECODERS: list[tuple[str, callable]] = [
    ("zxingcpp_local_avg",    _make_zxingcpp_decoder("LocalAverage")),
    ("zxingcpp_global_hist",  _make_zxingcpp_decoder("GlobalHistogram")),
    ("zxingcpp_fixed_thresh", _make_zxingcpp_decoder("FixedThreshold")),
    ("zxingcpp_bool_cast",    _make_zxingcpp_decoder("BoolCast")),
    ("cv2_aruco",             _decode_cv2_aruco),
    ("cv2_classic",           _decode_cv2_classic),
    ("pyzbar",                _decode_pyzbar),
    ("qreader",               _decode_qreader),
]


# ── Public function ───────────────────────────────────────────────────────────

def decode_qrcodes(
    image: np.ndarray,
    expected_count: int = 1,
) -> list[QRResult]:
    """
    Decode QR codes from a colour image, exhausting all combinations of
    enhancement pipelines and decoders until ``expected_count`` unique
    payloads have been collected.

    Parameters
    ----------
    image : np.ndarray
        BGR (or grayscale / BGRA) image as returned by ``cv2.imread``.
    expected_count : int
        Number of distinct QR codes expected in the image.  The function
        stops as soon as this many unique payloads have been found.
        Pass ``0`` to run all 80 combinations and collect everything found.

    Returns
    -------
    list[QRResult]
        One ``QRResult`` per unique payload, in discovery order.
        Each result carries:
          .payload     – decoded string
          .bbox        – (x1, y1, x2, y2) in original image pixels, or None
          .enhancement – pipeline name that produced the readable image
          .decoder     – decoder name that found this code

    Notes
    -----
    Search order: for each enhancement pipeline, every decoder is tried
    before moving on.  Bboxes returned by the upscale pipeline are
    automatically mapped back to original image coordinates.
    Duplicates (same payload) are silently ignored after first discovery.

    Available decoders (used only when installed):
      cv2_aruco, cv2_classic, pyzbar,
      zxingcpp_local_avg, zxingcpp_global_hist, zxingcpp_fixed_thresh,
      zxingcpp_bool_cast, qreader

    Available enhancement pipelines:
      raw_gray, global_otsu, adaptive, clahe_otsu, channel_best,
      saturation_mask, sharpened, morph_close, invert, upscale
    """
    seen: set[str] = set()
    results: list[QRResult] = []

    for enh_name, enh_fn in _ENHANCEMENTS:
        try:
            enhanced, scale_x, scale_y = enh_fn(image)
        except Exception:
            continue

        for dec_name, dec_fn in _DECODERS:
            try:
                detections = dec_fn(enhanced)
            except Exception:
                continue

            for det in detections:
                payload = det.payload.strip()
                if not payload or payload in seen:
                    continue
                seen.add(payload)

                # Map bbox from enhanced-image space back to original
                bbox = _scale_bbox(det.bbox, scale_x, scale_y)

                results.append(QRResult(
                    payload=payload,
                    bbox=bbox,
                    enhancement=enh_name,
                    decoder=dec_name,
                ))

            if expected_count > 0 and len(results) >= expected_count:
                return results

    return results


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pathlib

    if len(sys.argv) < 2:
        print("Usage: python fix_qrcode.py <image_path> [expected_count]")
        print("  expected_count defaults to 1")
        sys.exit(1)

    path = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read image: {path}")

    found = decode_qrcodes(img, expected_count=count)

    print(f"Found: {len(found)} / {count}")
    for i, r in enumerate(found, 1):
        print(f"  [{i}] payload={r.payload!r}")
        print(f"       bbox={r.bbox}  ({r.enhancement} + {r.decoder})")

        # Draw bbox on a debug copy
        if r.bbox:
            debug = img.copy()
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug, r.payload[:30], (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            out = pathlib.Path(path).with_stem(
                pathlib.Path(path).stem + f"_debug{i}"
            )
            cv2.imwrite(str(out), debug)
            print(f"       → debug image saved: {out}")

    if len(found) < count:
        print(f"\nWARNING: only {len(found)} of {count} codes found.")
        sys.exit(2)