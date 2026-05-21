"""
fix_qrcode.py
-------------
Single function that exhausts every combination of image-enhancement
pipeline × QR decoder until it has collected the expected number of
unique QR payloads from a colour image.

Decoders tried (whichever are installed):
  • cv2.QRCodeDetectorAruco  – fast, handles perspective distortion
  • cv2.QRCodeDetector       – OpenCV classic detector
  • pyzbar                   – zbar-based, excellent on clean binary images
  • qreader                  – deep-learning detector, handles damaged codes

Enhancement pipelines (applied in order, least → most destructive):
  raw_gray | global_otsu | adaptive | clahe_otsu | channel_best |
  saturation_mask | sharpened | morph_close | invert | upscale

The function iterates over every (enhancement, decoder) pair and stops
as soon as the running pool of unique payloads reaches `expected_count`.
If all combinations are exhausted, it returns whatever was found.

Dependencies:
    pip install opencv-contrib-python numpy pyzbar pillow qreader
    # pyzbar system lib:  apt-get install libzbar0  /  brew install zbar
    # qreader requires torch; see https://github.com/Eric-Canas/qreader
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Optional decoder imports ──────────────────────────────────────────────────

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

try:
    from qreader import QReader as _QReader
    _qreader_instance: Optional[_QReader] = None   # lazy singleton
    HAS_QREADER = True
except ImportError:
    HAS_QREADER = False


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QRResult:
    """
    Aggregated result returned by ``decode_qrcodes``.

    Attributes
    ----------
    payloads : list[str]
        Unique decoded strings, in discovery order.
    found : int
        Number of unique payloads found.
    expected : int
        Target number of codes that was requested.
    complete : bool
        True when ``found >= expected``.
    hits : list[dict]
        Per-hit detail records:
        ``{"payload": str, "enhancement": str, "decoder": str, "image": ndarray}``
    attempts : int
        Total (enhancement × decoder) combinations tried before stopping.
    """
    payloads: list[str]
    found: int
    expected: int
    complete: bool
    hits: list[dict] = field(default_factory=list)
    attempts: int = 0


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


# ── Enhancement pipelines ─────────────────────────────────────────────────────

def _enh_raw_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale only – no further processing."""
    return _to_gray(img)

def _enh_global_otsu(img: np.ndarray) -> np.ndarray:
    """Global Otsu binarisation."""
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def _enh_adaptive(img: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian threshold – handles uneven lighting."""
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

def _enh_clahe_otsu(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast equalisation, then Otsu."""
    gray = _to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.threshold(clahe.apply(gray), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def _enh_channel_best(img: np.ndarray) -> np.ndarray:
    """Pick the colour channel with the highest contrast, then Otsu."""
    if img.ndim == 2:
        return _enh_global_otsu(img)
    best = max(cv2.split(img[:, :, :3]), key=lambda ch: float(ch.std()))
    return cv2.threshold(best, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def _enh_saturation_mask(img: np.ndarray) -> np.ndarray:
    """Zero-out vivid-coloured pixels (background suppression), then Otsu."""
    if img.ndim == 2:
        return _enh_global_otsu(img)
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].copy()
    v[hsv[:, :, 1] > 80] = 255          # high-saturation → white
    return cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def _enh_sharpened(img: np.ndarray) -> np.ndarray:
    """Unsharp-mask sharpening, then Otsu."""
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def _enh_morph_close(img: np.ndarray) -> np.ndarray:
    """Otsu + morphological closing – fills broken QR modules."""
    binary = _enh_global_otsu(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

def _enh_invert(img: np.ndarray) -> np.ndarray:
    """Inverted Otsu – for light-on-dark or colour-inverted codes."""
    gray = _to_gray(img)
    return cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def _enh_upscale(img: np.ndarray) -> np.ndarray:
    """2-3× upscale for tiny / low-resolution codes, then Otsu."""
    gray = _to_gray(img)
    h, w = gray.shape
    scale = 3 if max(h, w) < 200 else 2
    big = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Ordered list – least destructive first so we stop as early as possible
_ENHANCEMENTS: list[tuple[str, callable]] = [
    ("raw_gray",        _enh_raw_gray),
    ("global_otsu",     _enh_global_otsu),
    ("adaptive",        _enh_adaptive),
    ("clahe_otsu",      _enh_clahe_otsu),
    ("channel_best",    _enh_channel_best),
    ("saturation_mask", _enh_saturation_mask),
    ("sharpened",       _enh_sharpened),
    ("morph_close",     _enh_morph_close),
    ("invert",          _enh_invert),
    ("upscale",         _enh_upscale),
]


# ── Per-decoder decode functions ──────────────────────────────────────────────

def _decode_cv2_aruco(img: np.ndarray) -> list[str]:
    """cv2.QRCodeDetectorAruco – handles perspective-distorted codes."""
    try:
        detector = cv2.QRCodeDetectorAruco()
        data, _, _ = detector.detectAndDecodeMulti(img)
        return [s for s in (data or []) if s]
    except Exception:
        return []

def _decode_cv2_classic(img: np.ndarray) -> list[str]:
    """cv2.QRCodeDetector – standard OpenCV decoder."""
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecodeMulti(img)
        return [s for s in (data or []) if s]
    except Exception:
        return []

def _decode_pyzbar(img: np.ndarray) -> list[str]:
    """pyzbar – zbar library; excellent on clean binary images."""
    if not HAS_PYZBAR:
        return []
    try:
        results = _pyzbar_decode(img.astype(np.uint8))
        return [obj.data.decode("utf-8", errors="replace") for obj in results]
    except Exception:
        return []

def _decode_qreader(img: np.ndarray) -> list[str]:
    """QReader – deep-learning detector; best on damaged/rotated codes."""
    if not HAS_QREADER:
        return []
    global _qreader_instance
    try:
        if _qreader_instance is None:
            _qreader_instance = _QReader()
        bgr = _ensure_bgr(img)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = _qreader_instance.detect_and_decode(image=rgb)
        return [r for r in (results or []) if r]
    except Exception:
        return []


_DECODERS: list[tuple[str, callable]] = [
    ("cv2_aruco",  _decode_cv2_aruco),
    ("cv2_classic", _decode_cv2_classic),
    ("pyzbar",     _decode_pyzbar),
    ("qreader",    _decode_qreader),
]


# ── Public function ───────────────────────────────────────────────────────────

def decode_qrcodes(
    image: np.ndarray,
    expected_count: int = 1,
) -> QRResult:
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
        stops as soon as this many unique payloads are found.
        Pass ``0`` to run every combination and collect everything found.

    Returns
    -------
    QRResult
        ``.payloads``  – unique decoded strings (UTF-8), discovery order
        ``.found``     – how many unique codes were found
        ``.expected``  – the requested target
        ``.complete``  – True when found >= expected (or expected == 0)
        ``.hits``      – per-hit detail: enhancement, decoder, image used
        ``.attempts``  – total (enhancement × decoder) combos tried

    Notes
    -----
    The search order is: for each enhancement pipeline, try every decoder
    before moving to the next pipeline.  Within each decoder call, all
    returned codes are added to the pool; duplicates are silently ignored.
    Early exit occurs as soon as the pool reaches ``expected_count``.

    Available decoders (used only when installed):
      cv2_aruco, cv2_classic, pyzbar, qreader

    Available enhancement pipelines (always available via OpenCV/NumPy):
      raw_gray, global_otsu, adaptive, clahe_otsu, channel_best,
      saturation_mask, sharpened, morph_close, invert, upscale
    """
    seen: set[str] = set()        # unique payload pool
    hits: list[dict] = []
    attempts = 0

    for enh_name, enh_fn in _ENHANCEMENTS:
        # Build the enhanced image once per pipeline
        try:
            enhanced = enh_fn(image)
        except Exception:
            continue

        for dec_name, dec_fn in _DECODERS:
            attempts += 1

            try:
                raw_payloads = dec_fn(enhanced)
            except Exception:
                continue

            for payload in raw_payloads:
                payload = payload.strip()
                if payload and payload not in seen:
                    seen.add(payload)
                    hits.append({
                        "payload":     payload,
                        "enhancement": enh_name,
                        "decoder":     dec_name,
                        "image":       enhanced,
                    })

            # Early exit once we have enough unique codes
            if expected_count > 0 and len(seen) >= expected_count:
                return QRResult(
                    payloads=list(seen),
                    found=len(seen),
                    expected=expected_count,
                    complete=True,
                    hits=hits,
                    attempts=attempts,
                )

    return QRResult(
        payloads=list(seen),
        found=len(seen),
        expected=expected_count,
        complete=(expected_count == 0 or len(seen) >= expected_count),
        hits=hits,
        attempts=attempts,
    )


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

    result = decode_qrcodes(img, expected_count=count)

    print(f"Found    : {result.found} / {result.expected}  (complete={result.complete})")
    print(f"Attempts : {result.attempts}")
    for i, p in enumerate(result.payloads, 1):
        hit = next(h for h in result.hits if h["payload"] == p)
        print(f"  [{i}] {p!r}  via {hit['enhancement']} + {hit['decoder']}")
        out = pathlib.Path(path).with_stem(
            pathlib.Path(path).stem + f"_enh{i}_{hit['enhancement']}"
        )
        cv2.imwrite(str(out), hit["image"])
        print(f"       → enhanced image saved: {out}")

    if not result.complete:
        print(f"\nWARNING: only {result.found} of {result.expected} codes found.")
        sys.exit(2)