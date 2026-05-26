"""
text_extractor.py
-----------------
Extract handwritten / printed text from a region-of-interest (ROI) bbox in a
scanned document image.

Hard assumptions about the source text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  • Only uppercase Latin + Polish letters (A–Z, Ą Ć Ę Ł Ń Ó Ś Ź Ż).
  • No digits, punctuation, or special characters.

These assumptions drive three optimisations:
  1. Tesseract is configured with a whitelist covering exactly those characters
     and is used as the *primary* (always-on) engine.
  2. All post-processing strips every character outside that alphabet.
  3. The dictionary corrector only considers uppercase Polish words.

Engines (in priority order)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Tesseract    – primary; always enabled; PSM 6 & 11; uppercase whitelist
  2. RysOCR       – kacperwikiel/RysOCR LoRA on PaddleOCR-VL via PEFT;
                    best available for Polish diacritics
  3. EasyOCR      – secondary deep-learning engine (pl + en)
  4. TrOCR        – microsoft/trocr-base-handwritten transformer
  5. docTR        – document-understanding transformer pipeline
  6. RapidOCR     – fast ONNX engine
  7. PaddleOCR    – plain PaddleOCR without the LoRA (fallback)

Image enhancement variants fed to every engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  raw | denoised | contrast_stretched | clahe | adaptive_bin | upscaled_sharp

Merger & dictionary correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  All (engine × enhancement) candidates are:
    1. Filtered to the uppercase Polish alphabet.
    2. Scored with a Polish-coverage heuristic.
    3. The winning candidate is post-processed word-by-word through a
       Levenshtein-based closest-match lookup against a built-in Polish word
       list (extended from morfologik-python when available).

Debug
~~~~~
  draw_ocr_debug()       – annotate a single ROI on the page image
  draw_multi_ocr_debug() – annotate several ROIs in one pass

Dependencies (install what you need; missing engines are skipped silently):
    pip install opencv-python-headless numpy pillow
    pip install pytesseract                                  # + tesseract binary
    pip install easyocr
    pip install transformers torch torchvision peft          # RysOCR + TrOCR
    pip install python-doctr[torch]                          # docTR
    pip install rapidocr-onnxruntime
    pip install paddlepaddle paddleocr                       # plain PaddleOCR
    pip install morfologik-python                            # optional richer dict
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


# ── Character whitelist (the only characters we ever expect) ──────────────────

# Uppercase base Latin + all uppercase Polish diacritics + space
_ALLOWED_RE = re.compile(r"[^A-ZĄĆĘŁŃÓŚŹŻ ]", re.UNICODE)
_TOKEN_RE   = re.compile(r"[A-ZĄĆĘŁŃÓŚŹŻ]+", re.UNICODE)

# Tesseract character whitelist string (passed via config)
_TESS_WHITELIST = "AĄBCĆDEĘFGHIJKLŁMNOÓPRQSŚTUVWXYZŹŻ "


# ── Optional engine imports ───────────────────────────────────────────────────

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr as _easyocr
    _easyocr_reader: Optional[_easyocr.Reader] = None
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, logging as _hf_log
    import torch as _torch
    _trocr_processor: Optional[TrOCRProcessor] = None
    _trocr_model = None
    HAS_TROCR = True
except ImportError:
    HAS_TROCR = False

try:
    from doctr.models import ocr_predictor as _doctr_predictor_factory
    from doctr.io import DocumentFile as _DocFile
    _doctr_model = None
    HAS_DOCTR = True
except ImportError:
    HAS_DOCTR = False

try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    _rapid_engine: Optional[_RapidOCR] = None
    HAS_RAPIDOCR = True
except ImportError:
    HAS_RAPIDOCR = False

try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _paddle_engine: Optional[_PaddleOCR] = None
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

# RysOCR: PaddleOCR-VL base + kacperwikiel/RysOCR LoRA adapter
try:
    from peft import PeftModel as _PeftModel
    from transformers import (
        AutoProcessor as _AutoProcessor,
        AutoModelForCausalLM as _AutoModelForCausalLM,
    )
    _rysocr_model = None
    _rysocr_processor = None
    HAS_RYSOCR = True
except ImportError:
    HAS_RYSOCR = False


# ── Public data classes ───────────────────────────────────────────────────────

@dataclass
class EngineResult:
    """
    Raw OCR output from a single engine × enhancement combination.

    Attributes
    ----------
    engine : str
        Name of the OCR engine (e.g. 'tesseract_psm6', 'rysocr').
    enhancement : str
        Name of the image enhancement applied (e.g. 'clahe', 'upscaled_sharp').
    text : str
        Raw text returned by the engine (may still contain noise before
        the alphabet filter is applied in the merger).
    confidence : float
        Engine-reported confidence in [0, 1] where available; else -1.
    """
    engine: str
    enhancement: str
    text: str
    confidence: float = -1.0

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", "↵")
        return (f"EngineResult(engine={self.engine!r}, "
                f"enhancement={self.enhancement!r}, "
                f"confidence={self.confidence:.2f}, "
                f"text={preview!r})")


@dataclass
class OCRAnalysis:
    """
    Full result returned by ``extract_text_from_bbox``.

    Attributes
    ----------
    merged_text : str
        Final best-scoring text after filtering, merging, and dictionary
        correction.  Contains only uppercase Polish letters and spaces.
    corrected_text : str
        Word-by-word dictionary-corrected version of merged_text.
    roi_bbox : tuple[int, int, int, int]
        The ROI bounding box passed in (x0, y0, x1, y1).
    engine_results : list[EngineResult]
        All raw results from every (engine × enhancement) combination.
    best_engine : str
        Engine that produced the winning pre-correction candidate.
    best_enhancement : str
        Enhancement that produced the winning pre-correction candidate.
    polish_score : float
        Heuristic Polish-dictionary coverage score of the merged text.
    """
    merged_text: str
    corrected_text: str
    roi_bbox: tuple[int, int, int, int]
    engine_results: list[EngineResult] = field(default_factory=list)
    best_engine: str = ""
    best_enhancement: str = ""
    polish_score: float = 0.0

    def __repr__(self) -> str:
        preview = self.corrected_text[:80].replace("\n", "↵")
        return (f"OCRAnalysis(corrected_text={preview!r}, "
                f"best_engine={self.best_engine!r}, "
                f"polish_score={self.polish_score:.3f})")


# ── Polish word dictionary ────────────────────────────────────────────────────
# Seeded with high-frequency words + vocabulary from the sample survey forms.
# At module import time we attempt to extend it from morfologik-python.

_POLISH_DICT: set[str] = {
    # Prepositions / conjunctions / particles
    "W", "Z", "I", "NA", "DO", "SIĘ", "NIE", "TO", "JE", "GO", "A",
    "ZE", "PO", "TE", "TAM", "TEN", "TEJ", "TAK", "CO", "JAK", "OD",
    "ALE", "CZY", "BY", "ZA", "WE", "LUB", "GDY", "BO", "TU", "JUŻ",
    "JEJ", "ICH", "PAN", "ONI", "ONA", "ONO", "ORAZ", "PRZEZ", "PRZY",
    "JAKO", "JEST", "SĄ", "BYŁ", "BYŁA", "BYŁO", "BĘDZIE",
    # Survey-form domain words
    "OTWARTOŚĆ", "PROWADZĄCEGO", "KORYGOWANIE", "WYPOWIEDZI", "PISEMNYCH",
    "STUDENTÓW", "ZWRACANIE", "UWAGĘ", "CZĘSTO", "WYSTĘPUJĄCE", "BŁĘDY",
    "DUŻO", "INFORMACJI", "TEORETYCZNYCH", "ZWIĄZANYCH", "NAUKĄ",
    "JĘZYKU", "JĘZYKA", "GRAMATYKĄ", "GRAMATYCZNĄ", "GRAMATYCZNEJ",
    "WIEDZY", "ZACHĘCAĆ", "BARDZIEJ", "TWORZENIA", "WŁASNYCH",
    "PISANIA", "PRACY", "ANALIZA", "TEKSTÓW", "POPULARNONAUKOWYCH",
    "UDOSTĘPNIANYCH", "TWORZENIE", "FORMALNYM", "STYLU", "STYLEM",
    "ZWRÓCIĆ", "WIĘKSZĄ", "POPULARNE", "INTERPUNKCYJNE", "JĘZYKOWE",
    "POPEŁNIANE", "ZAANGAŻOWANIA", "GRUPY", "PRÓBA", "ZWIĘKSZENIA",
    "ZAINTERESOWANIA", "ELEMENTÓW", "MERYTORYCZNYCH", "NALEŻAŁOBY",
    "DODAĆ", "PRZEDMIOTU", "ZAJĘĆ", "SPOSOBIE", "POPRAWIĆ",
    "NAJMOCNIEJSZĄ", "NAJSŁABSZĄ", "STRONĘ", "STRON", "STRONĄ",
    "PRZYDATNE", "INSPIRUJĄCE", "NAJBARDZIEJ", "ILOŚĆ",
    "PRZEKAZYWANEJ", "ZMNIEJSZYĆ", "PROWADZENIA", "ORGANIZACJI",
    "ZAINTERESOWANIE", "ZAINTERESOWANIA", "BRAKU", "BRAK",
    "PRZYDATNYCH", "INFORMACJI", "ELEMENTY", "ELEMENTY",
    "POPULARNONAUKOWYCH", "UDOSTĘPNIANYCH", "WYPOWIEDZI",
    # General Polish vocabulary
    "PRACA", "NAUKA", "JĘZYK", "TEKST", "SŁOWO", "ZDANIE", "FORMA",
    "WIEDZA", "ĆWICZENIA", "PRZYKŁADY", "METODA", "TEMAT", "KURS",
    "ZAJĘCIA", "STUDENT", "PROWADZĄCY", "PRZEDMIOT", "OCENA",
    "OPIS", "ANALIZA", "WSTĘP", "KONIEC", "MATERIAŁ", "MATERIAŁY",
    "TREŚĆ", "TREŚCI", "ZADANIE", "ZADANIA", "PYTANIE", "PYTANIA",
    "ODPOWIEDŹ", "ODPOWIEDZI", "CZĘŚĆ", "SEKCJA", "STRONA",
    "UWAGI", "KOMENTARZ", "KOMENTARZE", "WYNIK", "WYNIKI",
}

# Attempt to bulk-load morfologik-python's Polish wordlist
try:
    import morfologik  # type: ignore
    _morph = morfologik.Analyzer()
    # morfologik does not expose a word list directly; we enrich the dict lazily
    # via _morph.analyse(word) during correction instead.
    HAS_MORFOLOGIK = True
except Exception:
    HAS_MORFOLOGIK = False
    _morph = None


def _is_polish_word(word: str) -> bool:
    """Return True if *word* (uppercase) is a valid Polish word."""
    if word in _POLISH_DICT:
        return True
    if HAS_MORFOLOGIK and _morph is not None:
        try:
            results = _morph.analyse(word.lower())
            if results:
                _POLISH_DICT.add(word)  # cache for next call
                return True
        except Exception:
            pass
    return False


# ── Polish scoring heuristics ─────────────────────────────────────────────────

_POLISH_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.UNICODE) for p in [
        r"[ĄĆĘŁŃÓŚŹŻ]",           # Polish diacritics present
        r"OWAĆ$|YWAĆ$",            # infinitive endings
        r"ENIA$|ANIE$|IENIE$",     # nominal endings
        r"NYCH$|OWEJ$|OWEGO$",     # genitive adjective endings
        r"ÓW$|OM$|AMI$",           # plural endings
    ]
]


def _polish_coverage(text: str) -> float:
    """
    Heuristic score [0, 1] measuring how well a text matches Polish uppercase.

    Combines:
      • fraction of tokens found in the Polish dictionary
      • presence of Polish diacritic characters
      • matches against common Polish morphological patterns
      • token-length distribution (Polish words average 6–8 letters)
    """
    if not text or not text.strip():
        return 0.0
    upper = text.upper().strip()
    tokens = _TOKEN_RE.findall(upper)
    if not tokens:
        return 0.0

    word_score      = sum(1 for t in tokens if _is_polish_word(t)) / len(tokens)
    diacritic_score = min(1.0, sum(1 for ch in upper if ch in "ĄĆĘŁŃÓŚŹŻ") / max(1, len(tokens)))
    pattern_score   = min(1.0,
        sum(1 for p in _POLISH_PATTERNS if any(p.search(t) for t in tokens))
        / len(_POLISH_PATTERNS)
    )
    avg_len      = float(np.mean([len(t) for t in tokens]))
    length_score = min(1.0, avg_len / 7.0)

    return (
        0.40 * word_score
        + 0.25 * diacritic_score
        + 0.20 * pattern_score
        + 0.15 * length_score
    )


# ── Polish dictionary corrector ───────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[lb]


def _closest_polish_word(token: str, max_distance: int = 2) -> str:
    """
    Return the closest word in _POLISH_DICT to *token* (uppercase).

    Returns *token* unchanged when:
      • an exact match exists, or
      • all dictionary candidates are farther than *max_distance*.

    Short tokens (≤ 2 letters) are never substituted to avoid false positives.
    """
    if len(token) <= 2 or _is_polish_word(token):
        return token

    best_word  = token
    best_dist  = max_distance + 1

    # Only compare against same-length ± 2 words for speed
    target_len = len(token)
    candidates = [
        w for w in _POLISH_DICT
        if abs(len(w) - target_len) <= 2
    ]

    for candidate in candidates:
        d = _levenshtein(token, candidate)
        if d < best_dist:
            best_dist = d
            best_word = candidate
            if d == 1:
                break  # close enough; stop early

    return best_word


def _correct_text(text: str) -> str:
    """
    Apply word-by-word dictionary correction to *text* (uppercase Polish).

    Only substitutes a token when a strictly closer dictionary word is found
    within Levenshtein distance 2.  Tokens already in the dictionary or
    shorter than 3 characters are left untouched.
    """
    if not text:
        return text
    corrected_tokens: list[str] = []
    for token in text.split():
        corrected_tokens.append(_closest_polish_word(token))
    return " ".join(corrected_tokens)


# ── Text normalisation ────────────────────────────────────────────────────────

def _filter_to_alphabet(text: str) -> str:
    """
    Strip every character outside the uppercase Polish alphabet + space.

    Multiple spaces are collapsed; the result is stripped of leading/trailing
    whitespace.
    """
    if not text:
        return ""
    upper = unicodedata.normalize("NFC", text).upper()
    filtered = _ALLOWED_RE.sub(" ", upper)
    filtered = re.sub(r" {2,}", " ", filtered)
    return filtered.strip()


# ── Image enhancement pipelines ──────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _enhance_raw(roi: np.ndarray) -> np.ndarray:
    return _to_gray(roi)


def _enhance_denoised(roi: np.ndarray) -> np.ndarray:
    gray = _to_gray(roi)
    return cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)


def _enhance_contrast_stretched(roi: np.ndarray) -> np.ndarray:
    gray = _to_gray(roi)
    mn, mx = int(gray.min()), int(gray.max())
    if mx == mn:
        return gray
    return ((gray.astype(np.float32) - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def _enhance_clahe(roi: np.ndarray) -> np.ndarray:
    gray = _to_gray(roi)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)


def _enhance_adaptive_bin(roi: np.ndarray) -> np.ndarray:
    gray    = _to_gray(roi)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )


def _enhance_upscaled_sharp(roi: np.ndarray) -> np.ndarray:
    gray  = _to_gray(roi)
    h, w  = gray.shape
    big   = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    blur  = cv2.GaussianBlur(big, (0, 0), 3)
    return cv2.addWeighted(big, 1.6, blur, -0.6, 0)


_ENHANCEMENTS: list[tuple[str, callable]] = [
    ("raw",                _enhance_raw),
    ("denoised",           _enhance_denoised),
    ("contrast_stretched", _enhance_contrast_stretched),
    ("clahe",              _enhance_clahe),
    ("adaptive_bin",       _enhance_adaptive_bin),
    ("upscaled_sharp",     _enhance_upscaled_sharp),
]


# ── Engine runners ────────────────────────────────────────────────────────────
# Every runner returns list[EngineResult].  Errors are silently swallowed so
# that one broken engine never aborts the whole pipeline.

# --- Tesseract (primary / always-on) ----------------------------------------

def _run_tesseract(enhanced: np.ndarray, enhancement_name: str) -> list[EngineResult]:
    """
    Primary engine.  Three PSM configurations tuned for uppercase-only text:
      PSM 6  – assume a single uniform block of text
      PSM 11 – sparse text, finds as much text as possible
      PSM 7  – single text line (useful for short boxes)

    The character whitelist restricts Tesseract to the allowed alphabet so no
    digit / punctuation noise can enter the output.
    """
    if not HAS_TESSERACT:
        return []

    results: list[EngineResult] = []
    whitelist_cfg = f"-c tessedit_char_whitelist={_TESS_WHITELIST}"
    configs = [
        ("pol", f"--psm 6  --oem 3 {whitelist_cfg}"),
        ("pol", f"--psm 11 --oem 3 {whitelist_cfg}"),
        ("pol", f"--psm 7  --oem 3 {whitelist_cfg}"),
    ]
    pil_img = Image.fromarray(enhanced)
    for lang, cfg in configs:
        try:
            raw  = pytesseract.image_to_string(pil_img, lang=lang, config=cfg)
            data = pytesseract.image_to_data(
                pil_img, lang=lang, config=cfg,
                output_type=pytesseract.Output.DICT,
            )
            confs = [c for c in data["conf"] if isinstance(c, (int, float)) and c != -1]
            conf  = float(np.mean(confs)) / 100.0 if confs else -1.0
            text  = raw.strip()
            if text:
                psm = re.search(r"--psm\s+(\d+)", cfg)
                label = f"tesseract_psm{psm.group(1) if psm else '?'}({lang})"
                results.append(EngineResult(
                    engine=label,
                    enhancement=enhancement_name,
                    text=text,
                    confidence=conf,
                ))
        except Exception:
            pass
    return results


# --- RysOCR (Polish LoRA on PaddleOCR-VL) ------------------------------------

_RYSOCR_PROMPT = (
    "Transcribe every word in this image exactly as written. "
    "Output only the transcribed text in uppercase Polish. "
    "Use only capital letters A-Z and Polish diacritics Ą Ć Ę Ł Ń Ó Ś Ź Ż. "
    "No punctuation, no digits, no explanations."
)


def _load_rysocr() -> bool:
    """Lazy-load RysOCR (base model + LoRA) into module-level singletons."""
    global _rysocr_model, _rysocr_processor
    if _rysocr_model is not None:
        return True
    if not HAS_RYSOCR:
        return False
    try:
        _hf_log.set_verbosity_error()
        base = _AutoModelForCausalLM.from_pretrained(
            "PaddlePaddle/PaddleOCR-VL",
            trust_remote_code=True,
        )
        _rysocr_model = _PeftModel.from_pretrained(base, "kacperwikiel/RysOCR")
        _rysocr_model.eval()
        _rysocr_processor = _AutoProcessor.from_pretrained(
            "PaddlePaddle/PaddleOCR-VL",
            trust_remote_code=True,
        )
        _hf_log.set_verbosity_warning()
        return True
    except Exception:
        _rysocr_model = None
        _rysocr_processor = None
        return False


def _run_rysocr(enhanced: np.ndarray, enhancement_name: str) -> list[EngineResult]:
    """
    RysOCR engine: PaddleOCR-VL base fine-tuned with a Polish LoRA adapter.
    Excels at correct diacritic restoration (ą/ę/ł/ó misread by other engines).
    Only runs on the 'raw' and 'clahe' enhancements to avoid redundant inference.
    """
    if not HAS_RYSOCR:
        return []
    # Limit to 2 enhancements for this heavier model
    if enhancement_name not in ("raw", "clahe"):
        return []
    if not _load_rysocr():
        return []
    try:
        import torch
        if enhanced.ndim == 2:
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text": _RYSOCR_PROMPT},
                ],
            }
        ]
        inputs = _rysocr_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output_ids = _rysocr_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        text = _rysocr_processor.decode(output_ids[0], skip_special_tokens=True).strip()
        # Strip everything before the last assistant turn marker if present
        if "\nassistant\n" in text.lower():
            text = text.split("\nassistant\n")[-1].strip()
        if text:
            return [EngineResult(
                engine="rysocr",
                enhancement=enhancement_name,
                text=text,
                confidence=-1.0,
            )]
    except Exception:
        pass
    return []


# --- EasyOCR -----------------------------------------------------------------

def _run_easyocr(enhanced: np.ndarray, enhancement_name: str) -> list[EngineResult]:
    if not HAS_EASYOCR:
        return []
    global _easyocr_reader
    try:
        if _easyocr_reader is None:
            _easyocr_reader = _easyocr.Reader(["pl", "en"], gpu=False, verbose=False)
        detections = _easyocr_reader.readtext(enhanced, detail=1, paragraph=False)
        if not detections:
            return []
        lines: list[str] = []
        confs: list[float] = []
        for _bbox, txt, conf in detections:
            if txt.strip():
                lines.append(txt.strip())
                confs.append(float(conf))
        text     = " ".join(lines)
        avg_conf = float(np.mean(confs)) if confs else -1.0
        if text:
            return [EngineResult(engine="easyocr", enhancement=enhancement_name,
                                 text=text, confidence=avg_conf)]
    except Exception:
        pass
    return []


# --- TrOCR -------------------------------------------------------------------

def _run_trocr(enhanced: np.ndarray, enhancement_name: str) -> list[EngineResult]:
    if not HAS_TROCR:
        return []
    global _trocr_processor, _trocr_model
    try:
        if _trocr_processor is None:
            _hf_log.set_verbosity_error()
            _trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-handwritten"
            )
            _trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten",
                ignore_mismatched_sizes=True,
            )
            _trocr_model.eval()
            _hf_log.set_verbosity_warning()

        if enhanced.ndim == 2:
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert("RGB")

        pixel_values = _trocr_processor(images=pil_img, return_tensors="pt").pixel_values
        with _torch.no_grad():
            generated_ids = _trocr_model.generate(pixel_values)
        text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text:
            return [EngineResult(engine="trocr", enhancement=enhancement_name,
                                 text=text, confidence=-1.0)]
    except Exception:
        pass
    return []


# --- docTR -------------------------------------------------------------------

def _run_doctr(enhanced: np.ndarray, enhancement_name: str) -> list[EngineResult]:
    if not HAS_DOCTR:
        return []
    global _doctr_model
    try:
        if _doctr_model is None:
            _doctr_model = _doctr_predictor_factory(pretrained=True)
        if enhanced.ndim == 2:
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        doc = _DocFile.from_images([rgb])
        result = _doctr_model(doc)
        lines: list[str] = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words if w.value.strip()]
                    if words:
                        lines.append(" ".join(words))
        text = "\n".join(lines).strip()
        if text:
            return [EngineResult(engine="doctr", enhancement=enhancement_name,
                                 text=text, confidence=-1.0)]
    except Exception:
        pass
    return []


# --- RapidOCR ----------------------------------------------------------------

def _run_rapidocr(enhanced: np.ndarray, enhancement_name: str) -> list[EngineResult]:
    if not HAS_RAPIDOCR:
        return []
    global _rapid_engine
    try:
        if _rapid_engine is None:
            _rapid_engine = _RapidOCR()
        result, _ = _rapid_engine(enhanced)
        if not result:
            return []
        lines: list[str] = []
        confs: list[float] = []
        for item in result:
            txt  = item[1].strip() if len(item) > 1 else ""
            conf = float(item[2])  if len(item) > 2 else -1.0
            if txt:
                lines.append(txt)
                confs.append(conf)
        text     = " ".join(lines)
        avg_conf = float(np.mean(confs)) if confs else -1.0
        if text:
            return [EngineResult(engine="rapidocr", enhancement=enhancement_name,
                                 text=text, confidence=avg_conf)]
    except Exception:
        pass
    return []


# --- PaddleOCR (plain, no LoRA) ----------------------------------------------

def _run_paddleocr(enhanced: np.ndarray, enhancement_name: str) -> list[EngineResult]:
    if not HAS_PADDLEOCR:
        return []
    global _paddle_engine
    try:
        if _paddle_engine is None:
            _paddle_engine = _PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = _paddle_engine.ocr(enhanced, cls=True)
        if not result or not result[0]:
            return []
        lines: list[str] = []
        confs: list[float] = []
        for item in result[0]:
            txt  = item[1][0].strip() if item[1] else ""
            conf = float(item[1][1])  if item[1] else -1.0
            if txt:
                lines.append(txt)
                confs.append(conf)
        text     = " ".join(lines)
        avg_conf = float(np.mean(confs)) if confs else -1.0
        if text:
            return [EngineResult(engine="paddleocr", enhancement=enhancement_name,
                                 text=text, confidence=avg_conf)]
    except Exception:
        pass
    return []


# Engine registry — Tesseract is first and always-on
_ENGINE_RUNNERS: list[tuple[str, callable]] = [
    ("tesseract", _run_tesseract),
    ("rysocr",    _run_rysocr),
    ("easyocr",   _run_easyocr),
    ("trocr",     _run_trocr),
    ("doctr",     _run_doctr),
    ("rapidocr",  _run_rapidocr),
    ("paddleocr", _run_paddleocr),
]


# ── Candidate merger ──────────────────────────────────────────────────────────

def _merge_candidates(results: list[EngineResult]) -> tuple[str, str, str, float]:
    """
    Select the best text from all (engine × enhancement) candidates.

    Steps
    -----
    1. Apply alphabet filter to every candidate.
    2. Score with _polish_coverage().
    3. Apply a length penalty for candidates below 60 % of the longest.
    4. Add a small confidence bonus where available.
    5. Tesseract and RysOCR results receive a 0.05 priority boost.

    Returns
    -------
    (merged_text, best_engine, best_enhancement, best_score)
    """
    if not results:
        return "", "", "", 0.0

    # Priority boost for our preferred engines
    _PRIORITY_ENGINES = {"tesseract_psm6(pol)", "tesseract_psm11(pol)",
                         "tesseract_psm7(pol)", "rysocr"}

    filtered: list[tuple[EngineResult, str, float]] = []
    for er in results:
        text  = _filter_to_alphabet(er.text)
        score = _polish_coverage(text)
        filtered.append((er, text, score))

    max_len = max((len(t) for _, t, _ in filtered), default=1)
    max_len = max(max_len, 1)

    scored: list[tuple[float, EngineResult, str]] = []
    for er, text, cov in filtered:
        length_ratio  = len(text) / max_len
        length_penalty = 0.0 if length_ratio >= 0.6 else (length_ratio - 0.6) * 1.5
        conf_bonus    = er.confidence * 0.05 if er.confidence >= 0 else 0.0
        priority_bonus = 0.05 if any(er.engine.startswith(e) for e in _PRIORITY_ENGINES) else 0.0
        total = cov + length_penalty + conf_bonus + priority_bonus
        scored.append((total, er, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_er, best_text = scored[0]
    return best_text, best_er.engine, best_er.enhancement, best_score


# ── Public function ───────────────────────────────────────────────────────────

def extract_text_from_bbox(
    image: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    *,
    engines: Optional[list[str]] = None,
    enhancements: Optional[list[str]] = None,
    mark_threshold: float = 0.0,
    apply_dictionary_correction: bool = True,
) -> OCRAnalysis:
    """
    Extract uppercase Polish text from a rectangular region of a scanned page.

    The input text is assumed to consist **exclusively** of uppercase Latin and
    Polish letters (A–Z, Ą Ć Ę Ł Ń Ó Ś Ź Ż).  Digits, punctuation, and any
    other characters are stripped from every candidate before scoring.

    Tesseract (PSM 6 / 11 / 7, ``pol`` language, uppercase whitelist) is the
    primary engine and always runs.  The RysOCR LoRA model, EasyOCR, TrOCR,
    docTR, RapidOCR, and PaddleOCR are used as supplementary engines when
    installed.

    After the best raw candidate is selected it is optionally post-processed
    with a Levenshtein-based Polish dictionary corrector that replaces
    obviously mis-read tokens with the closest known Polish word.

    Parameters
    ----------
    image : np.ndarray
        Full BGR (or grayscale) image of the scanned page.
    roi_bbox : tuple[int, int, int, int]
        Text ROI bounding box (x0, y0, x1, y1) in image pixel coordinates.
    engines : list[str] | None
        Restrict to specific engine names.  ``None`` = all available.
        Valid names: 'tesseract', 'rysocr', 'easyocr', 'trocr', 'doctr',
                     'rapidocr', 'paddleocr'.
    enhancements : list[str] | None
        Restrict to specific enhancement names.  ``None`` = all.
        Valid names: 'raw', 'denoised', 'contrast_stretched', 'clahe',
                     'adaptive_bin', 'upscaled_sharp'.
    mark_threshold : float
        Minimum Polish coverage score for the result to be non-empty.
        Default 0.0 always returns the best candidate found.
    apply_dictionary_correction : bool
        When True (default), apply the Levenshtein dictionary corrector to the
        merged text and populate ``OCRAnalysis.corrected_text``.

    Returns
    -------
    OCRAnalysis
        ``.merged_text``      – best-scoring raw text (alphabet-filtered)
        ``.corrected_text``   – dictionary-corrected version of merged_text
        ``.roi_bbox``         – ROI bbox passed in
        ``.engine_results``   – all EngineResult objects
        ``.best_engine``      – engine that produced the winning candidate
        ``.best_enhancement`` – enhancement that produced the winning candidate
        ``.polish_score``     – heuristic score of the merged text

    Raises
    ------
    ValueError
        If ``roi_bbox`` is degenerate (zero width or height).
    """
    x0, y0, x1, y1 = roi_bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Degenerate roi_bbox: {roi_bbox}")

    img_h, img_w = image.shape[:2]
    roi = image[max(0, y0): min(img_h, y1), max(0, x0): min(img_w, x1)]

    active_enhancements = [
        (name, fn) for name, fn in _ENHANCEMENTS
        if enhancements is None or name in enhancements
    ]
    active_runners = [
        (name, fn) for name, fn in _ENGINE_RUNNERS
        if engines is None or name in engines
    ]

    all_results: list[EngineResult] = []
    for enh_name, enh_fn in active_enhancements:
        try:
            enhanced = enh_fn(roi)
        except Exception:
            continue
        for _eng_name, runner in active_runners:
            try:
                all_results.extend(runner(enhanced, enh_name))
            except Exception:
                pass

    merged_text, best_engine, best_enhancement, polish_score = _merge_candidates(all_results)

    if polish_score < mark_threshold:
        merged_text = ""

    corrected_text = _correct_text(merged_text) if apply_dictionary_correction else merged_text

    return OCRAnalysis(
        merged_text=merged_text,
        corrected_text=corrected_text,
        roi_bbox=roi_bbox,
        engine_results=all_results,
        best_engine=best_engine,
        best_enhancement=best_enhancement,
        polish_score=polish_score,
    )


# ── Debug visualisation ───────────────────────────────────────────────────────

def draw_ocr_debug(
    image: np.ndarray,
    analysis: OCRAnalysis,
    *,
    colour_roi:   tuple[int, int, int] = (0, 180, 0),
    colour_text:  tuple[int, int, int] = (0, 0, 200),
    colour_label: tuple[int, int, int] = (0, 140, 0),
    colour_bg:    tuple[int, int, int] = (255, 255, 240),
    thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Return a copy of *image* with the ROI bbox and OCR result annotated.

    Annotations
    -----------
    Green rectangle  – ROI bounding box
    Green label      – engine, enhancement, and Polish score (above ROI)
    Blue banner      – corrected_text (wrapped, tinted background below ROI)

    Parameters match the style of ``draw_checkbox_debug`` in checkbox_analyzer.py.
    """
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    x0, y0, x1, y1 = analysis.roi_bbox
    cv2.rectangle(out, (x0, y0), (x1, y1), colour_roi, thickness)

    label   = (f"engine={analysis.best_engine}  "
               f"enh={analysis.best_enhancement}  "
               f"score={analysis.polish_score:.3f}")
    label_y = max(y0 - 6, 14)
    cv2.putText(out, label, (x0, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, colour_label, 1, cv2.LINE_AA)

    display_text = analysis.corrected_text or analysis.merged_text
    if display_text:
        max_chars = 65
        wrapped: list[str] = []
        for raw_line in display_text.split("\n"):
            current = ""
            for word in raw_line.split():
                if len(current) + len(word) + 1 > max_chars:
                    if current:
                        wrapped.append(current)
                    current = word
                else:
                    current = (current + " " + word).strip()
            if current:
                wrapped.append(current)

        line_h   = int(20 * font_scale / 0.45)
        banner_h = line_h * len(wrapped) + 8
        bx0, by0 = x0, y1 + 4
        bx1, by1 = min(out.shape[1], x0 + 720), by0 + banner_h

        if by1 <= out.shape[0] and bx1 <= out.shape[1] and by0 >= 0:
            sub = out[by0:by1, bx0:bx1]
            bg  = np.full_like(sub, colour_bg)
            cv2.addWeighted(bg, 0.55, sub, 0.45, 0, sub)
            out[by0:by1, bx0:bx1] = sub

        for i, line in enumerate(wrapped):
            ly = by0 + line_h * (i + 1)
            if ly >= out.shape[0]:
                break
            cv2.putText(out, line, (bx0 + 4, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        colour_text, 1, cv2.LINE_AA)

    return out


def draw_multi_ocr_debug(
    image: np.ndarray,
    analyses: list[OCRAnalysis],
    *,
    colours: Optional[list[tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Annotate multiple OCR analyses on a single debug image.

    Cycles through *colours* (6 defaults) when more analyses are provided.
    """
    _default_colours: list[tuple[int, int, int]] = [
        (0, 180, 0),
        (200, 100, 0),
        (0, 0, 200),
        (180, 0, 180),
        (0, 180, 180),
        (180, 180, 0),
    ]
    if colours is None:
        colours = _default_colours

    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, analysis in enumerate(analyses):
        colour = colours[i % len(colours)]
        out = draw_ocr_debug(
            out, analysis,
            colour_roi=colour,
            colour_text=colour,
            colour_label=colour,
            thickness=thickness,
            font_scale=font_scale,
        )
    return out


# ── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pathlib

    usage = (
        "Usage: python text_extractor.py <image> <x0> <y0> <x1> <y1> "
        "[engine,...] [enhancement,...]\n"
        "  x0 y0 x1 y1       = ROI bounding box in image pixels\n"
        "  engine,...         = comma-separated engine names (optional, default=all)\n"
        "  enhancement,...    = comma-separated enhancement names (optional, default=all)\n"
        "\nAvailable engines:\n"
        "  tesseract (primary/default), rysocr, easyocr, trocr, doctr, "
        "rapidocr, paddleocr\n"
        "Available enhancements:\n"
        "  raw, denoised, contrast_stretched, clahe, adaptive_bin, upscaled_sharp\n"
    )
    if len(sys.argv) < 6:
        print(usage)
        sys.exit(1)

    img_path = sys.argv[1]
    bbox     = tuple(int(v) for v in sys.argv[2:6])   # type: ignore[assignment]
    eng_arg  = sys.argv[6].split(",") if len(sys.argv) > 6 else None
    enh_arg  = sys.argv[7].split(",") if len(sys.argv) > 7 else None

    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"Cannot read image: {img_path}")

    analysis = extract_text_from_bbox(img, bbox, engines=eng_arg, enhancements=enh_arg)

    print(f"ROI bbox         : {analysis.roi_bbox}")
    print(f"Best engine      : {analysis.best_engine}")
    print(f"Best enhancement : {analysis.best_enhancement}")
    print(f"Polish score     : {analysis.polish_score:.4f}")
    print(f"Merged text      :\n  {analysis.merged_text}")
    print(f"Corrected text   :\n  {analysis.corrected_text}")

    if analysis.engine_results:
        print(f"\nTop 10 candidates (of {len(analysis.engine_results)} total):")
        ranked = sorted(
            analysis.engine_results,
            key=lambda e: _polish_coverage(_filter_to_alphabet(e.text)),
            reverse=True,
        )
        for er in ranked[:10]:
            preview = _filter_to_alphabet(er.text)[:70].replace("\n", "↵")
            score   = _polish_coverage(_filter_to_alphabet(er.text))
            print(f"  [{score:.3f}] {er.engine:35s} ({er.enhancement:20s}): {preview!r}")

    debug    = draw_ocr_debug(img, analysis)
    out_path = pathlib.Path(img_path).with_stem(pathlib.Path(img_path).stem + "_ocr_debug")
    cv2.imwrite(str(out_path), debug)
    print(f"\nDebug image saved → {out_path}")