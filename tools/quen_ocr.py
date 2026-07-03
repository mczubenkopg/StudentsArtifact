"""
qwen_ocr.py
-----------
OCR for handwritten Polish form images using Qwen2.5-VL.

The model is chosen automatically based on available RAM, VRAM, and disk space.
Models (all Qwen2.5-VL-Instruct variants, largest-first):
    72B  → ~145 GB disk / ~80 GB RAM
    32B  → ~65  GB disk / ~35 GB RAM
     7B  → ~15  GB disk /  ~8 GB RAM   ← typical laptop choice
     3B  → ~6   GB disk /  ~4 GB RAM   ← minimum

Usage:
    import cv2
    from qwen_ocr import recognize

    img = cv2.imread("form.png")          # BGR np.ndarray
    text = recognize(img)
    print(text)
"""

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Model catalogue  (id, min_ram_gb, disk_gb)
# ---------------------------------------------------------------------------
MODELS = [
    ("Qwen/Qwen2.5-VL-72B-Instruct",  80.0, 145.0),
    ("Qwen/Qwen2.5-VL-32B-Instruct",  35.0,  65.0),
    ("Qwen/Qwen2.5-VL-7B-Instruct",    8.0,  15.0),
    ("Qwen/Qwen2.5-VL-3B-Instruct",    4.0,   6.5),
]

OCR_PROMPT = (
    "This is a scanned handwritten Polish form field with five lines. "
    "Read all handwritten text exactly as written. "
    "Return only the text, nothing else."
)

# Module-level cache so the model survives repeated calls in the same process
_processor: Optional[AutoProcessor] = None
_model: Optional[Qwen2_5_VLForConditionalGeneration] = None
_loaded_model_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Resource detection
# ---------------------------------------------------------------------------

def _available_ram_gb() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def _available_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def _free_disk_gb() -> float:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    cache.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(cache).free / 1024 ** 3


def _pick_model() -> str:
    """Return the largest model id that fits in available resources."""
    ram  = _available_ram_gb()
    vram = _available_vram_gb()
    disk = _free_disk_gb()
    # Use whichever is larger — GPU VRAM or CPU RAM
    compute = max(ram, vram)

    log.info(f"RAM {ram:.1f} GB  |  GPU VRAM {vram:.1f} GB  |  Disk {disk:.1f} GB")

    for model_id, min_ram, min_disk in MODELS:
        if compute >= min_ram and disk >= min_disk * 1.05:   # 5 % headroom
            log.info(f"Selected model: {model_id}")
            return model_id

    raise RuntimeError(
        "No suitable model fits in available resources. "
        f"Need at least {MODELS[-1][1]} GB RAM and {MODELS[-1][2]} GB disk."
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load(model_id: str, hf_token: Optional[str] = None) -> None:
    """Load processor + model into module-level cache (no-op if already loaded)."""
    global _processor, _model, _loaded_model_id

    if _loaded_model_id == model_id:
        return

    # Unload previous model if a different one was cached
    if _model is not None:
        unload()

    # Disable the xet chunked-transfer protocol — it stalls silently
    os.environ["HF_HUB_DISABLE_XET"]    = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    vram = _available_vram_gb()
    _, min_ram, _ = next(m for m in MODELS if m[0] == model_id)
    on_gpu = vram >= min_ram * 0.9

    dtype      = torch.float16 if on_gpu else torch.bfloat16
    device_map = "cuda:0" if on_gpu else "cpu"

    log.info(f"Loading {model_id} on {'GPU' if on_gpu else 'CPU'} …")

    load_kw: dict = dict(
        torch_dtype=dtype,
        device_map=device_map,
        token=hf_token,
    )

    # 4-bit quantisation on CPU halves RAM usage (needs bitsandbytes)
    if not on_gpu:
        try:
            import bitsandbytes  # noqa: F401
            load_kw.update(load_in_4bit=True, device_map="auto")
            log.info("4-bit quantisation enabled")
        except ImportError:
            pass

    _processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    _model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, **load_kw
    )
    _loaded_model_id = model_id
    log.info("Model ready.")


def unload() -> None:
    """Release the cached model from memory."""
    global _processor, _model, _loaded_model_id
    if _model is not None:
        del _model, _processor
        _model = _processor = _loaded_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Model unloaded.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize(
    image: np.ndarray,
    hf_token: Optional[str] = None,
    model_id:  Optional[str] = None,
) -> str:
    """
    Recognise handwritten text in a form-field image.

    Parameters
    ----------
    image     : np.ndarray  — BGR image (e.g. from cv2.imread)
    hf_token  : str, optional  — HuggingFace token for faster downloads
    model_id  : str, optional  — override auto-selection (must be a Qwen2.5-VL id)

    Returns
    -------
    str  — recognised text (may be empty if the field is blank)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")

    # Resolve token: argument → env var → huggingface-cli login
    token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass

    selected = model_id or _pick_model()
    _load(selected, hf_token=token)

    # Convert BGR → RGB PIL image
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text",  "text":  OCR_PROMPT},
            ],
        }
    ]

    from qwen_vl_utils import process_vision_info

    text_input = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=128)

    # Strip the prompt tokens; decode only newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    result = _processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Qwen2.5-VL OCR for handwritten Polish forms")
    p.add_argument("image",      help="Path to the form image")
    p.add_argument("--token",    default=None, help="HuggingFace token (or set HF_TOKEN)")
    p.add_argument("--model-id", default=None, help="Override model selection")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    print(recognize(img, hf_token=args.token, model_id=args.model_id))