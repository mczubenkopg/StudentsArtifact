"""
Polish Handwritten OCR Pipeline
- Tesseract for initial OCR
- Qwen (via Ollama) for Polish text correction
- Auto-selects model size based on available RAM/VRAM/disk
"""

import subprocess
import shutil
import numpy as np
import psutil
import pytesseract
from PIL import Image


# ---------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------

def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def get_available_vram_gb() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            free_mb = int(result.stdout.strip().split("\n")[0])
            return free_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


def get_available_disk_gb() -> float:
    usage = shutil.disk_usage("/")
    return usage.free / (1024 ** 3)


# ---------------------------------------------------------------------------
# Model selection
# Model sizes (approx. disk / RAM needed):
#   qwen2.5:0.5b  ~0.4 GB disk,  ~1.5 GB RAM
#   qwen2.5:1.5b  ~1.1 GB disk,  ~3.0 GB RAM
#   qwen2.5:3b    ~2.0 GB disk,  ~5.0 GB RAM
#   qwen2.5:7b    ~4.7 GB disk,  ~9.0 GB RAM
# ---------------------------------------------------------------------------

MODELS = [
    {"name": "qwen2.5:7b",  "disk_gb": 5.0, "ram_gb": 9.0},
    {"name": "qwen2.5:3b",  "disk_gb": 2.5, "ram_gb": 5.0},
    {"name": "qwen2.5:1.5b","disk_gb": 1.5, "ram_gb": 3.0},
    {"name": "qwen2.5:0.5b","disk_gb": 0.8, "ram_gb": 1.5},
]


def select_model() -> str:
    ram   = get_available_ram_gb()
    vram  = get_available_vram_gb()
    disk  = get_available_disk_gb()
    # Use whichever memory pool is larger for inference
    mem   = max(ram, vram)

    print(f"[Resources] RAM: {ram:.1f} GB | VRAM: {vram:.1f} GB | Disk: {disk:.1f} GB")

    for model in MODELS:
        if disk >= model["disk_gb"] and mem >= model["ram_gb"]:
            print(f"[Model]     Selected → {model['name']}")
            return model["name"]

    raise RuntimeError(
        "Insufficient resources for the smallest Qwen model. "
        "Free at least 0.8 GB disk and 1.5 GB RAM."
    )


# ---------------------------------------------------------------------------
# Ensure the chosen model is available locally (pull if needed)
# ---------------------------------------------------------------------------

def ensure_model(model_name: str) -> None:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model_name not in result.stdout:
        print(f"[Ollama]    Pulling {model_name} (this may take a while)…")
        subprocess.run(["ollama", "pull", model_name], check=True)
    else:
        print(f"[Ollama]    {model_name} already present.")


# ---------------------------------------------------------------------------
# Tesseract OCR
# ---------------------------------------------------------------------------

def ocr_with_tesseract(image: np.ndarray) -> str:
    pil_img = Image.fromarray(image)
    # OEM 1 = LSTM, PSM 6 = single block of text (good for multi-line forms)
    config = "--oem 1 --psm 6 -l pol"
    raw = pytesseract.image_to_string(pil_img, config=config)
    return raw.strip()


# ---------------------------------------------------------------------------
# Qwen correction via Ollama
# ---------------------------------------------------------------------------

def fix_with_qwen(raw_text: str, model_name: str) -> str:
    prompt = (
        "Jesteś ekspertem języka polskiego. Poniższy tekst pochodzi z OCR "
        "odręcznego pisma i może zawierać błędy. Popraw go tak, aby był "
        "poprawny gramatycznie i ortograficznie. Zwróć TYLKO poprawiony tekst, "
        "bez żadnych komentarzy ani objaśnień.\n\n"
        f"Tekst do poprawy:\n{raw_text}"
    )

    result = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")

    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize_handwritten_polish(image: np.ndarray) -> str:
    """
    Recognize handwritten Polish text from a numpy image array (5 lines).

    Args:
        image: np.ndarray  (H x W) grayscale or (H x W x 3) RGB

    Returns:
        Corrected Polish text string.
    """
    # 1. Pick & prepare model
    model_name = select_model()
    ensure_model(model_name)

    # 2. Raw OCR
    print("[OCR]       Running Tesseract…")
    raw_text = ocr_with_tesseract(image)
    print(f"[OCR]       Raw output:\n{raw_text}\n")
    return raw_text
    # 3. LLM correction
    # print("[Qwen]      Fixing Polish text…")
    # corrected = fix_with_qwen(raw_text, model_name)
    # print(f"[Qwen]      Corrected output:\n{corrected}\n")

    # return corrected


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a blank white test image (5 lines placeholder)
    dummy_image = np.ones((250, 800, 3), dtype=np.uint8) * 255
    text = recognize_handwritten_polish(dummy_image)
    print("=== Final result ===")
    print(text)