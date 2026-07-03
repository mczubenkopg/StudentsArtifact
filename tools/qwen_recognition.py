import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE = torch.device("cuda:0")

assert torch.cuda.is_available(), "CUDA is not available."

DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    trust_remote_code=True,
).to(DEVICE)

model.eval()

# -----------------------------------------------------------------------------
# OCR
# -----------------------------------------------------------------------------

def recognize_handwriting(image: np.ndarray) -> str:
    """
    Parameters
    ----------
    image : np.ndarray
        RGB or grayscale image.

    Returns
    -------
    str
        Handwritten Polish transcription.
    """

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        pil_image = Image.fromarray(image).convert("RGB")
    elif image.ndim == 3:
        pil_image = Image.fromarray(image)
    else:
        raise ValueError("Expected image shape (H,W) or (H,W,3)")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "text",
                    "text": (
                        "You are an OCR engine.\n"
                        "Transcribe the handwritten Polish text exactly.\n"
                        "Preserve line breaks.\n"
                        "Preserve spelling mistakes.\n"
                        "Preserve punctuation.\n"
                        "Do not translate.\n"
                        "Do not correct grammar.\n"
                        "If a word cannot be read write [illegible].\n"
                        "Output ONLY the transcription."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt",
    )

    # Move every tensor to GPU
    inputs = {
        k: v.to(device=DEVICE)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )

    generated = generated[:, inputs["input_ids"].shape[1]:]

    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(f"Recognized text: {text.strip()}")
    return text.strip()