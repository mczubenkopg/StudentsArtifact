import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
).eval()


def recognize_handwriting(image: np.ndarray) -> str:
    """
    Recognize handwritten Polish text from a NumPy image.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3) or grayscale image (H, W).

    Returns
    -------
    str
        Transcribed text.
    """

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L").convert("RGB")
    elif image.ndim == 3:
        pil_image = Image.fromarray(image)
    else:
        raise ValueError("Expected image shape (H, W) or (H, W, 3)")

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
                        "Do not correct grammar.\n"
                        "If a word is unreadable write [illegible].\n"
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
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    outputs = outputs[:, inputs.input_ids.shape[1]:]

    return processor.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()