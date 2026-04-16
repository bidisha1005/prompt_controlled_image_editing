"""
InstructPix2Pix Image Editing Engine
Wraps timbrooks/instruct-pix2pix from Hugging Face.
"""

import io
import base64
import logging
import os
from typing import Optional, Tuple
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

logger = logging.getLogger(__name__)

MODEL_ID = "timbrooks/instruct-pix2pix"


class ImageEditor:
    """
    Lazy-loading wrapper around InstructPix2Pix.
    First call triggers model download (~5 GB).
    """

    def __init__(self):
        self._pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ImageEditor device: {self.device}")

    def _load(self):
        if self._pipe is not None:
            return
        cache_dir = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or "default Hugging Face cache"
        logger.info(
            "Loading InstructPix2Pix pipeline. First run may download several GB of model weights to %s.",
            cache_dir,
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipe.scheduler.config
        )
        self._pipe = self._pipe.to(self.device)
        if self.device == "cuda":
            self._pipe.enable_attention_slicing()
        logger.info("Pipeline ready ✅")

    def is_ready(self) -> bool:
        return self._pipe is not None

    def status(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "device": self.device,
            "loaded": self.is_ready(),
        }

    # ─────────────────────────────────────────
    # Core edit method
    # ─────────────────────────────────────────
    def edit(
        self,
        image: Image.Image,
        instruction: str,
        image_guidance_scale: float = 1.5,
        text_guidance_scale: float  = 7.5,
        num_inference_steps: int    = 30,
        num_images: int             = 1,
        seed: Optional[int]         = None,
    ) -> Image.Image:
        """
        Apply an instruction to an image using InstructPix2Pix.

        Args:
            image:                Input PIL image
            instruction:          Editing instruction string
            image_guidance_scale: How much to preserve original image structure (1–2)
            text_guidance_scale:  How strictly to follow text instruction (5–10)
            num_inference_steps:  Diffusion steps (20–50 recommended)
            num_images:           Number of candidates to generate
            seed:                 Optional random seed for reproducibility
        Returns:
            Best edited PIL image
        """
        self._load()
        image = _preprocess(image)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # ── Reset scheduler state to prevent index overflow ──
        self._pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.inference_mode():
            results = self._pipe(
                prompt=instruction,
                image=image,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=text_guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
            ).images

        return results[0]

    def unload(self):
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _preprocess(image: Image.Image, size: int = 512) -> Image.Image:
    """Resize + centre-crop to square expected by InstructPix2Pix."""
    image = image.convert("RGB")
    w, h  = image.size
    short = min(w, h)
    left  = (w - short) // 2
    top   = (h - short) // 2
    image = image.crop((left, top, left + short, top + short))
    return image.resize((size, size), Image.LANCZOS)


def pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")
