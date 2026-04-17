"""
Replicate-backed FLUX.1 Kontext image editing engine.

This file is intentionally standalone so the existing pipeline can stay exactly
the same. It exposes the same public API shape as `backend/editor.py`:

- `ImageEditor`
- `pil_to_base64`
- `base64_to_pil`

Swap the import in your server only when you want to use the remote model:

    from backend.editor_replicate_flux_kontext import (
        ImageEditor,
        pil_to_base64,
        base64_to_pil,
    )

Environment variables:
- `REPLICATE_API_TOKEN`                Required
- `REPLICATE_FLUX_KONTEXT_MODEL`       Optional, default:
                                       `black-forest-labs/flux-kontext-pro`
- `REPLICATE_PREDICTION_TIMEOUT`       Optional, default: 180
- `REPLICATE_POLL_INTERVAL`            Optional, default: 1.5

Notes:
- This adapter preserves the local-editor interface but executes remotely.
- It sends the input image as a compact data URI so you do not need an
  intermediate upload service.
- Per Replicate's official model/API docs, FLUX Kontext accepts `prompt`,
  `input_image`, `aspect_ratio`, `seed`, `output_format`, and related fields.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)

REPLICATE_API_BASE = "https://api.replicate.com/v1"
DEFAULT_MODEL = "black-forest-labs/flux-kontext-pro"


class ImageEditor:
    """
    Drop-in replacement for the local InstructPix2Pix editor.

    From the rest of the application's point of view, this behaves like a lazy
    model wrapper. Internally it delegates the edit call to Replicate's hosted
    FLUX.1 Kontext model.
    """

    def __init__(self):
        self.api_token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
        self.model = os.environ.get(
            "REPLICATE_FLUX_KONTEXT_MODEL",
            DEFAULT_MODEL,
        ).strip() or DEFAULT_MODEL
        self.timeout_s = int(os.environ.get("REPLICATE_PREDICTION_TIMEOUT", "180"))
        self.poll_interval_s = float(os.environ.get("REPLICATE_POLL_INTERVAL", "1.5"))
        self._ready = False
        logger.info("Replicate Flux Kontext editor configured for model: %s", self.model)

    def is_ready(self) -> bool:
        """Check whether the adapter has validated its configuration."""
        return self._ready

    def _load(self):
        """
        Validate configuration on first use.

        This mirrors the local editor's lazy-load behavior without downloading
        model weights.
        """
        if self._ready:
            return
        if not self.api_token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN is not set. "
                "Add it to your environment or .env before using the Replicate editor."
            )
        self._ready = True
        logger.info("Replicate Flux Kontext adapter ready")

    def edit(
        self,
        image: Image.Image,
        instruction: str,
        image_guidance_scale: float = 1.8,
        text_guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Apply an editing instruction using Replicate-hosted FLUX.1 Kontext.

        The signature matches the local editor so the existing pipeline can
        treat this as a drop-in replacement. Parameters not used directly by
        Kontext are accepted for compatibility.
        """
        del image_guidance_scale
        del text_guidance_scale
        del num_inference_steps
        del num_images

        self._load()
        processed = _preprocess(image)
        image_data_uri = pil_to_data_uri(processed, fmt="JPEG", quality=88)

        payload = {
            "input": {
                "prompt": instruction,
                "input_image": image_data_uri,
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "prompt_upsampling": False,
                "safety_tolerance": 2,
            }
        }
        if seed is not None:
            payload["input"]["seed"] = seed

        prediction = self._create_prediction(payload)
        final_prediction = self._wait_for_prediction(prediction)
        output_url = self._extract_output_url(final_prediction)
        if not output_url:
            raise RuntimeError(
                f"Replicate prediction succeeded but returned no output URL: {final_prediction}"
            )
        return self._download_image(output_url)

    def unload(self):
        """
        Reset local readiness state.

        There is no local model to unload, but keeping the method preserves
        compatibility with the current pipeline.
        """
        self._ready = False

    def _headers(self, include_content_type: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _prediction_url(self) -> str:
        owner, name = self._split_model_name(self.model)
        return f"{REPLICATE_API_BASE}/models/{owner}/{name}/predictions"

    def _create_prediction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Submitting Replicate prediction to %s", self.model)
        response = requests.post(
            self._prediction_url(),
            headers={**self._headers(), "Prefer": "wait=60"},
            json=payload,
            timeout=70,
        )
        self._raise_for_status(response, "create prediction")
        prediction = response.json()
        logger.info(
            "Replicate prediction created: id=%s status=%s",
            prediction.get("id"),
            prediction.get("status"),
        )
        return prediction

    def _wait_for_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        status = prediction.get("status")
        if status == "succeeded":
            return prediction
        if status in {"failed", "canceled"}:
            raise RuntimeError(
                f"Replicate prediction ended early with status={status}: {prediction.get('error')}"
            )

        get_url = prediction.get("urls", {}).get("get")
        if not get_url:
            raise RuntimeError(f"Prediction response missing poll URL: {prediction}")

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            response = requests.get(
                get_url,
                headers=self._headers(include_content_type=False),
                timeout=30,
            )
            self._raise_for_status(response, "poll prediction")
            prediction = response.json()
            status = prediction.get("status")

            if status == "succeeded":
                logger.info(
                    "Replicate prediction completed: id=%s",
                    prediction.get("id"),
                )
                return prediction
            if status in {"failed", "canceled"}:
                raise RuntimeError(
                    f"Replicate prediction {status}: {prediction.get('error')}"
                )

            time.sleep(self.poll_interval_s)

        raise TimeoutError(
            f"Timed out after {self.timeout_s}s waiting for Replicate prediction "
            f"{prediction.get('id')}"
        )

    def _extract_output_url(self, prediction: Dict[str, Any]) -> Optional[str]:
        output = prediction.get("output")
        if isinstance(output, str):
            return output
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return first.get("url")
        if isinstance(output, dict):
            return output.get("url")
        return None

    def _download_image(self, url: str) -> Image.Image:
        response = requests.get(
            url,
            headers=self._headers(include_content_type=False),
            timeout=60,
        )
        self._raise_for_status(response, "download generated image")
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    def _raise_for_status(self, response: requests.Response, action: str) -> None:
        if response.ok:
            return
        detail = ""
        try:
            detail = str(response.json())
        except Exception:
            detail = response.text[:500]
        raise RuntimeError(
            f"Replicate API failed while trying to {action}: "
            f"{response.status_code} {detail}"
        )

    @staticmethod
    def _split_model_name(model: str) -> tuple[str, str]:
        parts = model.split("/", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "REPLICATE_FLUX_KONTEXT_MODEL must look like "
                "'owner/model-name', for example 'black-forest-labs/flux-kontext-pro'"
            )
        return parts[0], parts[1]


def _preprocess(image: Image.Image, size: int = 512) -> Image.Image:
    """Resize and center-crop to the same square format used by the local editor."""
    image = image.convert("RGB")
    w, h = image.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    image = image.crop((left, top, left + short, top + short))
    return image.resize((size, size), Image.LANCZOS)


def pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_data_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 88) -> str:
    """
    Encode an image as a data URI for Replicate input files.

    Replicate's docs support data URIs for file inputs. Keeping the image at
    512x512 helps stay compact and avoids introducing another upload step.
    """
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=quality, optimize=True)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
