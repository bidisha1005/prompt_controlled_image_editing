"""
Modal-backed FLUX Kontext image editing engine.

This adapter preserves the public editor interface used by the existing
pipeline and can be swapped in exactly the same way as
`backend/editor_replicate_flux_kontext.py`.

From the rest of the app, this exposes:

- `ImageEditor`
- `pil_to_base64`
- `base64_to_pil`

The Modal endpoint is expected to accept JSON with `prompt` and `image`
(base64 string), and to return JSON with `image` containing a base64-encoded
edited image.
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

# Load environment variables from .env file first if available.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except ImportError:
    pass

logger = logging.getLogger(__name__)

DEFAULT_MODAL_ENDPOINT = (
    "https://bidisha-paul10--flux-kontext-fastapi-fastapi-app.modal.run"
)
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_REQUEST_RETRIES = 0
DEFAULT_BACKOFF_FACTOR = 1.0
DEFAULT_MODAL_PATH = "/edit_image"


class ImageEditor:
    """
    Drop-in replacement for the local InstructPix2Pix editor using a Modal-hosted
    FLUX Kontext service.
    """

    def __init__(self):
        self.endpoint = os.environ.get("MODAL_ENDPOINT", DEFAULT_MODAL_ENDPOINT).strip()
        self.timeout_s = int(os.environ.get("MODAL_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT)))
        self.max_retries = int(os.environ.get("MODAL_REQUEST_RETRIES", str(DEFAULT_REQUEST_RETRIES)))
        self.backoff_factor = float(os.environ.get("MODAL_REQUEST_BACKOFF", str(DEFAULT_BACKOFF_FACTOR)))
        self.bearer_token = os.environ.get("MODAL_BEARER_TOKEN", "").strip() or None
        self._ready = False
        self._session = requests.Session()
        self._session.headers.update(self._headers())
        logger.info("Modal Flux Kontext editor configured for endpoint: %s", self.endpoint)

    def is_ready(self) -> bool:
        return self._ready

    def _load(self) -> None:
        if self._ready:
            return
        if not self.endpoint:
            raise RuntimeError(
                "MODAL_ENDPOINT is not set. Set it in the environment or .env file "
                "to use the Modal Flux Kontext editor."
            )
        if self.timeout_s <= 0:
            raise RuntimeError("MODAL_REQUEST_TIMEOUT must be a positive integer")
        self._ready = True
        logger.info("Modal Flux Kontext adapter ready")

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
        self._load()

        image = _preprocess(image)
        image_b64 = pil_to_base64(image, fmt="JPEG")

        payload: Dict[str, Any] = {
            "prompt": instruction,
            "guidance_scale": 3.5,
            "num_inference_steps": num_inference_steps,
        }
        if seed is not None:
            payload["seed"] = seed

        image_bytes = _image_to_bytes(image, fmt="JPEG")
        files = {
            "image": ("input.jpg", image_bytes, "image/jpeg"),
        }

        logger.info(
            "Modal edit request started: prompt=%s, image_size=%dx%d, timeout=%s",
            instruction,
            image.width,
            image.height,
            self.timeout_s,
        )

        start = time.perf_counter()
        response = self._send_request(payload, files=files)
        elapsed = time.perf_counter() - start

        content_type = response.headers.get("Content-Type", "")
        logger.info(
            "Modal edit request completed: status=%s, duration=%.2fs, content-type=%s",
            response.status_code,
            elapsed,
            content_type,
        )

        if "image" in content_type.lower() or response.headers.get("Content-Disposition"):
            return Image.open(io.BytesIO(response.content)).convert("RGB")

        data = self._parse_response(response)
        edited_b64 = data.get("image")
        if not edited_b64 or not isinstance(edited_b64, str):
            raise RuntimeError(
                "Modal endpoint returned unexpected payload: missing 'image' field"
            )

        return base64_to_pil(edited_b64)

    def unload(self) -> None:
        self._ready = False

    def _send_request(
        self,
        payload: Dict[str, Any],
        files: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                response = self._session.post(
                    self._resolve_endpoint(),
                    data=payload,
                    files=files,
                    timeout=(10, self.timeout_s),
                )
                self._raise_for_status(response)
                return response
            except requests.RequestException as exc:
                last_error = exc
                attempt += 1
                logger.warning(
                    "Modal request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt > self.max_retries:
                    break
                time.sleep(self.backoff_factor * attempt)

        raise RuntimeError(
            "Failed to complete Modal request after %d attempts: %s" % (
                self.max_retries + 1,
                last_error,
            )
        )

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _resolve_endpoint(self) -> str:
        if self.endpoint.endswith("/"):
            return f"{self.endpoint.rstrip('/')}{DEFAULT_MODAL_PATH}"
        if self.endpoint.endswith(DEFAULT_MODAL_PATH):
            return self.endpoint
        if "/" not in self.endpoint.split("//", 1)[-1]:
            return f"{self.endpoint.rstrip('/')}{DEFAULT_MODAL_PATH}"
        return self.endpoint

    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Modal endpoint returned invalid JSON: %s" % exc
            ) from exc

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        detail = None
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(
            "Modal endpoint request failed: %s %s" % (
                response.status_code,
                detail,
            )
        )


def pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    cleaned = _strip_data_uri(b64)
    data = base64.b64decode(cleaned)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=90)
    return buf.getvalue()


def _strip_data_uri(payload: str) -> str:
    if payload.startswith("data:") and "," in payload:
        return payload.split(",", 1)[1]
    return payload


def _preprocess(image: Image.Image, size: int = 512) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    return image.crop((left, top, left + short, top + short)).resize((size, size), Image.LANCZOS)
