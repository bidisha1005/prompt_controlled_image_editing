"""
Replicate-backed server entrypoint.

This reuses the existing FastAPI app and full RAG pipeline from `backend.server`
but swaps the local diffusion editor for the Replicate FLUX.1 Kontext adapter.
"""

from __future__ import annotations

import backend.server as base_server
from backend.editor_replicate_flux_kontext import (
    ImageEditor as ReplicateImageEditor,
    base64_to_pil as replicate_base64_to_pil,
    pil_to_base64 as replicate_pil_to_base64,
)

# Swap the editor implementation while reusing the exact same pipeline and app.
base_server.ImageEditor = ReplicateImageEditor
base_server.pil_to_base64 = replicate_pil_to_base64
base_server.base64_to_pil = replicate_base64_to_pil
base_server._editor = None

app = base_server.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server_replicate:app", host="0.0.0.0", port=8000, reload=False)
