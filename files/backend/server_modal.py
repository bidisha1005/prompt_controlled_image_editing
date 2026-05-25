"""
Modal-backed FastAPI server shim.

This file reuses the existing `backend.server` application and swaps the
editor implementation to use the Modal-based FLUX Kontext adapter.
"""

from __future__ import annotations

import backend.server as base_server
from backend.editor_modal_flux_kontext import (
    ImageEditor as ModalImageEditor,
    base64_to_pil as modal_base64_to_pil,
    pil_to_base64 as modal_pil_to_base64,
)

# Swap the editor implementation while preserving the exact same API surface.
base_server.ImageEditor = ModalImageEditor
base_server.pil_to_base64 = modal_pil_to_base64
base_server.base64_to_pil = modal_base64_to_pil
base_server._editor = None

app = base_server.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server_modal:app", host="0.0.0.0", port=8000, reload=False)
