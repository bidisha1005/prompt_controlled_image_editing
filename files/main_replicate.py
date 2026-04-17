"""
RAG Image Editor — main_replicate.py
Run this to start the full stack using Replicate FLUX.1 Kontext for editing.
"""

import os

# Keep runtime environment aligned with the original launcher.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import time
import webbrowser
from pathlib import Path


def check_index():
    idx = Path("rag_pipeline/index/magicbrush.faiss")
    if not idx.exists():
        print("📦 FAISS index not found — building from MagicBrush (first run only)…")
        from rag_pipeline.build_index import build_rag_index

        build_rag_index()
    else:
        print(f"✅ FAISS index found: {idx}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  RAG Image Editor")
    print("  Prompt-Decomposed RAG + Replicate FLUX.1 Kontext")
    print("=" * 60)

    check_index()

    print("\n🚀 Starting API server on http://localhost:8000")
    print("🖥  Open frontend/index.html in your browser\n")
    print("ℹ️  Edit requests will use the Replicate API instead of a local diffusion model.")
    print("ℹ️  Make sure REPLICATE_API_TOKEN is set in your environment or .env.\n")

    time.sleep(1)
    ui_path = Path("frontend/index.html").resolve()
    webbrowser.open(f"file://{ui_path}")

    uvicorn.run(
        "backend.server_replicate:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
