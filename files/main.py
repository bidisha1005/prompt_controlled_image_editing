"""
RAG Image Editor — main.py
Run this to start the full stack.
"""

import os
# Allow one OpenMP runtime when multiple libraries are loaded in the same process.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import subprocess, sys, webbrowser, time
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
    print("  Prompt-Decomposed RAG + InstructPix2Pix")
    print("=" * 60)

    check_index()

    print("\n🚀 Starting API server on http://localhost:8000")
    print("🖥  Open frontend/index.html in your browser\n")
    print("ℹ️  The first edit request will download the InstructPix2Pix model once.")
    print("   This can take a while and several GB, especially on CPU.\n")

    # Open the UI automatically
    time.sleep(1)
    ui_path = Path("frontend/index.html").resolve()
    webbrowser.open(f"file://{ui_path}")

    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
