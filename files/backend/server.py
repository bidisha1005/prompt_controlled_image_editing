"""
FastAPI backend for RAG-guided Image Editing
"""

import sys
import os

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Fix OpenMP duplicate initialization on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.editor import ImageEditor, pil_to_base64, base64_to_pil
from backend.music_selector import ASSETS_DIR, get_library_count, select_track
from backend.prompt_reconstructor import GroqPromptReconstructor
from rag_pipeline.prompt_decomposer import (
    EditStateMemory,
    RAGRetrieverScorer,
    decompose_prompt,
)
from rag_pipeline.build_index import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Image Editor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# ─── Singletons (lazy-loaded on first request) ───────────────
_editor: Optional[ImageEditor] = None
_retriever: Optional[RAGRetriever] = None
_reconstructor: Optional[GroqPromptReconstructor] = None
_memory: EditStateMemory = EditStateMemory()
_scorer = RAGRetrieverScorer()

# ─── Edit History / Snapshots ───────────────────────────────
# Store snapshots: {s0: {image, state}, s1: {image, state}, ...}
_snapshots: Dict[str, Dict[str, Any]] = {}
_current_base_snapshot: str = "s0"  # Current snapshot to edit from


def get_editor() -> ImageEditor:
    global _editor
    if _editor is None:
        _editor = ImageEditor()
    return _editor


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def get_reconstructor() -> GroqPromptReconstructor:
    global _reconstructor
    if _reconstructor is None:
        _reconstructor = GroqPromptReconstructor()
    return _reconstructor


# ─── Request / Response models ───────────────────────────────
class EditRequest(BaseModel):
    image_b64: str          # Base64-encoded input image
    instruction: str        # User's editing instruction
    image_guidance_scale: float = 1.8
    text_guidance_scale: float  = 7.0
    num_steps: int              = 30
    seed: Optional[int]         = None
    use_rag: bool               = True  # Toggle for ablation
    include_audio: bool         = False
    branch_from: Optional[str]  = None  # Snapshot to branch from (e.g., "s0", "s1")


class SelectedTrack(BaseModel):
    id: str
    title: str
    file: str
    file_url: str
    duration_sec: Optional[int] = None
    mood: List[str] = Field(default_factory=list)
    genre: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    reason: str


class EditResponse(BaseModel):
    edited_image_b64: str
    decomposed_sub_tasks: List[str]
    preserve_hints: List[str]
    retrieved_references: Dict[str, Any]
    reconstructed_prompts: List[str]
    enriched_prompt: str
    prompt_reconstruction: Dict[str, Any]
    edit_state: Dict[str, Any]
    selected_track: Optional[SelectedTrack] = None
    snapshot_id: str = "s0"  # Current snapshot label (s0, s1, s2, ...)
    available_snapshots: List[str] = Field(default_factory=list)  # List of available snapshot IDs


class DecomposeRequest(BaseModel):
    instruction: str


class AudioSelectRequest(BaseModel):
    instruction: str


class ResetRequest(BaseModel):
    pass


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "editor_loaded": _editor is not None and _editor.is_ready(),
        "retriever_loaded": _retriever is not None,
        "prompt_reconstructor": get_reconstructor().status(),
        "music_library_count": get_library_count(),
    }


@app.post("/decompose")
def decompose(req: DecomposeRequest):
    """Preview prompt decomposition without running the model."""
    result = decompose_prompt(req.instruction)
    return {
        "sub_tasks":      result.sub_tasks,
        "preserve_hints": result.preserve_hints,
        "edit_types":     result.edit_types,
    }

@app.post("/select-audio", response_model=Optional[SelectedTrack])
def select_audio(req: AudioSelectRequest):
    """Pick a background track from the local curated library."""
    try:
        if not req.instruction.strip():
            return None
        return select_track(req.instruction)
    except Exception as e:
        logger.exception("Audio selection failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit", response_model=EditResponse)
def edit_image(req: EditRequest):
    """
    Full RAG-guided editing pipeline with snapshot history:
    1. Load from base snapshot (s0 or selected branch point)
    2. Decompose prompt
    3. Retrieve references per sub-task (RAG)
    4. Enrich prompt with retrieved context + state memory
    5. Run InstructPix2Pix
    6. Update state memory
    7. Save new snapshot (s1, s2, etc.)
    """
    global _snapshots, _current_base_snapshot, _memory
    
    try:
        # ── Determine starting image & state ──
        # AUTO-BRANCH AFTER RESET: If only s0 exists (just reset), automatically branch from it
        if "s0" in _snapshots and len(_snapshots) == 1 and not req.branch_from:
            req.branch_from = "s0"
            logger.info("Auto-branching from s0 after reset")
        
        if req.branch_from and req.branch_from in _snapshots:
            # Branch from existing snapshot (e.g., s0, s1)
            snap = _snapshots[req.branch_from]
            current_image = snap["image"]
            _memory = snap["memory"]  # Restore memory from snapshot
            _current_base_snapshot = req.branch_from
            logger.info(f"Branching from snapshot: {req.branch_from}")
        else:
            # Use provided image or error
            if not req.image_b64:
                raise HTTPException(status_code=400, detail="No image_b64 provided")
            current_image = base64_to_pil(req.image_b64)
            # Initialize or reset snapshots if new starting image
            if req.branch_from is None or req.branch_from not in _snapshots:
                # IMPORTANT: Preserve s0 if it already exists (after reset)
                preserved_s0 = _snapshots.get("s0") if "s0" in _snapshots else None
                _snapshots.clear()
                _memory = EditStateMemory()
                _current_base_snapshot = "s0"
                # Save s0 (original) - either the preserved one or the new image
                if preserved_s0:
                    # Restore previously saved s0
                    _snapshots["s0"] = preserved_s0
                    logger.info("Restored preserved snapshot s0")
                else:
                    # First time - save current image as s0
                    _snapshots["s0"] = {
                        "image": current_image,
                        "memory": EditStateMemory(),  # Fresh state for original
                        "instruction": "Original",
                        "metadata": {"type": "original"}
                    }
                    logger.info("Initialized snapshot s0 (original)")

        # 1. Decode input image (already done above)
        input_image = current_image

        # 2. Decompose prompt  [Novelty 1]
        decomposed = decompose_prompt(req.instruction)
        logger.info(f"Sub-tasks: {decomposed.sub_tasks}")
        logger.info(f"Preserve:  {decomposed.preserve_hints}")

        # 3. RAG retrieval per sub-task  [Novelty 1]
        references: Dict[str, Any] = {}

        if req.use_rag:
            retriever = get_retriever()
            ref_map = retriever.retrieve_for_decomposed_prompt(
                decomposed.sub_tasks, k_each=2
            )
            for sub_task, refs in ref_map.items():
                reranked = _scorer.score_results(
                    decompose_prompt(sub_task),
                    refs,
                    top_k=2,
                )
                references[sub_task] = [
                    {
                        "instruction": r["instruction"],
                        "similarity": round(r["similarity"], 3),
                        "relevance_score": round(
                            r.get("relevance_score", r["similarity"]), 3
                        ),
                        "edit_type": r.get("edit_type", "generic"),
                    }
                    for r in reranked
                ]

        # 4. Reconstruct a concise prompt per sub-task & run sequentially
        reconstructor = get_reconstructor()
        reconstructed_prompts: List[str] = []
        prompt_reconstruction: Dict[str, Any] = {}

        for i, (sub_task, etype) in enumerate(zip(decomposed.sub_tasks, decomposed.edit_types)):
            task_refs = references.get(sub_task, [])
            reconstruction = reconstructor.reconstruct(
                sub_task=sub_task,
                references=task_refs,
                state=_memory.to_dict(),
                preserve_hints=decomposed.preserve_hints,
                edit_type=etype,
            )
            task_prompt = reconstruction.prompt
            reconstructed_prompts.append(task_prompt)
            prompt_reconstruction[sub_task] = {
                "prompt": task_prompt,
                "used_llm": reconstruction.used_llm,
                "model": reconstruction.model,
                "references_used": reconstruction.references_used,
            }
            logger.info(
                "Step %s/%s reconstructed prompt: %s",
                i + 1,
                len(decomposed.sub_tasks),
                task_prompt,
            )

            # Run inference for this specific sub-task
            editor = get_editor()
            current_image = editor.edit(
                image=current_image,
                instruction=task_prompt,
                image_guidance_scale=req.image_guidance_scale,
                text_guidance_scale=req.text_guidance_scale,
                num_inference_steps=req.num_steps,
                seed=req.seed,
            )

            # Update state memory with this sub-task
            decomposed_step = decompose_prompt(sub_task)
            _memory.update(decomposed_step)

        # ── Save new snapshot ──
        edited = current_image
        next_snapshot_id = f"s{len(_snapshots)}"
        _snapshots[next_snapshot_id] = {
            "image": edited,
            "memory": _memory,  # Save current state
            "instruction": req.instruction,
            "metadata": {
                "type": "edit",
                "branch_from": _current_base_snapshot,
                "sub_tasks": decomposed.sub_tasks,
            }
        }
        _current_base_snapshot = next_snapshot_id
        logger.info(f"Saved snapshot: {next_snapshot_id}")

        enriched_prompt = "; ".join(reconstructed_prompts)
        selected_track = select_track(req.instruction) if req.include_audio else None

        return EditResponse(
            edited_image_b64=pil_to_base64(edited),
            decomposed_sub_tasks=decomposed.sub_tasks,
            preserve_hints=decomposed.preserve_hints,
            retrieved_references=references,
            reconstructed_prompts=reconstructed_prompts,
            enriched_prompt=enriched_prompt,
            prompt_reconstruction=prompt_reconstruction,
            edit_state=_memory.to_dict(),
            selected_track=selected_track,
            snapshot_id=next_snapshot_id,
            available_snapshots=sorted(_snapshots.keys(), key=lambda x: int(x[1:])),
        )

    except Exception as e:
        logger.exception("Edit failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
@app.get("/reset")  # Also accept GET for convenience
def reset_memory(_: Optional[ResetRequest] = None):
    """Reset the edit state memory but preserve original (s0).
    
    Keeps s0 (original), clears s1, s2, s3, etc.
    Resets edit state memory to fresh state.
    """
    global _memory, _snapshots, _current_base_snapshot
    
    # Keep only s0, clear everything else
    if "s0" in _snapshots:
        s0_snapshot = _snapshots["s0"]
        _snapshots.clear()
        _snapshots["s0"] = s0_snapshot
    else:
        _snapshots.clear()
    
    # Reset memory to fresh state
    _memory = EditStateMemory()
    _current_base_snapshot = "s0"
    
    logger.info("Edit state memory reset — s0 preserved, all edits cleared")
    return {
        "status": "memory reset",
        "preserved": "s0 (original)",
        "available_snapshots": list(_snapshots.keys()),
    }


@app.get("/snapshots")
def get_snapshots():
    """Get all available snapshots with metadata (no large images).
    
    Returns: {
        s0: {id: "s0", instruction: "Original", metadata: {...}},
        s1: {id: "s1", instruction: "...", metadata: {...}},
        ...
    }
    """
    result = {}
    for snap_id in sorted(_snapshots.keys(), key=lambda x: int(x[1:])):
        snap = _snapshots[snap_id]
        result[snap_id] = {
            "id": snap_id,
            "instruction": snap.get("instruction", ""),
            "metadata": snap.get("metadata", {}),
        }
    return result


@app.get("/snapshots/{snap_id}/thumbnail")
def get_snapshot_thumbnail(snap_id: str):
    """Get thumbnail (small image) for a snapshot."""
    if snap_id not in _snapshots:
        raise HTTPException(status_code=404, detail=f"Snapshot {snap_id} not found")
    
    img = _snapshots[snap_id]["image"]
    # Resize to thumbnail size (200x150)
    thumbnail = img.copy()
    thumbnail.thumbnail((200, 150))
    return {"image_b64": pil_to_base64(thumbnail)}


@app.get("/state")
def get_state():
    global _current_base_snapshot
    return {
        **_memory.to_dict(),
        "current_snapshot": _current_base_snapshot,
        "available_snapshots": sorted(_snapshots.keys(), key=lambda x: int(x[1:])),
    }


# ─── Dev entry point ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
