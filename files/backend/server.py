"""
FastAPI backend for RAG-guided Image Editing
"""

import sys
import os
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
    image_guidance_scale: float = 1.5
    text_guidance_scale: float  = 7.5
    num_steps: int              = 30
    seed: Optional[int]         = None
    use_rag: bool               = True  # Toggle for ablation
    include_audio: bool         = False


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
    Full RAG-guided editing pipeline:
    1. Decompose prompt
    2. Retrieve references per sub-task (RAG)
    3. Enrich prompt with retrieved context + state memory
    4. Run InstructPix2Pix
    5. Update state memory
    """
    try:
        # 1. Decode input image
        input_image = base64_to_pil(req.image_b64)

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
        current_image = input_image
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

        edited = current_image
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
        )

    except Exception as e:
        logger.exception("Edit failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_memory(_: ResetRequest = None):
    """Reset the edit state memory for a new session."""
    _memory.reset()
    return {"status": "memory reset"}


@app.get("/state")
def get_state():
    return _memory.to_dict()


# ─── Dev entry point ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
