"""
Evaluation Metrics
──────────────────
• CLIP similarity  – prompt-image alignment
• LPIPS            – perceptual similarity to reference
• FID              – requires torchmetrics[image]
"""

import numpy as np
from typing import List, Optional
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class CLIPEvaluator:
    def __init__(self):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()

    @torch.no_grad()
    def text_image_similarity(
        self, image: Image.Image, text: str
    ) -> float:
        """
        Cosine similarity between image embedding and text embedding.
        Range [-1, 1]; higher = better prompt alignment.
        """
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        img_emb  = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb  = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((img_emb * txt_emb).sum(dim=-1).cpu().item())

    @torch.no_grad()
    def image_image_similarity(
        self, img1: Image.Image, img2: Image.Image
    ) -> float:
        """Cosine similarity between two image embeddings."""
        inputs = self.processor(
            images=[img1, img2], return_tensors="pt"
        ).to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return float((feats[0] * feats[1]).sum().cpu().item())


def evaluate_edit(
    original: Image.Image,
    edited: Image.Image,
    instruction: str,
    evaluator: Optional[CLIPEvaluator] = None,
) -> dict:
    """
    Run all available metrics for a single edit.
    Returns a dict of metric_name → value.
    """
    if evaluator is None:
        evaluator = CLIPEvaluator()

    results = {}

    # CLIP text-image alignment (main metric)
    results["clip_text_image"] = evaluator.text_image_similarity(edited, instruction)

    # Image similarity to original (structural preservation)
    results["clip_img_preservation"] = evaluator.image_image_similarity(original, edited)

    # Simple pixel-level difference
    orig_arr  = np.array(original.resize((224, 224))).astype(float)
    edit_arr  = np.array(edited.resize((224, 224))).astype(float)
    results["pixel_mse"] = float(np.mean((orig_arr - edit_arr) ** 2))
    results["pixel_diff_pct"] = float(
        np.mean(np.abs(orig_arr - edit_arr) > 20) * 100
    )

    logger.info(f"Metrics: {results}")
    return results


def run_ablation(
    original: Image.Image,
    edited_no_rag: Image.Image,
    edited_with_rag: Image.Image,
    instruction: str,
) -> dict:
    """
    Compare RAG vs no-RAG for a single sample.
    Returns a comparison dict.
    """
    ev = CLIPEvaluator()
    base    = evaluate_edit(original, edited_no_rag,   instruction, ev)
    proposed = evaluate_edit(original, edited_with_rag, instruction, ev)

    delta = {
        k: round(proposed[k] - base[k], 4) for k in base
    }
    return {
        "baseline_no_rag":  base,
        "proposed_with_rag": proposed,
        "delta":            delta,
    }
