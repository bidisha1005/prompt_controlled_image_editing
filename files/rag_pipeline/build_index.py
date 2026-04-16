"""
RAG Index Builder for MagicBrush Dataset
Builds a FAISS vector index from MagicBrush editing examples.
Run this once before starting the server.
"""

import os
import gc
import json
import pickle
import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

# ── Prevent crashes on Mac CPU ───────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

from datasets import load_dataset, Dataset
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
INDEX_DIR = Path("rag_pipeline/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "magicbrush.faiss"
METADATA_PATH    = INDEX_DIR / "metadata.pkl"
CLIP_MODEL_NAME  = "openai/clip-vit-base-patch32"
MAX_SAMPLES      = 8000


# ─────────────────────────────────────────────
# CLIP Embedder
# ─────────────────────────────────────────────
class CLIPEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        self.device = "cpu"
        torch.set_num_threads(1)  # Prevents segfault on Mac

        logger.info("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        logger.info("Loading CLIP processor...")
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Warm-up: force full load before any loop
        logger.info("Warming up CLIP...")
        with torch.no_grad():
            _ = self.embed_text(["warmup"])
        gc.collect()
        logger.info("CLIP ready!")

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        feats = self.model.get_text_features(**inputs)

        if isinstance(feats, torch.Tensor):
            emb = feats
        elif hasattr(feats, "text_embeds") and feats.text_embeds is not None:
            emb = feats.text_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            emb = feats.pooler_output
        elif hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            emb = feats.last_hidden_state[:, 0, :]
        else:
            raise RuntimeError(
                f"Unexpected CLIP text feature output type: {type(feats).__name__}"
            )

        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return emb.detach().cpu().numpy()


# ─────────────────────────────────────────────
# Build Index
# ─────────────────────────────────────────────
def build_rag_index():
    logger.info("Loading MagicBrush dataset via streaming (no full download)...")

    # Streaming mode — downloads one sample at a time, no 51-file download
    raw = load_dataset("osunlp/MagicBrush", split="train", streaming=True)

    instructions_list = []
    for i, sample in enumerate(raw):
        if i >= MAX_SAMPLES:
            break
        instructions_list.append({
            "instruction": sample["instruction"],
            "idx": i,
        })
        if i % 50 == 0:
            logger.info(f"  Collected {i}/{MAX_SAMPLES} samples...")

    logger.info(f"Collected {len(instructions_list)} instructions. Building dataset...")
    dataset = Dataset.from_list(instructions_list)

    # Load CLIP fully before embedding loop
    embedder = CLIPEmbedder()

    dim   = 512  # CLIP ViT-B/32 output dimension
    index = faiss.IndexFlatIP(dim)
    metadata: List[Dict] = []

    batch_size = 2  # Ultra-small batches for Mac RAM safety
    n = len(dataset)
    logger.info(f"Embedding {n} samples in batches of {batch_size}...")

    for i in range(0, n, batch_size):
        batch        = dataset[i : i + batch_size]
        instructions = batch["instruction"]

        try:
            text_embs = embedder.embed_text(instructions).astype("float32")
            index.add(text_embs)

            for j, instr in enumerate(instructions):
                metadata.append({
                    "instruction":  instr,
                    "source_image": None,
                    "target_image": None,
                    "idx":          i + j,
                })

            del text_embs
            gc.collect()

            if i % 40 == 0:
                logger.info(f"  Embedded {min(i + batch_size, n)}/{n}...")

        except Exception as e:
            logger.error(f"Error at batch {i}: {e}")
            raise

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Index built: {index.ntotal} vectors -> {FAISS_INDEX_PATH}")
    return index, metadata


# ─────────────────────────────────────────────
# Retriever with Hybrid Search
# ─────────────────────────────────────────────
class RAGRetriever:
    """
    Hybrid retriever combining semantic (FAISS) + keyword (BM25) search.
    Improves precision by fusing multiple signal types.
    """

    def __init__(self):
        if not FAISS_INDEX_PATH.exists():
            logger.warning("FAISS index not found — building now...")
            build_rag_index()

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        self.embedder = CLIPEmbedder()
        
        # Build BM25 index for keyword retrieval
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [
                entry["instruction"].lower().split() 
                for entry in self.metadata
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built for hybrid search")
        except ImportError:
            logger.warning("rank-bm25 not installed — semantic-only retrieval")
            self.bm25 = None
        
        logger.info(f"RAG Retriever ready — {self.index.ntotal} entries.")

    def retrieve(self, query: str, k: int = 3, hybrid_weight: float = 0.6) -> List[Dict]:
        """
        Hybrid retrieval: blend semantic (FAISS) + keyword (BM25) scores.
        
        Args:
            query: Search string
            k: Top-k results
            hybrid_weight: Weight for semantic [0-1], keyword gets (1-hybrid_weight)
        
        Returns:
            Top-k results with fused scores
        """
        # ── Semantic retrieval (FAISS) ──
        emb = self.embedder.embed_text([query]).astype("float32")
        scores_semantic, ids = self.index.search(emb, min(k * 3, len(self.metadata)))  # Retrieve 3x for fusion
        
        semantic_results = {}
        for score, idx in zip(scores_semantic[0], ids[0]):
            if idx >= 0:
                semantic_results[idx] = float(score)
        
        # ── Keyword retrieval (BM25) ──
        keyword_results = {}
        if self.bm25:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            # Normalize BM25 scores to [0, 1]
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            for idx, score in enumerate(bm25_scores):
                if score > 0:
                    keyword_results[idx] = score / max_bm25
        
        # ── Hybrid fusion: reciprocal rank fusion ──
        fused_scores = {}
        for idx in set(list(semantic_results.keys()) + list(keyword_results.keys())):
            sem_score = semantic_results.get(idx, 0.0)
            kw_score = keyword_results.get(idx, 0.0)
            # Blend scores
            fused_score = hybrid_weight * sem_score + (1 - hybrid_weight) * kw_score
            fused_scores[idx] = fused_score
        
        # ── Sort by fused score and return top-k ──
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in sorted_results:
            entry = dict(self.metadata[int(idx)])
            entry["similarity"] = float(score)
            entry["semantic_score"] = semantic_results.get(int(idx), 0.0)
            entry["keyword_score"] = keyword_results.get(int(idx), 0.0)
            results.append(entry)
        
        return results

    def retrieve_for_decomposed_prompt(
        self, sub_prompts: List[str], k_each: int = 2
    ) -> Dict[str, List[Dict]]:
        """
        Novelty 1: Decomposed retrieval.
        Retrieves references per semantic sub-prompt independently using hybrid search.
        """
        results = {}
        for sp in sub_prompts:
            results[sp] = self.retrieve(sp, k=k_each, hybrid_weight=0.6)
        return results


if __name__ == "__main__":
    build_rag_index()
