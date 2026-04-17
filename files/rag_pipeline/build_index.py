"""
RAG Index Builder for MagicBrush Dataset
Builds a FAISS vector index from MagicBrush editing examples.
Run this once before starting the server.
"""

import os
# Fix OpenMP duplicate initialization on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

    def _extract_text_features(self, outputs) -> torch.Tensor:
        """
        Handle CLIP output shape differences across transformers versions.
        Prefer the projected CLIP embedding, then fall back to pooled output.
        """
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if isinstance(outputs, tuple) and outputs:
            return outputs[0]
        raise TypeError(
            f"Unsupported CLIP text output type: {type(outputs).__name__}"
        )

    @torch.no_grad()
    def embed_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        # Only pass text keys into the text tower. Calling the full CLIP model
        # forward path can require `pixel_values` on some transformers versions.
        text_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }

        outputs = self.model.get_text_features(**text_inputs)
        feats = self._extract_text_features(outputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().numpy()


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
        Enhanced hybrid retrieval with query expansion and polarity-aware re-ranking.
        
        Args:
            query: Search string
            k: Top-k results
            hybrid_weight: Weight for semantic [0-1], keyword gets (1-hybrid_weight)
        
        Returns:
            Top-k results with fused scores, polarity-matched
        """
        from rag_pipeline.prompt_decomposer import (
            DecomposedPrompt,
            EditTask,
            QueryExpander,
            RAGRetrieverScorer,
        )
        
        # ── Step 1: Query expansion with attributes ──
        expanded_queries = QueryExpander.expand(query, num_expansions=2)
        logger.debug(f"Query expanded: {expanded_queries}")
        
        # ── Step 2: Aggregate results from all expansions ──
        all_results = {}  # idx -> result dict
        
        for expanded_query in expanded_queries:
            # Semantic retrieval (FAISS)
            emb = self.embedder.embed_text([expanded_query]).astype("float32")
            scores_semantic, ids = self.index.search(emb, min(k * 5, len(self.metadata)))  # 5x for aggressive re-ranking
            
            for score, idx in zip(scores_semantic[0], ids[0]):
                if idx >= 0:
                    if idx not in all_results:
                        all_results[idx] = {
                            "semantic_scores": [],
                            "keyword_scores": [],
                            "metadata": self.metadata[int(idx)]
                        }
                    all_results[idx]["semantic_scores"].append(float(score))
        
        # ── Step 3: BM25 keyword retrieval bonus ──
        if self.bm25:
            for expanded_query in expanded_queries:
                query_tokens = expanded_query.lower().split()
                bm25_scores = self.bm25.get_scores(query_tokens)
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                
                for idx, score in enumerate(bm25_scores):
                    if score > 0:
                        if idx not in all_results:
                            all_results[idx] = {
                                "semantic_scores": [],
                                "keyword_scores": [],
                                "metadata": self.metadata[int(idx)]
                            }
                        all_results[idx]["keyword_scores"].append(score / max_bm25)
        
        # ── Step 4: Combine and normalize scores ──
        candidates = []
        for idx, result_data in all_results.items():
            # Average scores across expansions
            avg_semantic = sum(result_data["semantic_scores"]) / len(result_data["semantic_scores"]) if result_data["semantic_scores"] else 0.0
            avg_keyword = sum(result_data["keyword_scores"]) / len(result_data["keyword_scores"]) if result_data["keyword_scores"] else 0.0
            
            fused_score = hybrid_weight * avg_semantic + (1 - hybrid_weight) * avg_keyword
            
            candidate = dict(result_data["metadata"])
            candidate["similarity"] = float(fused_score)
            candidate["semantic_score"] = float(avg_semantic)
            candidate["keyword_score"] = float(avg_keyword)
            candidates.append(candidate)
        
        # ── Step 5: Polarity-aware re-ranking ──
        # Create a decomposed prompt from the original query for attribute extraction
        decomposed = DecomposedPrompt(
            original=query,
            tasks=[EditTask(raw=query, edit_type="generic")],
        )
        
        scorer = RAGRetrieverScorer()
        rescored = scorer.score_results(decomposed, candidates, top_k=k * 2)
        
        # Return top-k with final scores
        final_results = rescored[:k]
        
        logger.info(f"Retrieved {len(final_results)} results for '{query}' (polarity-aware)")
        for r in final_results[:3]:
            logger.debug(f"  - {r.get('instruction', '')[:60]}... (score: {r.get('relevance_score', 0):.3f})")
        
        return final_results

    def retrieve_for_decomposed_prompt(
        self, sub_prompts: List[str], k_each: int = 2
    ) -> Dict[str, List[Dict]]:
        """
        Decomposed retrieval with query expansion per sub-task.
        Retrieves references per semantic sub-prompt independently using 
        hybrid search + polarity-aware re-ranking.
        """
        results = {}
        for sp in sub_prompts:
            results[sp] = self.retrieve(sp, k=k_each, hybrid_weight=0.6)
        return results


if __name__ == "__main__":
    build_rag_index()
