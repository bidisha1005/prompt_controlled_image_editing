"""
Prompt Decomposition + Edit State Memory
────────────────────────────────────────
Novelty 1 – Prompt-Decomposed RAG Retrieval
Novelty 2 – Edit State Memory with Constraint Propagation
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Prompt Decomposer
# ─────────────────────────────────────────────────────────────
# Trigger patterns for each supported edit operation
EDIT_PATTERNS = {
    "remove": [
        r"\b(remove|delete|erase|get rid of|eliminate)\b.{0,40}",
    ],
    "color_change": [
        r"\b(change|make|turn|color|paint)\b.{0,30}\b(color|to|into)\b.{0,20}",
        r"\b(red|blue|green|yellow|white|black|purple|orange|pink|gray|grey|brown)\b",
    ],
    "lighting": [
        r"\b(sunset|sunrise|golden hour|night|day|noon|dusk|dawn|dark|bright|dim)\b",
        r"\b(lighting|light|illuminate)\b",
    ],
    "add": [
        r"\b(add|insert|place|put|include)\b.{0,40}",
    ],
    "style": [
        r"\b(make it|style|look like|aesthetic|vintage|modern|cartoon|realistic)\b",
    ],
}

# Keywords that signal preservation constraints
PRESERVE_KEYWORDS = [
    "keep", "preserve", "maintain", "leave", "don't change",
    "do not change", "intact", "same", "unchanged",
]


@dataclass
class DecomposedPrompt:
    original: str
    sub_tasks: List[str]       = field(default_factory=list)
    preserve_hints: List[str]  = field(default_factory=list)
    edit_types: List[str]      = field(default_factory=list)
    objects: List[str]         = field(default_factory=list)  # Extracted objects being edited


def decompose_prompt(prompt: str) -> DecomposedPrompt:
    """
    Split a complex prompt into atomic editing sub-tasks
    and extract preservation constraints.

    Example:
        "Remove the car and make the sky sunset, keep mountains"
        → sub_tasks: ["Remove the car", "make the sky sunset"]
        → preserve_hints: ["mountains"]
    """
    result = DecomposedPrompt(original=prompt)

    # ── 1. Split on conjunctions / punctuation ──
    raw_parts = re.split(
        r"\s*(?:,|;|\band\b|\bthen\b|\balso\b|\bwhile\b)\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    raw_parts = [p.strip() for p in raw_parts if p.strip()]

    # ── 2. Separate preservation hints from edit tasks ──
    for part in raw_parts:
        is_preserve = any(
            kw in part.lower() for kw in PRESERVE_KEYWORDS
        )
        if is_preserve:
            # Extract the subject being preserved
            for kw in PRESERVE_KEYWORDS:
                part = re.sub(rf"\b{kw}\b\s*(?:the\s+)?", "", part,
                              flags=re.IGNORECASE)
            result.preserve_hints.append(part.strip())
        else:
            if part:
                result.sub_tasks.append(part)

    # ── 3. Tag edit type per sub-task ──
    for task in result.sub_tasks:
        found = "generic"
        for etype, patterns in EDIT_PATTERNS.items():
            if any(re.search(p, task, re.IGNORECASE) for p in patterns):
                found = etype
                break
        result.edit_types.append(found)
        
        # ── 4. Extract the object being edited ──
        obj = _extract_edited_object(task)
        result.objects.append(obj)

    if not result.sub_tasks:
        result.sub_tasks = [prompt]
        result.edit_types = ["generic"]
        result.objects = [""]

    logger.debug(f"Decomposed: {result}")
    return result


def _extract_edited_object(task: str) -> str:
    """
    Extract the main object/subject being edited from a task.
    Examples:
        "make the car red" → "car"
        "add a phone beside me" → "phone"
        "remove the tree" → "tree"
        "change the sky to sunset" → "sky"
    """
    # Pattern: verb + [the/a/an] + OBJECT
    patterns = [
        r"\b(?:remove|delete|erase|add|place|put|insert|make|change|color|paint|style|lighten|darken)\s+(?:the\s+)?(\w+)",
        r"(?:the|a|an)\s+(\w+)\s+(?:to|into|and)",  # "the car to red"
        r"\b(?:to|into)\s+(\w+)\b",  # "change to school bus"
    ]
    
    for pattern in patterns:
        m = re.search(pattern, task, re.IGNORECASE)
        if m:
            obj = m.group(1).lower().strip()
            # Filter out common non-nouns
            if obj not in ["it", "is", "the", "a", "an", "be", "make", "look"]:
                return obj
    
    return ""


# ─────────────────────────────────────────────────────────────
# Query Expansion for Better Retrieval
# ─────────────────────────────────────────────────────────────
class QueryExpander:
    """Expands queries with synonyms and related terms to improve retrieval."""
    
    SYNONYM_MAP = {
        # Objects
        "car": ["vehicle", "automobile", "vehicle"],
        "tree": ["plant", "vegetation", "foliage"],
        "sky": ["background", "top"],
        "person": ["human", "people", "figure", "subject"],
        "building": ["structure", "house", "architecture"],
        "water": ["ocean", "sea", "lake", "river"],
        
        # Actions
        "remove": ["delete", "erase", "eliminate", "get rid of"],
        "add": ["insert", "place", "put", "include"],
        "change": ["modify", "alter", "switch", "transform"],
        "make": ["create", "turn into", "convert to"],
        
        # Colors
        "red": ["crimson", "scarlet", "rouge"],
        "blue": ["azure", "cyan", "navy"],
        "green": ["emerald", "lime", "sage"],
        "sunset": ["golden hour", "warm", "orange"],
        "bright": ["light", "vivid", "luminous"],
        "dark": ["dim", "shadowy", "low light"],
    }
    
    @staticmethod
    def expand(query: str, num_expansions: int = 3) -> List[str]:
        """
        Generate expanded query variants with synonyms.
        Returns: [original_query, variant1, variant2, ...]
        """
        expansions = [query]
        query_lower = query.lower()
        
        # Find and substitute synonyms
        for term, synonyms in QueryExpander.SYNONYM_MAP.items():
            if term in query_lower:
                for synonym in synonyms[:num_expansions]:
                    variant = query_lower.replace(term, synonym)
                    if variant not in expansions:
                        expansions.append(variant)
                    if len(expansions) >= num_expansions + 1:
                        break
            if len(expansions) >= num_expansions + 1:
                break
        
        return expansions[:num_expansions + 1]


# ─────────────────────────────────────────────────────────────
# RAG Retrieval Filtering & Re-ranking (IMPROVED)
# ─────────────────────────────────────────────────────────────
class RAGRetrieverScorer:
    """
    Advanced re-ranking with:
    1. Multi-signal fusion (semantic + keyword + type + object matching)
    2. Diversity filtering (penalize similar results)
    3. Length normalization for fair comparison
    """
    
    def score_results(self, 
                     decomposed: DecomposedPrompt,
                     candidates: List[Dict],
                     top_k: int = 5) -> List[Dict]:
        """
        Score and re-rank retrieval candidates with diversity awareness.
        
        Args:
            decomposed: DecomposedPrompt with extracted objects
            candidates: List of retrieved examples with:
                - instruction (str): the example instruction
                - similarity (float): semantic similarity score
                - semantic_score (float, optional): from FAISS
                - keyword_score (float, optional): from BM25
            top_k: Return top-k results after re-ranking + diversity filtering
            
        Returns:
            List of re-ranked candidates with detailed scores
        """
        if not candidates or not decomposed.sub_tasks:
            return candidates[:top_k]
        
        # Infer edit types from candidates if not provided
        for candidate in candidates:
            if "edit_type" not in candidate:
                candidate["edit_type"] = self._infer_edit_type(
                    candidate.get("instruction", "")
                )
        
        # Score each candidate
        scored = []
        for candidate in candidates:
            score = self._compute_relevance_score(decomposed, candidate)
            scored.append({
                **candidate,
                "relevance_score": score
            })
        
        # Sort by relevance score (descending)
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # ── Diversity filtering ──
        # Among top candidates, penalize very similar ones
        diverse_results = self._filter_diversity(scored, top_k)
        
        logger.debug(f"Top-3 re-ranked + diverse: {diverse_results[:3]}")
        return diverse_results[:top_k]
    
    def _compute_relevance_score(self, 
                                decomposed: DecomposedPrompt, 
                                candidate: Dict) -> float:
        """
        Multi-signal relevance score combining:
        - Semantic similarity
        - Keyword matching
        - Edit type alignment
        - Object mention matching
        """
        # ── Base signals ──
        # Prefer hybrid scores if available, else fall back to similarity
        base_score = candidate.get("similarity", 0.0)  # [0, 1]
        semantic_score = candidate.get("semantic_score", base_score)
        keyword_score = candidate.get("keyword_score", 0.0)
        
        # Fuse semantic and keyword (if available)
        if keyword_score > 0:
            base_score = 0.7 * semantic_score + 0.3 * keyword_score
        else:
            base_score = semantic_score
        
        # ── Edit type matching ──
        type_match_weight = self._compute_type_match_weight(decomposed, candidate)
        
        # ── Object keyword matching (with length normalization) ──
        object_match_weight = self._compute_object_match_weight(decomposed, candidate)
        
        # ── Length penalty (prefer concise examples over long ones) ──
        instruction_len = len(candidate.get("instruction", "").split())
        length_penalty = 1.0 if instruction_len <= 20 else 0.95 - (0.01 * max(0, instruction_len - 20))
        length_penalty = max(0.8, length_penalty)  # Floor at 0.8
        
        # Combined score: weighted multi-signal fusion
        final_score = (
            0.4 * base_score +
            0.3 * type_match_weight +
            0.2 * object_match_weight +
            0.1 * length_penalty
        )
        
        return final_score
    
    def _compute_type_match_weight(self, decomposed: DecomposedPrompt, candidate: Dict) -> float:
        """Score based on edit type matching."""
        if not decomposed.edit_types or decomposed.edit_types[0] == "generic":
            return 1.0
        
        candidate_type = candidate.get("edit_type", "generic")
        query_type = decomposed.edit_types[0]
        
        if candidate_type == query_type:
            return 1.5  # Strong boost for type match
        elif candidate_type == "generic":
            return 0.9  # Slight penalty for generic candidate
        else:
            return 0.6  # Significant penalty for type mismatch
    
    def _compute_object_match_weight(self, decomposed: DecomposedPrompt, candidate: Dict) -> float:
        """Score based on object mention matching."""
        query_objects = [obj.lower() for obj in decomposed.objects if obj]
        
        if not query_objects:
            return 1.0
        
        candidate_text = candidate.get("instruction", "").lower()
        
        # Count word-level matches (avoid substring false positives)
        candidate_words = set(candidate_text.split())
        exact_matches = sum(
            1 for obj in query_objects 
            if obj in candidate_words or any(obj in word for word in candidate_words)
        )
        
        match_ratio = exact_matches / len(query_objects)
        
        if match_ratio > 0.5:
            return 1.0 + (0.5 * match_ratio)  # Boost for good matches
        elif match_ratio > 0:
            return 1.0 + (0.2 * match_ratio)  # Small boost for partial matches
        else:
            return 0.8  # Penalty for no object matches
    
    def _filter_diversity(self, scored: List[Dict], top_k: int, 
                         similarity_threshold: float = 0.85) -> List[Dict]:
        """
        Filter for diversity: penalize results too similar to already-selected ones.
        This prevents retrieving multiple near-duplicate examples.
        """
        if len(scored) <= top_k:
            return scored
        
        selected = []
        for candidate in scored:
            # Check if this candidate is too similar to any already-selected one
            is_duplicate = False
            for selected_candidate in selected:
                # Simple substring similarity check
                if self._str_similarity(
                    candidate.get("instruction", ""),
                    selected_candidate.get("instruction", "")
                ) > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected.append(candidate)
            
            if len(selected) >= top_k:
                break
        
        return selected if selected else scored[:top_k]
    
    @staticmethod
    def _str_similarity(s1: str, s2: str) -> float:
        """Compute Jaccard similarity between two strings (0-1)."""
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _infer_edit_type(self, instruction: str) -> str:
        """Infer edit type from instruction text (same logic as decomposer)."""
        for etype, patterns in EDIT_PATTERNS.items():
            if any(re.search(p, instruction, re.IGNORECASE) for p in patterns):
                return etype
        return "generic"


# ─────────────────────────────────────────────────────────────
# Edit State Memory  (Novelty 2)
# ─────────────────────────────────────────────────────────────
@dataclass
class EditState:
    """Tracks what has been removed / added / preserved so far."""
    removed_objects:  List[str] = field(default_factory=list)
    added_objects:    List[str] = field(default_factory=list)
    preserved_items:  List[str] = field(default_factory=list)
    applied_edits:    List[str] = field(default_factory=list)  # history
    step_index:       int       = 0


class EditStateMemory:
    """
    Maintains structured scene state across multi-step edits.
    After each edit the state is updated and injected into the
    next diffusion prompt to prevent drift.
    """

    def __init__(self):
        self.state = EditState()

    def update(self, decomposed: DecomposedPrompt):
        """Call after each edit step to update memory."""
        for task, etype in zip(decomposed.sub_tasks, decomposed.edit_types):
            if etype == "remove":
                obj = self._extract_object(task, "remove")
                if obj:
                    self.state.removed_objects.append(obj)
            elif etype == "add":
                obj = self._extract_object(task, "add")
                if obj:
                    self.state.added_objects.append(obj)
            self.state.applied_edits.append(task)

        for hint in decomposed.preserve_hints:
            if hint and hint not in self.state.preserved_items:
                self.state.preserved_items.append(hint)

        self.state.step_index += 1

    def build_constraint_prompt(self, new_prompt: str) -> str:
        """
        Inject preservation constraints into the diffusion prompt
        to prevent undoing prior edits.
        """
        constraints: List[str] = []

        if self.state.removed_objects:
            constraints.append(
                "without " + ", ".join(self.state.removed_objects)
            )
        if self.state.preserved_items:
            constraints.append(
                "preserving " + ", ".join(self.state.preserved_items)
            )
        if self.state.added_objects:
            constraints.append(
                "keeping " + ", ".join(self.state.added_objects)
            )

        if constraints:
            return new_prompt + ", " + ", ".join(constraints)
        return new_prompt

    def reset(self):
        self.state = EditState()

    def _extract_object(self, text: str, verb: str) -> Optional[str]:
        pattern = rf"\b{verb}\b\s+(?:the\s+|a\s+|an\s+)?(.+)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;")
        return None

    def to_dict(self) -> Dict:
        s = self.state
        return {
            "removed":   s.removed_objects,
            "added":     s.added_objects,
            "preserved": s.preserved_items,
            "history":   s.applied_edits,
            "step":      s.step_index,
        }
