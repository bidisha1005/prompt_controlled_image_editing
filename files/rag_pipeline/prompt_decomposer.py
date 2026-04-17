"""
Prompt Decomposition + Edit State Memory
─────────────────────────────────────────
Novelty 1 – Prompt-Decomposed RAG Retrieval
Novelty 2 – Edit State Memory with Constraint Propagation

Supported edit types
────────────────────
  add         – "add a snowman on the left"
  remove      – "remove the car"
  transform   – "change the potato to a rose", "turn the person into a building"
  background  – "make the background a sunset", "change sky to night"
  color       – "make the car red"
  style       – "make it look like a watercolor painting"
  lighting    – "add golden hour lighting"
  generic     – fallback

Key improvements over v1
─────────────────────────
1. SVO (subject–verb–object) parser extracts (subject, target) pairs cleanly.
2. `transform` is a first-class edit type with explicit from/to fields.
3. Multi-word object extraction (e.g. "school bus", "cherry blossom tree").
4. Background edits always capture what to change TO, not just keyword-match.
5. Preserve hints strip filler words robustly.
6. EditStateMemory tracks transformed_objects separately and builds richer prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

PRESERVE_KEYWORDS = [
    "keep", "preserve", "maintain", "leave", "don't change",
    "do not change", "intact", "same", "unchanged", "retain",
]

# Verb sets for edit classification
_ADD_VERBS    = {"add", "insert", "place", "put", "include", "draw", "create",
                 "introduce", "attach", "append", "stick"}
_REMOVE_VERBS = {"remove", "delete", "erase", "eliminate", "get rid of",
                 "cut out", "wipe out", "take out", "take away", "hide"}
_TRANSFORM_VERBS = {"change", "turn", "convert", "replace", "swap", "make",
                    "transform", "morph", "switch"}
_COLOR_VERBS     = {"color", "paint", "colour", "recolor", "tint", "dye"}
_STYLE_VERBS     = {"stylize", "style", "render", "illustrate"}
_LIGHTING_VERBS  = {"illuminate", "light", "brighten", "darken", "shade"}

# Background scene words – if any appear and the verb is transform-like, it's a `background` edit
_BACKGROUND_WORDS = {
    "sunset", "sunrise", "morning", "dawn", "dusk", "evening",
    "night", "nighttime", "noon", "midday", "day", "afternoon",
    "snow", "snowy", "winter", "summer", "stormy", "foggy", "misty",
    "rainy", "cloudy", "golden hour", "blue hour",
    "beach", "forest", "mountain", "city", "urban", "desert",
    "underwater", "space", "countryside", "field", "meadow",
    "sky", "background", "scene", "environment", "atmosphere",
}

# Color names for color-change detection
_COLOR_NAMES = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "grey", "brown", "cyan", "magenta",
    "gold", "silver", "violet", "indigo", "teal", "maroon", "navy",
    "beige", "khaki", "coral", "turquoise", "crimson", "scarlet",
}

# Prepositions / location words that signal placement for `add`
_PLACEMENT_PREPS = {
    "on", "in", "at", "beside", "next to", "behind", "in front of",
    "above", "below", "under", "over", "near", "around", "between",
    "on top of", "to the left", "to the right", "left of", "right of",
    "corner", "background", "foreground",
}

# Conjunction patterns for splitting compound prompts
_SPLIT_PATTERN = re.compile(
    r"\s*(?:,\s*(?:and\b)?|;\s*|"
    r"\band\b(?!\s+a\b|\s+an\b|\s+the\b)|"   # "and" but not "and a/an/the" (part of NP)
    r"\bthen\b|\balso\b|\bwhile\b|\bplus\b|\badditionally\b"
    r")\s*",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class EditTask:
    """
    A single, atomic editing instruction parsed from the user prompt.

    Fields
    ------
    raw         : original text of this sub-task
    edit_type   : one of add | remove | transform | background | color |
                  style | lighting | generic
    subject     : the object being acted upon  (e.g. "the car", "sky")
    target      : what to change it TO         (e.g. "rose", "sunset")
                  None for add/remove where target is the object itself
    placement   : location hint for `add`      (e.g. "to the left")
    color       : explicit color name if edit_type == color
    """
    raw:        str
    edit_type:  str        = "generic"
    subject:    str        = ""
    target:     str        = ""
    placement:  str        = ""
    color:      str        = ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class DecomposedPrompt:
    original:       str
    tasks:          List[EditTask] = field(default_factory=list)
    preserve_hints: List[str]      = field(default_factory=list)

    # --- convenience proxies (keep old consumers working) ---
    @property
    def sub_tasks(self) -> List[str]:
        return [t.raw for t in self.tasks]

    @property
    def edit_types(self) -> List[str]:
        return [t.edit_type for t in self.tasks]

    @property
    def objects(self) -> List[str]:
        return [t.subject for t in self.tasks]


# ─────────────────────────────────────────────────────────────
# Core decomposer helpers
# ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Lower-case, collapse whitespace, strip leading articles."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^(?:the|a|an)\s+", "", text)


def _extract_multiword_object(text: str, after_verb: bool = True) -> str:
    """
    Extract a (possibly multi-word) noun phrase.
    Stops at prepositions, conjunctions, or clause boundaries.

    Examples
    --------
    "the school bus"            → "school bus"
    "a red cherry blossom tree" → "red cherry blossom tree"
    "car to rose"               → "car"   (stops at "to")
    """
    # Strip leading determiners
    text = re.sub(r"^(?:the|a|an|my|your|this|that)\s+", "", text.strip(), flags=re.IGNORECASE)

    # Stop tokens
    stop_pattern = re.compile(
        r"\b(?:to|into|as|with|on|in|at|beside|near|above|below|behind|"
        r"under|over|between|next to|in front of|and|but|so|or|,|;)\b",
        re.IGNORECASE,
    )
    m = stop_pattern.search(text)
    if m:
        text = text[: m.start()].strip()

    # Cap at 4 words to avoid swallowing whole sentences
    words = text.split()
    return " ".join(words[:4]).strip(" .,;")


def _match_transform(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect "X to Y" / "X into Y" / "X as Y" patterns.
    Returns (from_subject, to_target) or None.

    Handles
    -------
    "change the potato to a rose"
    "turn the person into a building"
    "replace the car with a motorbike"
    "convert the sketch to a photograph"
    """
    patterns = [
        # change/turn/convert/transform X to/into Y
        r"(?:change|turn|convert|transform|morph|replace|swap|make)\s+"
        r"(?:the\s+|a\s+|an\s+|my\s+)?(.+?)\s+(?:to|into|as)\s+(?:a\s+|an\s+|the\s+)?(.+)",
        # replace X with Y
        r"replace\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+with\s+(?:a\s+|an\s+|the\s+)?(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            subj = _extract_multiword_object(m.group(1))
            tgt  = _extract_multiword_object(m.group(2))
            if subj and tgt and subj != tgt:
                return subj, tgt
    return None


def _match_background(text: str) -> Optional[str]:
    """
    Detect background scene changes.
    Returns the target scene string or None.

    Handles
    -------
    "make the background sunset"
    "change the sky to night"
    "make it morning"
    "set the scene to a foggy forest"
    "change the environment to underwater"
    """
    # Explicit background noun then target
    pat1 = re.compile(
        r"(?:change|make|set|turn|convert)\s+"
        r"(?:the\s+)?(?:background|sky|scene|environment|atmosphere|setting)\s+"
        r"(?:to|into|as|look like)?\s*(?:a\s+|an\s+|the\s+)?(.+)",
        re.IGNORECASE,
    )
    # "make it <scene_word>" / "make it look like <scene>"
    pat2 = re.compile(
        r"(?:make|turn|change)\s+(?:it|this|the image|the photo|the picture)\s+"
        r"(?:look like|into|to)?\s*(?:a\s+|an\s+|the\s+)?(.+)",
        re.IGNORECASE,
    )
    # bare scene word anywhere when no strong verb present
    pat3 = re.compile(
        r"(?:background|scene|atmosphere|sky|environment)\s*(?:to|:)?\s*(.+)",
        re.IGNORECASE,
    )

    for pat in [pat1, pat2, pat3]:
        m = pat.search(text)
        if m:
            candidate = _extract_multiword_object(m.group(1))
            if any(word in candidate for word in _BACKGROUND_WORDS) or \
               any(word in text.lower() for word in _BACKGROUND_WORDS):
                return candidate if candidate else None

    # Last resort: if any background word appears with a change-like verb, use that word
    text_lower = text.lower()
    has_change_verb = any(v in text_lower for v in
                          {"change", "make", "set", "turn", "convert", "switch"})
    if has_change_verb:
        for bw in _BACKGROUND_WORDS:
            if bw in text_lower:
                # Try to extract more context around the background word
                idx = text_lower.index(bw)
                snippet = text_lower[max(0, idx - 10): idx + len(bw) + 20]
                candidate = _extract_multiword_object(snippet.strip())
                return candidate or bw

    return None


def _match_color(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect color-change operations.
    Returns (subject, color) or None.

    Handles
    -------
    "make the car red"
    "paint the wall blue"
    "change the sky color to orange"
    """
    # verb + subject + color
    pat1 = re.compile(
        r"(?:make|paint|color|colour|recolor|tint|change)\s+"
        r"(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:color\s+)?(?:to\s+)?(" +
        "|".join(_COLOR_NAMES) + r")\b",
        re.IGNORECASE,
    )
    # color + subject + verb (rare)
    pat2 = re.compile(
        r"\b(" + "|".join(_COLOR_NAMES) + r")\b.{0,25}(?:the\s+|a\s+|an\s+)?(\w+)",
        re.IGNORECASE,
    )
    m = pat1.search(text)
    if m:
        subj  = _extract_multiword_object(m.group(1))
        color = m.group(2).lower()
        if subj:
            return subj, color

    # Simple "make it <color>"
    pat3 = re.compile(r"make\s+it\s+(" + "|".join(_COLOR_NAMES) + r")\b", re.IGNORECASE)
    m = pat3.search(text)
    if m:
        return "", m.group(1).lower()

    return None


def _match_add(text: str) -> Optional[Tuple[str, str]]:
    """
    Detect add operations.
    Returns (object_to_add, placement_hint) or None.

    Handles
    -------
    "add a snowman on the left"
    "place a building behind me"
    "put a motorbike beside the car"
    "insert a friend next to the person"
    """
    add_verb_pat = re.compile(
        r"^(?:add|insert|place|put|include|draw|create|introduce|attach|append|stick)\s+",
        re.IGNORECASE,
    )
    m = add_verb_pat.match(text.strip())
    if not m:
        return None

    rest = text[m.end():]
    # Try to find placement preposition
    placement_pat = re.compile(
        r"\s+(?:on|in|at|beside|next to|behind|in front of|above|below|under|"
        r"over|near|around|between|on top of|to the left|to the right|"
        r"left of|right of|corner|to the)\b(.+)?",
        re.IGNORECASE,
    )
    pm = placement_pat.search(rest)
    placement = ""
    obj_text  = rest
    if pm:
        placement = text[m.end() + pm.start():].strip()
        obj_text  = rest[: pm.start()].strip()

    obj = _extract_multiword_object(obj_text)
    return (obj, placement) if obj else None


def _match_remove(text: str) -> Optional[str]:
    """
    Detect remove operations.
    Returns the object to remove or None.

    Handles
    -------
    "remove the car"
    "delete the person on the right"
    "get rid of the tree"
    "erase the background text"
    """
    pat = re.compile(
        r"(?:remove|delete|erase|eliminate|get rid of|cut out|wipe out|"
        r"take out|take away|hide)\s+(?:the\s+|a\s+|an\s+)?(.+)",
        re.IGNORECASE,
    )
    m = pat.search(text)
    if m:
        return _extract_multiword_object(m.group(1))
    return None


# ─────────────────────────────────────────────────────────────
# Task classifier
# ─────────────────────────────────────────────────────────────

def _classify_task(part: str) -> EditTask:
    """
    Parse a single prompt fragment into an EditTask.
    Classification priority:
      1. remove
      2. add
      3. transform  (also catches background-as-transform)
      4. background (when transform target is a scene word)
      5. color
      6. style
      7. lighting
      8. generic
    """
    text = part.strip()
    text_lower = text.lower()

    # ── 1. remove ───────────────────────────────────────────
    obj = _match_remove(text)
    if obj:
        return EditTask(raw=text, edit_type="remove", subject=obj)

    # ── 2. add ──────────────────────────────────────────────
    result = _match_add(text)
    if result:
        obj, placement = result
        return EditTask(raw=text, edit_type="add", subject=obj, placement=placement)

    # ── 3. transform / background ───────────────────────────
    pair = _match_transform(text)
    if pair:
        subj, tgt = pair
        # Decide: is the target a background/scene concept?
        tgt_lower = tgt.lower()
        is_bg = (
            any(bw in tgt_lower for bw in _BACKGROUND_WORDS)
            or subj in {"sky", "background", "scene", "environment", "atmosphere"}
        )
        if is_bg:
            return EditTask(raw=text, edit_type="background",
                            subject=subj or "background", target=tgt)
        return EditTask(raw=text, edit_type="transform", subject=subj, target=tgt)

    # ── 4. bare background (no explicit subject) ────────────
    bg_target = _match_background(text)
    if bg_target:
        return EditTask(raw=text, edit_type="background",
                        subject="background", target=bg_target)

    # ── 5. color change ─────────────────────────────────────
    color_pair = _match_color(text)
    if color_pair:
        subj, color = color_pair
        return EditTask(raw=text, edit_type="color", subject=subj, color=color)

    # ── 6. style ────────────────────────────────────────────
    style_words = {
        "watercolor", "oil painting", "sketch", "cartoon", "anime",
        "photorealistic", "vintage", "retro", "impressionist", "pencil",
        "3d render", "cinematic", "noir", "comic", "pixel art",
    }
    if any(sw in text_lower for sw in style_words) or \
       any(sv in text_lower for sv in _STYLE_VERBS):
        # Try to extract the style name
        target = ""
        for sw in sorted(style_words, key=len, reverse=True):
            if sw in text_lower:
                target = sw
                break
        return EditTask(raw=text, edit_type="style", subject="image", target=target)

    # ── 7. lighting ─────────────────────────────────────────
    lighting_words = {
        "golden hour", "blue hour", "harsh light", "soft light",
        "backlit", "rim light", "neon", "candlelight", "sunlight",
        "overcast", "dramatic lighting", "studio lighting",
    }
    if any(lw in text_lower for lw in lighting_words) or \
       any(lv in text_lower for lv in _LIGHTING_VERBS):
        target = ""
        for lw in sorted(lighting_words, key=len, reverse=True):
            if lw in text_lower:
                target = lw
                break
        return EditTask(raw=text, edit_type="lighting", subject="scene", target=target)

    # ── 8. generic fallback ─────────────────────────────────
    return EditTask(raw=text, edit_type="generic", subject="")


# ─────────────────────────────────────────────────────────────
# Main decomposer
# ─────────────────────────────────────────────────────────────

def decompose_prompt(prompt: str) -> DecomposedPrompt:
    """
    Split a complex prompt into atomic EditTasks and preservation hints.

    Examples
    --------
    "Remove the car, change the sky to sunset, keep the mountains"
    → tasks:
        EditTask(edit_type='remove',     subject='car')
        EditTask(edit_type='background', subject='sky', target='sunset')
    → preserve_hints: ['mountains']

    "Add a snowman on the left and turn the rose into a lily"
    → tasks:
        EditTask(edit_type='add',       subject='snowman', placement='on the left')
        EditTask(edit_type='transform', subject='rose',    target='lily')

    "Change potato to rose and make background morning, preserve building"
    → tasks:
        EditTask(edit_type='transform',  subject='potato',     target='rose')
        EditTask(edit_type='background', subject='background', target='morning')
    → preserve_hints: ['building']
    """
    result = DecomposedPrompt(original=prompt)

    # ── 1. Split compound prompt ─────────────────────────────
    parts = _SPLIT_PATTERN.split(prompt)
    parts = [p.strip() for p in parts if p.strip()]

    # ── 2. Separate preserve hints from edit instructions ────
    edit_parts: List[str] = []
    for part in parts:
        part_lower = part.lower()
        is_preserve = any(kw in part_lower for kw in PRESERVE_KEYWORDS)
        if is_preserve:
            # Strip the preserve keyword itself to get the subject
            hint = part
            for kw in sorted(PRESERVE_KEYWORDS, key=len, reverse=True):
                hint = re.sub(rf"\b{re.escape(kw)}\b\s*(?:the\s+)?", "",
                              hint, flags=re.IGNORECASE)
            hint = hint.strip(" .,;")
            if hint:
                result.preserve_hints.append(hint)
        else:
            edit_parts.append(part)

    # ── 3. Classify each edit part ───────────────────────────
    if not edit_parts:
        # Whole prompt was preservation hints – treat original as generic
        result.tasks = [EditTask(raw=prompt, edit_type="generic")]
        return result

    for part in edit_parts:
        task = _classify_task(part)
        result.tasks.append(task)

    logger.debug("Decomposed: %s", result)
    return result


# ─────────────────────────────────────────────────────────────
# Attribute extraction (improved)
# ─────────────────────────────────────────────────────────────

class EditAttributeExtractor:
    """
    Extracts semantic attributes from an EditTask for RAG polarity filtering.
    Now operates on structured EditTask fields rather than raw text strings.
    """

    WARM_PALETTE   = {"red","orange","yellow","golden","warm","sunset","sunrise","hot","fire","amber","gold"}
    COOL_PALETTE   = {"blue","cyan","purple","gray","grey","cool","cloud","cloudy","ice","cold","silver","night"}
    BRIGHT_WORDS   = {"bright","light","sunny","vivid","daylight","noon","morning","sunrise","dawn","day"}
    DARK_WORDS     = {"dark","dim","night","shadowy","dusk","midnight","low light","nighttime","evening"}

    @classmethod
    def from_task(cls, task: EditTask) -> Dict[str, str]:
        """Extract attributes from a structured EditTask."""
        combined = f"{task.subject} {task.target} {task.raw}".lower()
        return {
            "color_temp":        cls._color_temp(combined),
            "brightness":        cls._brightness(task.target or task.raw),
            "action_type":       task.edit_type,
            "scope":             "scene" if task.edit_type == "background" else "object",
            "color":             task.color,
        }

    @classmethod
    def _color_temp(cls, text: str) -> str:
        w = sum(1 for w in cls.WARM_PALETTE if w in text)
        c = sum(1 for w in cls.COOL_PALETTE if w in text)
        return "warm" if w > c else ("cool" if c > w else "neutral")

    @classmethod
    def _brightness(cls, text: str) -> str:
        b = sum(1 for w in cls.BRIGHT_WORDS if w in text)
        d = sum(1 for w in cls.DARK_WORDS   if w in text)
        return "bright" if b > d else ("dark" if d > b else "neutral")

    # --- legacy interface for callers that pass raw strings ---
    @classmethod
    def extract(cls, instruction: str) -> Dict[str, str]:
        task = _classify_task(instruction)
        return cls.from_task(task)


# ─────────────────────────────────────────────────────────────
# Query expansion
# ─────────────────────────────────────────────────────────────

class QueryExpander:
    """Expands queries using EditTask structure + synonym map."""

    SYNONYM_MAP: Dict[str, List[str]] = {
        "car":       ["vehicle", "automobile"],
        "tree":      ["plant", "vegetation"],
        "sky":       ["background", "atmosphere"],
        "person":    ["human", "figure", "subject"],
        "building":  ["structure", "architecture"],
        "water":     ["ocean", "sea", "lake"],
        "remove":    ["delete", "erase", "eliminate"],
        "add":       ["insert", "place", "put"],
        "change":    ["modify", "alter", "transform"],
        "sunset":    ["golden hour", "warm sky", "orange sky"],
        "morning":   ["sunrise", "bright sky", "dawn"],
        "night":     ["nighttime", "dark sky", "moonlit"],
        "rose":      ["flower", "bloom"],
    }

    @staticmethod
    def expand_task(task: EditTask, num_expansions: int = 4) -> List[str]:
        """Generate query variants from an EditTask."""
        base = task.raw.lower()
        variants = [base]

        # Structured expansion from task fields
        if task.edit_type == "transform":
            variants.append(f"change {task.subject} to {task.target}")
            variants.append(f"replace {task.subject} with {task.target}")
        elif task.edit_type == "background":
            variants.append(f"background {task.target}")
            variants.append(f"scene {task.target} lighting atmosphere")
        elif task.edit_type == "add":
            variants.append(f"add {task.subject} {task.placement}".strip())
        elif task.edit_type == "remove":
            variants.append(f"remove {task.subject} erase delete")

        # Synonym expansion
        for term, syns in QueryExpander.SYNONYM_MAP.items():
            if term in base:
                for syn in syns[:1]:
                    v = base.replace(term, syn)
                    if v not in variants:
                        variants.append(v)
            if len(variants) >= num_expansions + 1:
                break

        return variants[:num_expansions + 1]

    @staticmethod
    def expand(query: str, num_expansions: int = 4) -> List[str]:
        """Legacy interface: expand a raw string query."""
        task = _classify_task(query)
        return QueryExpander.expand_task(task, num_expansions)


# ─────────────────────────────────────────────────────────────
# RAG retriever scorer  (improved)
# ─────────────────────────────────────────────────────────────

# Re-use EDIT_PATTERNS for backward-compat infer_edit_type
EDIT_PATTERNS = {
    "remove":     [r"\b(remove|delete|erase|get rid of|eliminate)\b"],
    "color":      [r"\b(color|colour|paint|recolor|tint)\b",
                   r"\b(red|blue|green|yellow|white|black|purple|orange|pink|gray|grey|brown)\b"],
    "background": [r"\b(morning|sunrise|golden hour|sunset|dusk|evening|night|nighttime|"
                   r"dawn|snow|snowy|storm|thunderstorm|lightning|landscape|cityscape|"
                   r"urban|beach|ocean|forest|mountain|field)\b"],
    "lighting":   [r"\b(lighting|light|illuminate|golden hour|blue hour)\b"],
    "add":        [r"\b(add|insert|place|put|include)\b"],
    "transform":  [r"\b(change|turn|convert|replace|swap|transform|morph)\b.{0,20}"
                   r"\b(to|into|as|with)\b"],
    "style":      [r"\b(watercolor|oil painting|sketch|cartoon|anime|photorealistic|vintage)\b"],
}


class RAGRetrieverScorer:
    """
    Multi-signal re-ranking with polarity hard-filtering.
    Updated to work with both legacy dict-based candidates and
    structured EditTask-aware scoring.
    """

    def score_results(
        self,
        decomposed: DecomposedPrompt,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        if not candidates or not decomposed.tasks:
            return candidates[:top_k]

        for c in candidates:
            if "edit_type" not in c:
                c["edit_type"] = self._infer_edit_type(c.get("instruction", ""))

        query_task = decomposed.tasks[0]

        scored = []
        for c in candidates:
            if self._polarity_reject(query_task, c):
                logger.debug("Hard-rejected (polarity): %s",
                             c.get("instruction", "")[:60])
                continue
            scored.append({**c, "relevance_score": self._score(query_task, c)})

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return self._diversify(scored, top_k)

    # ── Polarity hard-filter ─────────────────────────────────

    def _polarity_reject(self, task: EditTask, candidate: Dict) -> bool:
        """Return True if candidate should be HARD-REJECTED."""
        cand_text = candidate.get("instruction", "").lower()
        cand_task = _classify_task(candidate.get("instruction", ""))
        q_attrs   = EditAttributeExtractor.from_task(task)
        c_attrs   = EditAttributeExtractor.from_task(cand_task)

        # 1. Opposite brightness for background edits
        if task.edit_type == "background":
            if (q_attrs["brightness"] != "neutral" and
                    c_attrs["brightness"] != "neutral" and
                    q_attrs["brightness"] != c_attrs["brightness"]):
                return True

        # 2. Opposite color temperature for background edits
        if task.edit_type == "background":
            if (q_attrs["color_temp"] != "neutral" and
                    c_attrs["color_temp"] != "neutral" and
                    q_attrs["color_temp"] != c_attrs["color_temp"]):
                return True

        # 3. Opposite actions (add ↔ remove)
        if (task.edit_type == "add"    and cand_task.edit_type == "remove") or \
           (task.edit_type == "remove" and cand_task.edit_type == "add"):
            return True

        # 4. Transform to incompatible type (e.g. animate ↔ realistic)
        # — left to semantic similarity to handle

        return False

    # ── Relevance score ──────────────────────────────────────

    def _score(self, task: EditTask, candidate: Dict) -> float:
        base      = candidate.get("similarity", 0.0)
        kw_score  = candidate.get("keyword_score", 0.0)
        if kw_score > 0:
            base = 0.7 * candidate.get("semantic_score", base) + 0.3 * kw_score

        type_w   = self._type_weight(task, candidate)
        obj_w    = self._object_weight(task, candidate)
        target_w = self._target_weight(task, candidate)
        len_pen  = self._length_penalty(candidate.get("instruction", ""))

        return (
            0.30 * base +
            0.22 * type_w +
            0.18 * obj_w  +
            0.18 * target_w +
            0.12 * len_pen
        )

    def _type_weight(self, task: EditTask, candidate: Dict) -> float:
        ct = candidate.get("edit_type", "generic")
        if ct == task.edit_type:
            return 1.5
        if ct == "generic":
            return 0.9
        # Partial credit for related types
        related = {
            "transform": {"color", "style"},
            "background": {"lighting", "style"},
            "color": {"transform"},
        }
        if ct in related.get(task.edit_type, set()):
            return 1.0
        return 0.6

    def _object_weight(self, task: EditTask, candidate: Dict) -> float:
        if not task.subject:
            return 1.0
        cand_words = set(candidate.get("instruction", "").lower().split())
        subj_words = set(task.subject.lower().split())
        overlap    = len(subj_words & cand_words) / max(len(subj_words), 1)
        return 1.0 + 0.5 * overlap if overlap > 0.5 else 1.0 + 0.2 * overlap

    def _target_weight(self, task: EditTask, candidate: Dict) -> float:
        """NEW: boost candidates whose instruction mentions the target concept."""
        if not task.target:
            return 1.0
        cand_lower = candidate.get("instruction", "").lower()
        tgt_words  = set(task.target.lower().split())
        overlap    = sum(1 for w in tgt_words if w in cand_lower) / max(len(tgt_words), 1)
        return 1.0 + 0.4 * overlap

    @staticmethod
    def _length_penalty(instruction: str) -> float:
        n = len(instruction.split())
        if n <= 20:
            return 1.0
        return max(0.8, 0.95 - 0.01 * (n - 20))

    def _diversify(self, scored: List[Dict], top_k: int,
                   threshold: float = 0.85) -> List[Dict]:
        selected: List[Dict] = []
        for c in scored:
            if not any(self._jaccard(c.get("instruction", ""),
                                     s.get("instruction", "")) > threshold
                       for s in selected):
                selected.append(c)
            if len(selected) >= top_k:
                break
        return selected or scored[:top_k]

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

    def _infer_edit_type(self, instruction: str) -> str:
        for etype, patterns in EDIT_PATTERNS.items():
            if any(re.search(p, instruction, re.IGNORECASE) for p in patterns):
                return etype
        return "generic"


# ─────────────────────────────────────────────────────────────
# Edit State Memory  (improved)
# ─────────────────────────────────────────────────────────────

@dataclass
class EditState:
    removed_objects:     List[str] = field(default_factory=list)
    added_objects:       List[str] = field(default_factory=list)
    transformed_objects: List[Tuple[str, str]] = field(default_factory=list)  # (from, to)
    background:          str       = ""   # most recent background target
    preserved_items:     List[str] = field(default_factory=list)
    applied_edits:       List[str] = field(default_factory=list)
    step_index:          int       = 0


class EditStateMemory:
    """
    Maintains structured scene state across multi-step edits.
    After each edit the state is updated and injected into the
    next diffusion prompt to prevent drift.
    """

    def __init__(self):
        self.state = EditState()

    def update(self, decomposed: DecomposedPrompt) -> None:
        """Call after each edit step to update memory from a DecomposedPrompt."""
        for task in decomposed.tasks:
            if task.edit_type == "remove" and task.subject:
                if task.subject not in self.state.removed_objects:
                    self.state.removed_objects.append(task.subject)

            elif task.edit_type == "add" and task.subject:
                if task.subject not in self.state.added_objects:
                    self.state.added_objects.append(task.subject)

            elif task.edit_type == "transform" and task.subject and task.target:
                # Remove the old object from added/present tracking
                if task.subject in self.state.added_objects:
                    self.state.added_objects.remove(task.subject)
                self.state.transformed_objects.append((task.subject, task.target))

            elif task.edit_type == "background" and task.target:
                self.state.background = task.target

            self.state.applied_edits.append(task.raw)

        for hint in decomposed.preserve_hints:
            if hint and hint not in self.state.preserved_items:
                self.state.preserved_items.append(hint)

        self.state.step_index += 1

    def build_constraint_prompt(self, new_prompt: str) -> str:
        """
        Inject preservation constraints into the diffusion prompt
        to prevent undoing prior edits.

        Example output
        ──────────────
        "add a snowman, without car, keeping building, background: golden hour,
         [rose was replaced by lily], preserving mountains"
        """
        parts: List[str] = [new_prompt.rstrip(" ,.")]

        if self.state.removed_objects:
            parts.append("without " + ", ".join(self.state.removed_objects))

        if self.state.added_objects:
            parts.append("keeping " + ", ".join(self.state.added_objects))

        if self.state.transformed_objects:
            clauses = [f"{frm} replaced by {to}"
                       for frm, to in self.state.transformed_objects]
            parts.append("[" + "; ".join(clauses) + "]")

        if self.state.background:
            parts.append(f"background: {self.state.background}")

        if self.state.preserved_items:
            parts.append("preserving " + ", ".join(self.state.preserved_items))

        return ", ".join(parts)

    def reset(self) -> None:
        """Completely reset the edit state."""
        self.state = EditState()
        logger.info("EditStateMemory reset.")

    def to_dict(self) -> Dict:
        s = self.state
        return {
            "removed":     s.removed_objects,
            "added":       s.added_objects,
            "transformed": [{"from": f, "to": t} for f, t in s.transformed_objects],
            "background":  s.background,
            "preserved":   s.preserved_items,
            "history":     s.applied_edits,
            "step":        s.step_index,
        }