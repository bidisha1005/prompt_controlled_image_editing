"""
Prompt Reconstructor using Groq LLM API
Refines decomposed sub-tasks using RAG references for better image editing instructions.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_pipeline.prompt_decomposer import EditTask, decompose_prompt

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """Result of prompt reconstruction"""
    prompt: str
    used_llm: bool
    model: str
    references_used: int


class GroqPromptReconstructor:
    """
    Reconstructs editing prompts using Groq LLM + RAG references.
    Enhances decomposed sub-tasks with specific visual instructions.
    """

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.available = GROQ_AVAILABLE and bool(self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Latest available Groq model
        self.client = None

        if self.available:
            try:
                self.client = groq.Groq(api_key=self.api_key)
                logger.info("✅ Groq LLM initialized for prompt reconstruction")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Groq: {e}")
                self.available = False

    def status(self) -> Dict[str, Any]:
        """Return status of the reconstructor"""
        return {
            "available": self.available,
            "model": self.model if self.available else "fallback",
            "provider": "groq" if self.available else "none",
        }

    def reconstruct(
        self,
        sub_task: str,
        references: List[Dict[str, Any]],
        state: Dict[str, Any],
        preserve_hints: Optional[List[str]] = None,
        edit_type: str = "generic",
    ) -> ReconstructionResult:
        """
        Reconstruct a sub-task prompt using Groq LLM and RAG references.

        Args:
            sub_task: The decomposed sub-task description
            references: List of retrieved RAG references with matching examples
            state: Current editing state/memory
            preserve_hints: Hints to preserve from the original prompt
            edit_type: Type of edit (color, lighting, object, etc.)

        Returns:
            ReconstructionResult with refined prompt and metadata
        """

        if not self.available:
            # Fallback: use sub_task as-is with references appended
            return self._fallback_reconstruct(
                sub_task, references, preserve_hints, edit_type
            )

        task = self._parse_task(sub_task, edit_type)

        # Build context from references
        ref_context = self._build_reference_context(references)
        preserve_context = ""
        if preserve_hints:
            preserve_context = f"\nPreserve: {', '.join(preserve_hints)}"

        # Construct prompt for Groq
        system_prompt = """You are an expert image editing prompt engineer specializing in PRECISE OBJECT ISOLATION.

CRITICAL RULES:
1. Target ONLY the specific object being edited (e.g., 'ONLY the car', 'ONLY the person')
2. Explicitly state what should NOT change (e.g., 'Do NOT change the road', 'Keep background untouched')
3. Use specific object boundaries and anatomy
4. Avoid vague terms like 'exterior surfaces' or 'all surrounding'
5. Use negative framing: 'Keep [X] unchanged' not just 'focus on [Y]'

Focus on concrete, precise visual changes with clear object boundaries.
Keep responses concise (2-3 sentences max)."""

        # Extract structured task details so transform prompts can describe the target,
        # not just preservation constraints.
        edited_object = task.subject or self._extract_object(sub_task)
        target_object = task.target or ""
        target_guidance = self._target_visual_guidance(target_object, references)
        transform_context = ""
        if task.edit_type == "transform" and edited_object and target_object:
            transform_context = (
                f"\nTransformation: replace the {edited_object} with a {target_object}"
                f"\nTarget appearance cues: {target_guidance or f'clear {target_object} identity'}"
            )
        elif target_object:
            transform_context = (
                f"\nDesired result: emphasize {target_object}"
                f"\nVisual cues: {target_guidance or f'clear {target_object} appearance'}"
            )

        object_constraint = f"\nTarget object ONLY: {edited_object}" if edited_object else ""

        # Build a list of likely elements to preserve based on common scene elements
        # This helps prevent unwanted changes to background, ground, sky, etc.
        preserve_list = self._infer_preserve_hints(sub_task, edited_object)
        preserve_statement = ""
        if preserve_list:
            preserve_statement = f"\nDO NOT CHANGE: {', '.join(preserve_list)}"
        
        user_prompt = f"""Edit task: {sub_task}
Edit type: {task.edit_type or edit_type}{object_constraint}{transform_context}
{preserve_context}{preserve_statement}

Reference examples from similar edits:
{ref_context}

CREATE A PRECISE PROMPT THAT:
1. ONLY modifies the target object ({edited_object or 'specified element'})
2. If this is a transform, explicitly describe the NEW target object with 2-4 concrete visual cues
3. Explicitly forbids changes to background elements
4. Uses negative guidance for all non-target elements
5. Provides specific visual boundaries

Example structure:
"Transform only the [SOURCE] into a [TARGET] with [VISUAL CUES]. Keep the [BACKGROUND] and all other elements exactly unchanged."

Refined editing instruction:"""  

        try:
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )

            refined_prompt = message.choices[0].message.content.strip()

            return ReconstructionResult(
                prompt=refined_prompt,
                used_llm=True,
                model=self.model,
                references_used=len(references),
            )

        except Exception as e:
            logger.warning(f"Groq reconstruction failed: {e}, using fallback")
            return self._fallback_reconstruct(
                sub_task, references, preserve_hints, edit_type
            )

    def _build_reference_context(self, references: List[Dict[str, Any]]) -> str:
        """Build reference context from RAG results"""
        if not references:
            return "(No similar examples found)"

        context_lines = []
        for i, ref in enumerate(references[:3], 1):  # Use top 3
            score = ref.get("score", ref.get("relevance_score", ref.get("similarity", 0)))
            match = ref.get("instruction") or ref.get("match", "")
            match = match[:140]
            context_lines.append(
                f"{i}. [score: {score:.2f}] {match}"
            )

        return "\n".join(context_lines)

    def _parse_task(self, sub_task: str, edit_type: str) -> EditTask:
        """Parse a single sub-task using the shared decomposer."""
        try:
            decomposed = decompose_prompt(sub_task)
            if decomposed.tasks:
                task = decomposed.tasks[0]
                if edit_type != "generic" and task.edit_type == "generic":
                    task.edit_type = edit_type
                return task
        except Exception as e:
            logger.debug("Structured sub-task parsing failed for '%s': %s", sub_task, e)

        return EditTask(raw=sub_task, edit_type=edit_type or "generic")

    def _target_visual_guidance(
        self,
        target_object: str,
        references: List[Dict[str, Any]],
    ) -> str:
        """Generate concise target-object cues so transforms describe the new object clearly."""
        target = (target_object or "").lower().strip()
        if not target:
            return ""

        target_map = {
            "truck": "larger vehicle body, visible cargo bed or truck cabin, bigger wheels, truck-like proportions",
            "pickup truck": "pickup cabin, open cargo bed, rugged tires, truck proportions",
            "bus": "elongated body, large windows, passenger bus proportions, multiple visible wheels",
            "motorcycle": "two-wheel frame, handlebars, compact rider-scale body, exposed chassis",
            "bicycle": "two thin wheels, pedals, handlebars, lightweight frame",
            "rose": "layered petals, floral bloom shape, stem-centered flower form",
            "lily": "elongated petals, open flower shape, delicate botanical structure",
            "building": "architectural facade, clear windows or structural surfaces, rigid geometric form",
            "person": "human anatomy, head torso limbs, natural body proportions",
            "dog": "canine body shape, four legs, muzzle, fur texture",
            "cat": "feline body shape, pointed ears, compact face, fur texture",
        }

        for key, guidance in target_map.items():
            if key in target:
                return guidance

        ref_cues: List[str] = []
        target_words = set(target.split())
        for ref in references[:3]:
            instruction = (ref.get("instruction") or ref.get("match") or "").lower()
            if not instruction or not target_words.intersection(instruction.split()):
                continue
            if "with " in instruction:
                cue = instruction.split("with ", 1)[1].strip(" .,;")
                if cue:
                    ref_cues.append(cue[:90])

        if ref_cues:
            return "; ".join(ref_cues[:2])

        return f"recognizable {target} shape, proportions, and defining parts"

    def _infer_preserve_hints(self, sub_task: str, edited_object: str) -> List[str]:
        """Infer what elements should be preserved based on edited object.
        
        If editing a car, preserve: road, trees, sky, background
        If editing sky, preserve: car, road, objects
        Etc.
        """
        preserve_map = {
            "car": ["road", "pavement", "trees", "sky", "background", "surroundings"],
            "vehicle": ["road", "pavement", "trees", "sky", "background"],
            "road": ["car", "vehicle", "trees", "people", "sky"],
            "sky": ["car", "vehicle", "road", "trees", "buildings", "ground"],
            "tree": ["road", "car", "sky", "other trees", "people"],
            "building": ["sky", "ground", "road", "other buildings", "surroundings"],
            "person": ["background", "surroundings", "other people", "sky"],
        }
        
        if edited_object and edited_object in preserve_map:
            return preserve_map[edited_object]
        
        # Default preservation for any edit: preserve background elements
        return ["background", "other objects", "surroundings"]
    
    def _extract_object(self, sub_task: str) -> str:
        """Extract the main object being edited from a subtask.
        
        Examples:
            "change the car to red" → "car"
            "make the sky purple" → "sky"
            "remove the person" → "person"
            "add a dog" → "dog"
        """
        import re
        
        # Pattern: verb + [the/a/an] + OBJECT
        patterns = [
            r"\b(?:change|make|turn|color|paint|add|remove|delete|make|replace)\s+(?:the\s+)?(?:a\s+)?(?:an\s+)?(\w+)",
            r"\b(?:the|a|an)\s+(\w+)\s+(?:to|a|into)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sub_task.lower())
            if match:
                obj = match.group(1).lower().strip()
                # Filter out non-noun words
                if obj not in ["it", "is", "the", "a", "an", "this", "that", "to", "be"]:
                    return obj
        
        return ""

    def _fallback_reconstruct(
        self,
        sub_task: str,
        references: List[Dict[str, Any]],
        preserve_hints: Optional[List[str]] = None,
        edit_type: str = "generic",
    ) -> ReconstructionResult:
        """Enhanced fallback reconstruction without Groq LLM.
        
        Enriches prompts with visual descriptors and reference context.
        """
        import re
        task = self._parse_task(sub_task, edit_type)
        
        # Visual descriptor templates per edit type
        descriptors = {
            "background": [
                "with vivid, immersive atmosphere",
                "with rich environmental detail",
                "with atmospheric lighting",
                "with natural lighting and depth",
            ],
            "color_change": [
                "with vibrant color intensity",
                "with consistent color saturation",
                "with natural color grading",
                "with enhanced color depth",
            ],
            "add": [
                "naturally integrated into the scene",
                "with proper scale and perspective",
                "seamlessly blended with surroundings",
                "with appropriate lighting and shadows",
            ],
            "remove": [
                "maintaining background continuity",
                "with natural background restoration",
                "seamlessly without artifacts",
                "preserving scene coherence",
            ],
            "lighting": [
                "with realistic shadow placement",
                "with proper light distribution",
                "with natural lighting transitions",
                "with consistent illumination",
            ],
        }
        
        if task.edit_type == "transform" and task.subject and task.target:
            target_guidance = self._target_visual_guidance(task.target, references)
            preserve_items = self._merge_preserve_items(
                preserve_hints,
                self._infer_preserve_hints(sub_task, task.subject),
            )
            refined = (
                f"Transform only the {task.subject} into a {task.target} with {target_guidance}. "
                f"Keep the background, framing, lighting, and all other objects unchanged"
            )
            if preserve_items:
                refined += f", especially {', '.join(preserve_items[:3])}"
            return ReconstructionResult(
                prompt=refined,
                used_llm=False,
                model="fallback-enhanced",
                references_used=len(references),
            )

        # Start with sub_task
        refined = sub_task
        
        # Extract key keywords from references to enrich
        ref_keywords = []
        for ref in references[:2]:
            instruction = ref.get("instruction", "").lower()
            # Extract meaningful words
            words = re.findall(r'\b(?:with|add|make|create|render|show)\s+(\w+(?:\s+\w+)?)', instruction)
            ref_keywords.extend(words)
        
        # Add descriptive phrase based on edit type
        if edit_type in descriptors:
            descriptor = descriptors[edit_type][hash(sub_task) % len(descriptors[edit_type])]
            refined = f"{refined} {descriptor}"
        
        # Add key details from first reference if available
        if ref_keywords:
            if "dark" in str(ref_keywords).lower() or "night" in str(ref_keywords).lower():
                refined += ", emphasizing shadows and depth"
            elif "bright" in str(ref_keywords).lower() or "sunny" in str(ref_keywords).lower():
                refined += ", with bright, clear lighting"
            elif "sun" in str(ref_keywords).lower():
                refined += ", with golden hour lighting"
        
        # Add preservation hints
        if preserve_hints and len(preserve_hints) > 0:
            refined = f"{refined}; preserve: {', '.join(preserve_hints[:2])}"

        return ReconstructionResult(
            prompt=refined,
            used_llm=False,
            model="fallback-enhanced",
            references_used=len(references),
        )

    def _merge_preserve_items(
        self,
        preserve_hints: Optional[List[str]],
        inferred_items: List[str],
    ) -> List[str]:
        """Merge explicit and inferred preserve items without duplicates."""
        merged: List[str] = []
        for item in (preserve_hints or []) + inferred_items:
            cleaned = item.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged
