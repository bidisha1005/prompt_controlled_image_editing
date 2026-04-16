"""
Groq-backed prompt reconstruction for diffusion editing.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


@dataclass
class ReconstructionResult:
    prompt: str
    model: Optional[str]
    used_llm: bool
    references_used: List[str]


class GroqPromptReconstructor:
    """
    Rewrites a decomposed edit task into a concise diffusion-ready prompt.
    Falls back to deterministic reconstruction when Groq is unavailable.
    """

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
        }

    def reconstruct(
        self,
        sub_task: str,
        references: List[Dict[str, Any]],
        state: Dict[str, Any],
        preserve_hints: List[str],
        edit_type: str,
    ) -> ReconstructionResult:
        top_refs = [
            ref.get("instruction", "").strip()
            for ref in references[:2]
            if ref.get("instruction")
        ]

        fallback_prompt = self._fallback_prompt(
            sub_task=sub_task,
            state=state,
            preserve_hints=preserve_hints,
        )

        if not self.enabled:
            return ReconstructionResult(
                prompt=fallback_prompt,
                model=None,
                used_llm=False,
                references_used=top_refs,
            )

        try:
            client = self._get_client()
            completion = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_completion_tokens=180,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite image-editing tasks for diffusion models. "
                            "Return JSON only with keys rewritten_prompt and focus. "
                            "The rewritten prompt must be a single concise instruction, "
                            "visually specific, and must not mention similarity scores, "
                            "RAG, references, metadata, or explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_user_prompt(
                            sub_task=sub_task,
                            references=top_refs,
                            state=state,
                            preserve_hints=preserve_hints,
                            edit_type=edit_type,
                        ),
                    },
                ],
            )

            content = completion.choices[0].message.content or ""
            payload = json.loads(content)
            rewritten = (payload.get("rewritten_prompt") or "").strip()

            if not rewritten:
                raise ValueError("Groq returned an empty rewritten_prompt")

            return ReconstructionResult(
                prompt=rewritten,
                model=self.model,
                used_llm=True,
                references_used=top_refs,
            )
        except Exception as exc:
            logger.warning("Groq prompt reconstruction failed, using fallback: %s", exc)
            return ReconstructionResult(
                prompt=fallback_prompt,
                model=self.model,
                used_llm=False,
                references_used=top_refs,
            )

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "The groq package is not installed. Add it to your environment first."
            ) from exc

        self._client = Groq(api_key=self.api_key)
        return self._client

    def _build_user_prompt(
        self,
        sub_task: str,
        references: List[str],
        state: Dict[str, Any],
        preserve_hints: List[str],
        edit_type: str,
    ) -> str:
        lines = [
            "Rewrite this image editing task into one diffusion-ready prompt.",
            f"Current sub-task: {sub_task}",
            f"Edit type: {edit_type}",
            f"Preserve hints: {', '.join(preserve_hints) if preserve_hints else 'none'}",
            f"Previously removed: {', '.join(state.get('removed', [])) or 'none'}",
            f"Previously added: {', '.join(state.get('added', [])) or 'none'}",
            f"Previously preserved: {', '.join(state.get('preserved', [])) or 'none'}",
            "Top retrieved references:",
        ]

        if references:
            lines.extend(f"- {ref}" for ref in references)
        else:
            lines.append("- none")

        lines.extend(
            [
                "",
                "Rules:",
                "- Keep the prompt under 35 words.",
                "- Mention visual changes directly.",
                "- Preserve scene layout and unrelated content unless the task changes them.",
                "- Do not mention RAG, references, similar examples, scores, or explanations.",
                "- Output JSON only.",
            ]
        )
        return "\n".join(lines)

    def _fallback_prompt(
        self,
        sub_task: str,
        state: Dict[str, Any],
        preserve_hints: List[str],
    ) -> str:
        constraints: List[str] = []

        removed = state.get("removed", [])
        added = state.get("added", [])
        preserved = list(state.get("preserved", []))

        for hint in preserve_hints:
            if hint and hint not in preserved:
                preserved.append(hint)

        if preserved:
            constraints.append("preserve " + ", ".join(preserved))
        if removed:
            constraints.append("do not bring back " + ", ".join(removed))
        if added:
            constraints.append("keep " + ", ".join(added))

        if constraints:
            return f"{sub_task}, and " + ", and ".join(constraints)
        return sub_task
