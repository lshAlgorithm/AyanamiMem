"""Depth-aware LLM summarisation with three-level escalation.

The summarizer reads children ``.md`` files, feeds their content to an LLM
with a depth-appropriate prompt, and returns the summary text + metadata.
If the summary is too long, it escalates to a more aggressive prompt, then
falls back to deterministic truncation.
"""

import json

from typing import Any

from memos.log import get_logger
from memos.memories.textual.hierarchical_markdown.fs import read_md
from memos.templates.hierarchical_markdown_prompts import (
    AGGRESSIVE_SUMMARY_PROMPT,
    CONDENSED_D1_PROMPT,
    CONDENSED_D2_PLUS_PROMPT,
    LEAF_SUMMARY_PROMPT,
    UPDATE_SUMMARY_PROMPT,
)


logger = get_logger(__name__)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output."""
    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return {}
    fragment = text[start : end + 1]
    # Strip markdown code fences
    fragment = fragment.replace("```json", "").replace("```", "")
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return {}


class HierarchicalSummarizer:
    """Produce condensed ``_summary.md`` content from children nodes."""

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        condensed_target_tokens: int = 512,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.condensed_target_tokens = condensed_target_tokens

    def summarize(
        self,
        children_paths: list[str],
        depth: int,
        previous_summary: str | None = None,
    ) -> tuple[str, str, list[str]]:
        """Summarise children into a condensed node.

        Args:
            children_paths: Absolute paths to children ``.md`` files.
            depth: The depth of the new condensed node (children are depth-1).
            previous_summary: If re-summarising, the old summary text.

        Returns:
            ``(summary_body, key, tags)``
        """
        children_text = self._read_children(children_paths)

        # Pick prompt by depth
        if previous_summary is not None:
            prompt_template = UPDATE_SUMMARY_PROMPT
        elif depth == 1:
            prompt_template = LEAF_SUMMARY_PROMPT
        elif depth == 2:
            prompt_template = CONDENSED_D1_PROMPT
        else:
            prompt_template = CONDENSED_D2_PLUS_PROMPT

        # Level 1: Normal summarisation
        result = self._run_llm(prompt_template, children_text, previous_summary, temperature=0.2)
        if result and self._is_acceptable(result["summary"]):
            return result["summary"], result["key"], result.get("tags", [])

        # Level 2: Aggressive prompt
        logger.info("Escalating to aggressive summary (depth=%d)", depth)
        result = self._run_llm(
            AGGRESSIVE_SUMMARY_PROMPT, children_text, previous_summary, temperature=0.1
        )
        if result and self._is_acceptable(result["summary"]):
            return result["summary"], result["key"], result.get("tags", [])

        # Level 3: Deterministic truncation fallback
        logger.warning("Falling back to truncation (depth=%d)", depth)
        return self._truncate(children_text), "summary", []

    def summarize_for_key(self, text: str) -> tuple[str, list[str]]:
        """Lightweight: extract just a key and tags from text (no full summary)."""
        prompt = (
            "Given this text, produce a short descriptive key (5-15 words) and 1-5 tags.\n\n"
            f"<text>\n{text[:2000]}\n</text>\n\n"
            'Return ONLY valid JSON: {"key": "...", "tags": ["..."]}'
        )
        try:
            response = self.llm.generate([{"role": "user", "content": prompt}])
            data = _parse_json_response(response)
            return data.get("key", text[:60]), data.get("tags", [])
        except Exception:
            logger.warning("Key extraction failed, using fallback")
            return text[:60].replace("\n", " "), []

    def _run_llm(
        self,
        template: str,
        children_text: str,
        previous_summary: str | None,
        temperature: float,
    ) -> dict[str, Any] | None:
        """Fill prompt template and call LLM."""
        previous_context = ""
        if previous_summary:
            previous_context = (
                f"The previous summary was:\n<previous>\n{previous_summary}\n</previous>\n"
                "Update it based on the new children content."
            )

        prompt = template.replace("${conversation}", children_text)
        prompt = prompt.replace("${children_summaries}", children_text)
        prompt = prompt.replace("${content}", children_text)
        prompt = prompt.replace("${previous_summary}", previous_summary or "")
        prompt = prompt.replace("${previous_context}", previous_context)
        prompt = prompt.replace("${target_tokens}", str(self.condensed_target_tokens))

        try:
            response = self.llm.generate([{"role": "user", "content": prompt}])
            data = _parse_json_response(response)
            if "summary" in data and "key" in data:
                return data
            # Sometimes LLM puts summary in "content" or similar
            if "key" in data:
                data.setdefault("summary", data.get("content", children_text[:500]))
                return data
        except Exception:
            logger.exception("LLM summarisation failed")

        return None

    def _read_children(self, paths: list[str]) -> str:
        """Concatenate children content for the prompt."""
        parts: list[str] = []
        for path in paths:
            try:
                meta, body, _edges = read_md(path)
                key = meta.get("key", "")
                header = f"[{key}]" if key else ""
                parts.append(f"{header}\n{body}")
            except Exception:
                logger.warning("Could not read child: %s", path)
        return "\n\n---\n\n".join(parts)

    def _is_acceptable(self, summary: str) -> bool:
        """Check if the summary is within target token bounds."""
        estimated = len(summary) // 4
        # Allow 50% overshoot before escalating
        return estimated <= self.condensed_target_tokens * 1.5

    def _truncate(self, text: str) -> str:
        """Deterministic truncation fallback."""
        max_chars = self.condensed_target_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
