"""Chat messages → leaf ``.md`` files in ``_fresh/``.

The chunker splits a ``MessageList`` into token-bounded chunks and writes
each chunk as a leaf file in the ``_fresh/`` staging directory.  The
compactor later groups these, generates an LLM summary for each cluster,
and renames the leaf files using the summary key.

The chunker itself makes **no LLM calls** — it uses a fast heuristic to
generate an initial filename.  The leaf's final human-readable filename
is set by the compactor once the cluster's topic is known.
"""

import os

from datetime import datetime
from typing import Any

from memos.log import get_logger
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingCache
from memos.memories.textual.hierarchical_markdown.fs import (
    FRESH_DIR,
    leaf_filename,
    next_seq,
    write_md,
)
from memos.types import MessageList


logger = get_logger(__name__)

_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _format_message(msg: dict[str, Any]) -> str:
    """Format a single message as ``Role: content``."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts)
    label = {"user": "User", "assistant": "Assistant", "system": "System", "tool": "Tool"}.get(
        role, role.capitalize()
    )
    return f"{label}: {content}"


class ChatChunker:
    """Split conversation messages into token-bounded leaf chunks.

    Leaves are written with a fast heuristic filename (first sentence of the
    conversation).  Once the compactor groups leaves into a cluster and calls
    the LLM, it renames each leaf using the LLM-generated cluster key and
    updates the ``key`` / ``tags`` frontmatter fields in place.
    """

    def __init__(
        self,
        memory_dir: str,
        embedder: Any,
        leaf_chunk_tokens: int = 2048,
        emb_cache: EmbeddingCache | None = None,
    ) -> None:
        self.memory_dir = memory_dir
        self.embedder = embedder
        self.leaf_chunk_tokens = leaf_chunk_tokens
        self._emb_cache = emb_cache
        self._fresh_dir = os.path.join(memory_dir, FRESH_DIR)

    def chunk(
        self,
        messages: MessageList,
        session_id: str = "default",
    ) -> list[str]:
        """Split *messages* into leaf ``.md`` files in ``_fresh/``.

        Returns list of written filenames (relative to ``_fresh/``).
        The filename is a heuristic placeholder; the compactor renames
        leaves to the LLM-generated cluster key after grouping.
        """
        if not messages:
            return []

        os.makedirs(self._fresh_dir, exist_ok=True)
        chunks = self._split_messages(messages)

        written: list[str] = []
        emb_updates: dict[str, list[float]] = {}

        for chunk_text, earliest, latest in chunks:
            token_count = _estimate_tokens(chunk_text)
            # Fast heuristic key — no LLM call.
            # Compactor will replace this with the LLM cluster key.
            key = self._extract_key(chunk_text)

            meta: dict[str, Any] = {
                "key": key,
                "depth": 0,
                "node_kind": "leaf",
                "session_id": session_id,
                "earliest_at": earliest,
                "latest_at": latest,
                "token_count": token_count,
                "descendant_count": 1,
                "stale": False,
                "source": "conversation",
                "tags": [],
            }

            seq = next_seq(self._fresh_dir)
            fname = leaf_filename(seq, key)
            fpath = os.path.join(self._fresh_dir, fname)
            write_md(fpath, chunk_text, meta)
            written.append(fname)

            try:
                vec = self.embedder.embed([key])[0]
                emb_updates[fname] = vec
            except Exception:
                logger.warning("Failed to embed leaf key: %s", key)

        if emb_updates and self._emb_cache is not None:
            full_path_updates = {
                os.path.join(self._fresh_dir, fname): vec for fname, vec in emb_updates.items()
            }
            self._emb_cache.update(full_path_updates)

        logger.info("Chunked %d messages into %d leaves in _fresh/", len(messages), len(written))
        return written

    def _split_messages(self, messages: MessageList) -> list[tuple[str, str, str]]:
        """Split messages into token-bounded ``(text, earliest, latest)`` tuples."""
        now = datetime.now().isoformat()
        chunks: list[tuple[str, str, str]] = []
        current_lines: list[str] = []
        current_tokens = 0
        earliest = now
        latest = now

        for msg in messages:
            line = _format_message(msg)
            line_tokens = _estimate_tokens(line)

            if current_tokens + line_tokens > self.leaf_chunk_tokens and current_lines:
                chunks.append(("\n\n".join(current_lines), earliest, latest))
                current_lines = []
                current_tokens = 0
                earliest = now

            current_lines.append(line)
            current_tokens += line_tokens
            latest = now

        if current_lines:
            chunks.append(("\n\n".join(current_lines), earliest, latest))

        return chunks

    @staticmethod
    def _extract_key(chunk_text: str) -> str:
        """Fast heuristic: first meaningful sentence, role prefix stripped."""
        for line in chunk_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ": " in line:
                _, _, content = line.partition(": ")
                line = content
            if len(line) > 80:
                line = line[:77] + "..."
            return line
        return "conversation chunk"
