"""Chat messages → leaf ``.md`` files in ``_fresh/``.

The chunker splits a ``MessageList`` into token-bounded chunks and writes
each chunk as a leaf file in the ``_fresh/`` staging directory.  The
compactor later groups and moves these into the tree.
"""

import os

from datetime import datetime
from typing import Any

from memos.log import get_logger
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingIndex
from memos.memories.textual.hierarchical_markdown.fs import (
    FRESH_DIR,
    leaf_filename,
    next_seq,
    write_md,
)
from memos.types import MessageList


logger = get_logger(__name__)

# Rough token estimate: 1 token ≈ 4 chars (conservative for English)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _format_message(msg: dict[str, Any]) -> str:
    """Format a single message as ``Role: content``."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    if isinstance(content, list):
        # Handle multi-part content (text blocks)
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
    """Split conversation messages into token-bounded leaf chunks."""

    def __init__(
        self,
        memory_dir: str,
        embedder: Any,
        leaf_chunk_tokens: int = 2048,
    ) -> None:
        self.memory_dir = memory_dir
        self.embedder = embedder
        self.leaf_chunk_tokens = leaf_chunk_tokens
        self._fresh_dir = os.path.join(memory_dir, FRESH_DIR)

    def chunk(
        self,
        messages: MessageList,
        session_id: str = "default",
    ) -> list[str]:
        """Split *messages* into leaf ``.md`` files in ``_fresh/``.

        Returns list of written filenames (relative to ``_fresh/``).
        """
        if not messages:
            return []

        os.makedirs(self._fresh_dir, exist_ok=True)
        chunks = self._split_messages(messages)

        written: list[str] = []
        emb_updates: dict[str, list[float]] = {}

        for chunk_text, earliest, latest in chunks:
            token_count = _estimate_tokens(chunk_text)
            # Generate key from first line or first 80 chars
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
            }

            seq = next_seq(self._fresh_dir)
            fname = leaf_filename(seq, key)
            fpath = os.path.join(self._fresh_dir, fname)
            write_md(fpath, chunk_text, meta)
            written.append(fname)

            # Embed the key for search
            try:
                vec = self.embedder.embed([key])[0]
                emb_updates[fname] = vec
            except Exception:
                logger.warning("Failed to embed leaf key: %s", key)

        # Batch-update embeddings index
        if emb_updates:
            idx = EmbeddingIndex(self._fresh_dir)
            idx.update(emb_updates)

        logger.info("Chunked %d messages into %d leaves in _fresh/", len(messages), len(written))
        return written

    def _split_messages(self, messages: MessageList) -> list[tuple[str, str, str]]:
        """Split messages into token-bounded chunks.

        Returns list of ``(chunk_text, earliest_timestamp, latest_timestamp)``.
        """
        now = datetime.now().isoformat()
        chunks: list[tuple[str, str, str]] = []
        current_lines: list[str] = []
        current_tokens = 0
        earliest = now
        latest = now

        for msg in messages:
            line = _format_message(msg)
            line_tokens = _estimate_tokens(line)

            # If adding this message would exceed the limit, flush current chunk
            if current_tokens + line_tokens > self.leaf_chunk_tokens and current_lines:
                chunk_text = "\n\n".join(current_lines)
                chunks.append((chunk_text, earliest, latest))
                current_lines = []
                current_tokens = 0
                earliest = now

            current_lines.append(line)
            current_tokens += line_tokens
            latest = now

        # Flush remaining
        if current_lines:
            chunk_text = "\n\n".join(current_lines)
            chunks.append((chunk_text, earliest, latest))

        return chunks

    @staticmethod
    def _extract_key(chunk_text: str) -> str:
        """Extract a short key from the chunk content.

        Uses the first meaningful sentence or first 80 chars.
        """
        # Take first non-empty line, strip the role prefix
        for line in chunk_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip "User: " / "Assistant: " prefix
            if ": " in line:
                _, _, content = line.partition(": ")
                line = content
            # Truncate to reasonable length
            if len(line) > 80:
                line = line[:77] + "..."
            return line
        return "conversation chunk"
