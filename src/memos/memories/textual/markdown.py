"""Markdown file-based textual memory backend.

Each memory node is a human-readable ``.md`` file::

    ---
    id: <uuid>
    key: "short descriptive key"
    status: activated
    tags: [tag1, tag2]
    ...
    ---

    The actual memory content (value).

    ## Edges
    - RELATED: <target-uuid>
    - PARENT: <target-uuid>

Retrieval is local-only: embed the ``key`` fields with a
``sentence-transformers`` model and rank by cosine similarity.
"""

import json
import os
import re
import uuid

from datetime import datetime
from typing import Any

import numpy as np
import yaml

from tenacity import retry, retry_if_exception_type, stop_after_attempt

from memos.configs.memory import MarkdownTextMemoryConfig
from memos.embedders.factory import EmbedderFactory
from memos.llms.factory import LLMFactory
from memos.log import get_logger
from memos.memories.textual.base import BaseTextMemory
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
from memos.templates.mem_reader_prompts import SIMPLE_STRUCT_MEM_READER_PROMPT
from memos.types import MessageList


logger = get_logger(__name__)

# ── Frontmatter fields that map to TextualMemoryMetadata ──────────────────────
_META_FIELDS = set(TextualMemoryMetadata.model_fields)

# Regex for the YAML frontmatter block
_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)

# Heading that separates body from the edge list
_EDGES_HEADING = "## Edges"

# Pattern for a single edge line: ``- TYPE: uuid``
_EDGE_LINE_RE = re.compile(r"^-\s+(\w+):\s+([0-9a-f-]{36})\s*$")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _serialize_md(item: TextualMemoryItem, edges: list[dict[str, str]] | None = None) -> str:
    """Render a ``TextualMemoryItem`` + optional edges as a Markdown string."""
    meta = item.metadata.model_dump(exclude_none=True)
    meta["id"] = item.id
    # Ensure ``key`` is always present (use first 80 chars of memory as fallback)
    if not meta.get("key"):
        meta["key"] = item.memory[:80].replace("\n", " ")

    frontmatter = yaml.safe_dump(
        meta, default_flow_style=False, allow_unicode=True, sort_keys=False
    ).rstrip("\n")

    parts = [f"---\n{frontmatter}\n---\n", item.memory]

    if edges:
        lines = [f"- {e['type']}: {e['target']}" for e in edges]
        parts.append(f"\n\n{_EDGES_HEADING}\n" + "\n".join(lines))

    return "\n".join(parts) + "\n"


def _parse_md(text: str) -> tuple[TextualMemoryItem, list[dict[str, str]]]:
    """Parse a Markdown memory file into a ``TextualMemoryItem`` + edge list."""
    # 1. Extract frontmatter
    fm_match = _FRONTMATTER_RE.match(text)
    if not fm_match:
        raise ValueError("Missing YAML frontmatter in markdown memory file")
    raw_meta: dict[str, Any] = yaml.safe_load(fm_match.group(1)) or {}

    remainder = text[fm_match.end() :]

    # 2. Split body / edges
    edge_idx = remainder.find(_EDGES_HEADING)
    if edge_idx >= 0:
        body = remainder[:edge_idx].strip()
        edge_block = remainder[edge_idx + len(_EDGES_HEADING) :]
    else:
        body = remainder.strip()
        edge_block = ""

    # 3. Parse edges
    edges: list[dict[str, str]] = []
    for line in edge_block.strip().splitlines():
        m = _EDGE_LINE_RE.match(line.strip())
        if m:
            edges.append({"type": m.group(1), "target": m.group(2)})

    # 4. Build TextualMemoryItem
    node_id = str(raw_meta.pop("id", uuid.uuid4()))
    # Validate that the id is a proper UUID
    uuid.UUID(node_id)

    item = TextualMemoryItem(
        id=node_id,
        memory=body,
        metadata=TextualMemoryMetadata(**{k: v for k, v in raw_meta.items() if k in _META_FIELDS}),
    )
    return item, edges


# ── Main class ────────────────────────────────────────────────────────────────


class MarkdownTextMemory(BaseTextMemory):
    """Markdown file-backed textual memory with local embedding search.

    Storage layout::

        <memory_dir>/
            <uuid>.md
            <uuid>.md
            ...
            _embeddings.npy   (optional cache of key embeddings)
    """

    def __init__(self, config: MarkdownTextMemoryConfig) -> None:
        self.mode: str = getattr(self.__class__, "mode", "sync")
        self.config = config

        self.extractor_llm = LLMFactory.from_config(config.extractor_llm)
        self.embedder = EmbedderFactory.from_config(config.embedder)

        # In-memory index: id → (TextualMemoryItem, edges)
        self._index: dict[str, tuple[TextualMemoryItem, list[dict[str, str]]]] = {}
        # Parallel arrays kept in sync with _index for fast search
        self._ids: list[str] = []
        self._key_vecs: np.ndarray | None = None  # shape (N, dim)

    # ── Disk I/O ──────────────────────────────────────────────────────────────

    def load(self, dir: str) -> None:
        """Scan *dir* for ``*.md`` files and load them into memory."""
        if not os.path.isdir(dir):
            logger.warning("Markdown memory directory not found: %s", dir)
            return
        count = 0
        for fname in sorted(os.listdir(dir)):
            if not fname.endswith(".md") or fname.startswith("_"):
                continue
            fpath = os.path.join(dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    item, edges = _parse_md(f.read())
                self._index[item.id] = (item, edges)
                count += 1
            except Exception:
                logger.exception("Failed to parse markdown memory file: %s", fpath)
        logger.info("Loaded %d markdown memories from %s", count, dir)
        self._rebuild_embeddings()

    def dump(self, dir: str) -> None:
        """Write all memories to *dir* as individual ``.md`` files."""
        os.makedirs(dir, exist_ok=True)
        for node_id, (item, edges) in self._index.items():
            fpath = os.path.join(dir, f"{node_id}.md")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(_serialize_md(item, edges))
        logger.info("Dumped %d markdown memories to %s", len(self._index), dir)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, memories: list[TextualMemoryItem | dict[str, Any]], **kwargs: Any) -> list[str]:
        """Add memory items. Returns the list of new IDs."""
        items = [TextualMemoryItem(**m) if isinstance(m, dict) else m for m in memories]
        new_ids: list[str] = []
        for item in items:
            if item.id in self._index:
                logger.debug("Skipping duplicate memory id=%s", item.id)
                continue
            self._index[item.id] = (item, [])
            new_ids.append(item.id)
            # Write to disk immediately if memory_dir exists
            self._write_file(item.id)
        if new_ids:
            self._rebuild_embeddings()
        return new_ids

    def update(self, memory_id: str, new_memory: TextualMemoryItem | dict[str, Any]) -> None:
        item = TextualMemoryItem(**new_memory) if isinstance(new_memory, dict) else new_memory
        item.id = memory_id
        _, edges = self._index.get(memory_id, (None, []))
        self._index[memory_id] = (item, edges)
        self._write_file(memory_id)
        self._rebuild_embeddings()

    def get(self, memory_id: str, user_name: str | None = None) -> TextualMemoryItem:
        entry = self._index.get(memory_id)
        if entry is None:
            return TextualMemoryItem(id=memory_id, memory="", metadata=TextualMemoryMetadata())
        return entry[0]

    def get_by_ids(
        self, memory_ids: list[str], user_name: str | None = None
    ) -> list[TextualMemoryItem]:
        return [self.get(mid) for mid in memory_ids]

    def get_all(self) -> list[TextualMemoryItem]:
        return [item for item, _ in self._index.values()]

    def delete(self, memory_ids: list[str]) -> None:
        for mid in memory_ids:
            self._index.pop(mid, None)
            self._delete_file(mid)
        self._rebuild_embeddings()

    def delete_all(self) -> None:
        for mid in list(self._index):
            self._delete_file(mid)
        self._index.clear()
        self._ids.clear()
        self._key_vecs = None

    def drop(self) -> None:
        pass

    # ── Intelligence ──────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(json.JSONDecodeError),
        before_sleep=lambda retry_state: logger.warning(
            "Extract retry %d/3: %s",
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        ),
    )
    def extract(self, messages: MessageList) -> list[TextualMemoryItem]:
        """Use an LLM to extract structured memories from messages."""
        str_messages = "\n".join(f"{msg['role']}:{msg['content']}" for msg in messages)
        prompt = SIMPLE_STRUCT_MEM_READER_PROMPT.replace("${conversation}", str_messages).replace(
            "${custom_tags_prompt}", ""
        )

        response = self.extractor_llm.generate([{"role": "user", "content": prompt}])
        response_json = self._parse_json(response)

        return [
            TextualMemoryItem(
                memory=mem["value"],
                metadata=TextualMemoryMetadata(
                    key=mem["key"],
                    source="conversation",
                    tags=mem.get("tags", []),
                    updated_at=datetime.now().isoformat(),
                ),
            )
            for mem in response_json.get("memory list", [])
        ]

    def search(
        self, query: str, top_k: int, info: Any = None, **kwargs: Any
    ) -> list[TextualMemoryItem]:
        """Search by cosine similarity between *query* and stored ``key`` fields."""
        if not self._ids or self._key_vecs is None:
            return []
        query_vec = np.array(self.embedder.embed([query])[0], dtype=np.float32)
        # Cosine similarity
        norms = np.linalg.norm(self._key_vecs, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        scores = self._key_vecs @ query_vec / norms
        top_idx = np.argsort(-scores)[:top_k]
        return [self._index[self._ids[i]][0] for i in top_idx]

    # ── Edge helpers (not in base interface, bonus API) ────────────────────────

    def get_edges(self, memory_id: str) -> list[dict[str, str]]:
        """Return edges for a given memory node."""
        entry = self._index.get(memory_id)
        return entry[1] if entry else []

    def add_edge(self, source_id: str, target_id: str, edge_type: str = "RELATED") -> None:
        """Add a directional edge from *source_id* → *target_id*."""
        entry = self._index.get(source_id)
        if entry is None:
            raise ValueError(f"Source memory {source_id} not found")
        item, edges = entry
        # Avoid duplicates
        if not any(e["target"] == target_id and e["type"] == edge_type for e in edges):
            edges.append({"type": edge_type, "target": target_id})
            self._index[source_id] = (item, edges)
            self._write_file(source_id)

    def remove_edge(self, source_id: str, target_id: str, edge_type: str | None = None) -> None:
        """Remove edge(s) from *source_id* → *target_id*."""
        entry = self._index.get(source_id)
        if entry is None:
            return
        item, edges = entry
        new_edges = [
            e
            for e in edges
            if not (e["target"] == target_id and (edge_type is None or e["type"] == edge_type))
        ]
        self._index[source_id] = (item, new_edges)
        self._write_file(source_id)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _rebuild_embeddings(self) -> None:
        """Recompute the key-embedding index from the current in-memory state."""
        if not self._index:
            self._ids = []
            self._key_vecs = None
            return
        self._ids = list(self._index.keys())
        keys = [
            (self._index[mid][0].metadata.key or self._index[mid][0].memory[:80])
            for mid in self._ids
        ]
        vecs = self.embedder.embed(keys)
        self._key_vecs = np.array(vecs, dtype=np.float32)

    def _write_file(self, memory_id: str) -> None:
        """Persist a single memory to disk (if ``memory_dir`` is configured)."""
        memory_dir = self.config.memory_dir
        if not memory_dir:
            return
        os.makedirs(memory_dir, exist_ok=True)
        item, edges = self._index[memory_id]
        fpath = os.path.join(memory_dir, f"{memory_id}.md")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(_serialize_md(item, edges))

    def _delete_file(self, memory_id: str) -> None:
        """Remove a ``.md`` file from disk."""
        memory_dir = self.config.memory_dir
        if not memory_dir:
            return
        fpath = os.path.join(memory_dir, f"{memory_id}.md")
        if os.path.isfile(fpath):
            os.remove(fpath)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Best-effort JSON extraction from LLM output."""
        idx = text.find("{")
        if idx < 0:
            return {}
        text = text[idx:].replace("```", "").strip()
        if text and text[-1] != "}":
            text += "}"
        return json.loads(text)
