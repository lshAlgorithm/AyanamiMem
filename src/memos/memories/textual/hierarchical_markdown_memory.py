"""Hierarchical Markdown Memory — main orchestrator class.

Implements :class:`BaseTextMemory` by wiring together the chunker,
compactor, summarizer, assembler, and updater components.  All state
lives on the filesystem as ``.md`` files.
"""

import os
import shutil
import threading
import uuid

from typing import Any

from memos.configs.memory import HierarchicalMarkdownMemoryConfig
from memos.embedders.factory import EmbedderFactory
from memos.llms.factory import LLMFactory
from memos.log import get_logger
from memos.memories.textual.base import BaseTextMemory
from memos.memories.textual.hierarchical_markdown.assembler import Assembler
from memos.memories.textual.hierarchical_markdown.chunker import ChatChunker
from memos.memories.textual.hierarchical_markdown.compactor import Compactor
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingCache
from memos.memories.textual.hierarchical_markdown.fs import (
    FRESH_DIR,
    SUMMARY_FILENAME,
    read_md,
    walk_tree,
    write_md,
)
from memos.memories.textual.hierarchical_markdown.summarizer import HierarchicalSummarizer
from memos.memories.textual.hierarchical_markdown.updater import Updater
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
from memos.types import MessageList


logger = get_logger(__name__)


def _path_to_uuid(path: str) -> str:
    """Deterministic UUID from a filesystem path (for stable IDs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


def _meta_to_textual(meta: dict[str, Any], body: str, path: str) -> TextualMemoryItem:
    """Convert parsed frontmatter + body into a TextualMemoryItem."""
    return TextualMemoryItem(
        id=_path_to_uuid(path),
        memory=body,
        metadata=TextualMemoryMetadata(
            key=meta.get("key", ""),
            source=meta.get("source", "conversation"),
            tags=meta.get("tags", []),
        ),
    )


class HierarchicalMarkdownMemory(BaseTextMemory):
    """Hierarchical summary-tree memory stored as a directory of Markdown files.

    The filesystem mirrors the tree::

        memory_dir/
        ├── _fresh/           ← uncompacted conversation leaves
        ├── 01-topic-a/
        │   ├── _summary.md   ← condensed node
        │   ├── 001-leaf.md
        │   └── …
        └── _root.md          ← top-level summary

    Search walks the tree top-down using ``_embeddings.json`` at each level.
    The LLM can drill deeper by reading child files via relative Markdown links.
    """

    mode: str = "sync"

    def __init__(self, config: HierarchicalMarkdownMemoryConfig) -> None:
        self.config = config
        self.memory_dir = os.path.abspath(config.memory_dir)

        self.llm = LLMFactory.from_config(config.extractor_llm)
        self.embedder = EmbedderFactory.from_config(config.embedder)

        # Single in-memory embedding cache shared across all components
        self._emb_cache = EmbeddingCache()

        self.summarizer = HierarchicalSummarizer(
            self.llm, self.embedder, config.condensed_target_tokens
        )
        self.chunker = ChatChunker(
            self.memory_dir,
            self.embedder,
            config.leaf_chunk_tokens,
            emb_cache=self._emb_cache,
        )
        self.compactor = Compactor(
            memory_dir=self.memory_dir,
            summarizer=self.summarizer,
            embedder=self.embedder,
            fresh_tail_count=config.fresh_tail_count,
            condensed_min_fanout=config.condensed_min_fanout,
            compact_threshold=config.compact_threshold,
            similarity_threshold=config.compact_similarity_threshold,
            emb_cache=self._emb_cache,
        )
        self.assembler = Assembler(self.memory_dir, self.embedder, emb_cache=self._emb_cache)
        self.updater = Updater(
            self.memory_dir, self.summarizer, self.embedder, emb_cache=self._emb_cache
        )

        # Path→UUID cache for fast lookups
        self._path_cache: dict[str, str] = {}
        self._uuid_cache: dict[str, str] = {}
        self._compact_lock = threading.Lock()

        # Ensure memory directory structure exists
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(os.path.join(self.memory_dir, FRESH_DIR), exist_ok=True)

    # ── BaseTextMemory interface ──────────────────────────────────────────

    def extract(self, messages: MessageList) -> list[TextualMemoryItem]:
        """Chunk messages into leaf files in ``_fresh/`` and return them as items."""
        filenames = self.chunker.chunk(messages)
        items: list[TextualMemoryItem] = []
        fresh_dir = os.path.join(self.memory_dir, FRESH_DIR)
        for fname in filenames:
            fpath = os.path.join(fresh_dir, fname)
            try:
                meta, body, _edges = read_md(fpath)
                item = _meta_to_textual(meta, body, fpath)
                self._register_path(fpath, item.id)
                items.append(item)
            except Exception:
                logger.exception("Failed to read chunked leaf: %s", fpath)
        return items

    def add(self, memories: list[TextualMemoryItem | dict[str, Any]], **kwargs: Any) -> list[str]:
        """Add pre-chunked items or raw dicts as leaves in ``_fresh/``.

        If items already came from :meth:`extract`, they are already on disk
        and this is a no-op for those.  New dicts are written as fresh leaves.

        After adding, triggers async compaction.
        """
        fresh_dir = os.path.join(self.memory_dir, FRESH_DIR)
        new_ids: list[str] = []

        for mem in memories:
            if isinstance(mem, dict):
                mem = TextualMemoryItem(**mem)

            # Check if already on disk (from extract)
            if mem.id in self._uuid_cache:
                new_ids.append(mem.id)
                continue

            # Write new leaf
            meta = {
                "key": mem.metadata.key or mem.memory[:60],
                "depth": 0,
                "node_kind": "leaf",
                "source": mem.metadata.source or "manual",
                "tags": mem.metadata.tags or [],
                "stale": False,
                "descendant_count": 1,
            }
            from memos.memories.textual.hierarchical_markdown.fs import write_leaf

            fname = write_leaf(fresh_dir, mem.memory, meta)
            fpath = os.path.join(fresh_dir, fname)
            item_id = _path_to_uuid(fpath)
            self._register_path(fpath, item_id)
            new_ids.append(item_id)

            # Update in-memory embedding cache
            try:
                vec = self.embedder.embed([meta["key"]])[0]
                self._emb_cache.update({fpath: vec})
            except Exception:
                logger.warning("Failed to embed new leaf: %s", fname)

        # Trigger compaction in background
        self._maybe_compact()

        return new_ids

    def update(self, memory_id: str, new_memory: TextualMemoryItem | dict[str, Any]) -> None:
        """Update a leaf file in place, then mark ancestors stale."""
        path = self._uuid_cache.get(memory_id)
        if not path or not os.path.isfile(path):
            logger.warning("Cannot update: path not found for id=%s", memory_id)
            return

        if isinstance(new_memory, dict):
            new_memory = TextualMemoryItem(**new_memory)

        # Read existing metadata, update body
        meta, _old_body, edges = read_md(path)
        meta["key"] = new_memory.metadata.key or meta.get("key", "")
        meta["tags"] = new_memory.metadata.tags or meta.get("tags", [])

        write_md(path, new_memory.memory, meta, edges if edges else None)

        # Mark ancestors stale
        self.updater.mark_stale(path)

        # Process stale nodes
        self.updater.process_stale()

    def search(
        self, query: str, top_k: int, info: Any = None, **kwargs: Any
    ) -> list[TextualMemoryItem]:
        """Search via top-down tree walk + fresh tail."""
        results = self.assembler.assemble(
            query=query,
            budget_tokens=top_k * 500,  # rough budget
            fresh_tail_count=self.config.fresh_tail_count,
        )

        items: list[TextualMemoryItem] = []
        for r in results[:top_k]:
            meta = r.get("meta", {})
            body = r.get("content", "")
            path = r.get("path", "")
            item = _meta_to_textual(meta, body, path)
            items.append(item)

        return items

    def get(self, memory_id: str, user_name: str | None = None) -> TextualMemoryItem:
        """Get a memory by its UUID (derived from filesystem path)."""
        path = self._uuid_cache.get(memory_id)
        if not path or not os.path.isfile(path):
            return TextualMemoryItem(id=memory_id, memory="", metadata=TextualMemoryMetadata())
        meta, body, _edges = read_md(path)
        return _meta_to_textual(meta, body, path)

    def get_by_ids(
        self, memory_ids: list[str], user_name: str | None = None
    ) -> list[TextualMemoryItem]:
        return [self.get(mid) for mid in memory_ids]

    def get_all(self) -> list[TextualMemoryItem]:
        """Walk the entire filesystem tree and return all nodes."""
        self._rebuild_cache()
        items: list[TextualMemoryItem] = []
        for path in walk_tree(self.memory_dir):
            try:
                meta, body, _edges = read_md(path)
                items.append(_meta_to_textual(meta, body, path))
            except Exception:
                logger.warning("Failed to read: %s", path)
        # Also include _fresh/ leaves
        fresh_dir = os.path.join(self.memory_dir, FRESH_DIR)
        if os.path.isdir(fresh_dir):
            for fname in sorted(os.listdir(fresh_dir)):
                if fname.endswith(".md") and not fname.startswith("_"):
                    fpath = os.path.join(fresh_dir, fname)
                    try:
                        meta, body, _edges = read_md(fpath)
                        items.append(_meta_to_textual(meta, body, fpath))
                    except Exception:
                        logger.warning("Failed to read fresh leaf: %s", fpath)
        return items

    def delete(self, memory_ids: list[str]) -> None:
        """Delete memory files and mark parent summaries stale."""
        for mid in memory_ids:
            path = self._uuid_cache.get(mid)
            if not path or not os.path.isfile(path):
                continue

            parent_dir = os.path.dirname(path)

            # Remove the file
            os.remove(path)
            self._unregister_path(path)

            # Mark parent summary stale
            summary_path = os.path.join(parent_dir, SUMMARY_FILENAME)
            if os.path.isfile(summary_path):
                self.updater.mark_stale(path)

            # If directory is now empty (no .md files except _summary.md), remove it
            remaining = [
                f for f in os.listdir(parent_dir) if f.endswith(".md") and not f.startswith("_")
            ]
            if not remaining and parent_dir != self.memory_dir:
                shutil.rmtree(parent_dir, ignore_errors=True)

    def delete_all(self) -> None:
        """Remove all contents of the memory directory."""
        if os.path.isdir(self.memory_dir):
            for entry in os.listdir(self.memory_dir):
                full = os.path.join(self.memory_dir, entry)
                if os.path.isdir(full):
                    shutil.rmtree(full)
                elif os.path.isfile(full):
                    os.remove(full)
        self._path_cache.clear()
        self._uuid_cache.clear()
        # Re-create _fresh/
        os.makedirs(os.path.join(self.memory_dir, FRESH_DIR), exist_ok=True)

    def drop(self) -> None:
        """No-op (no external databases to drop)."""

    # ── Additional public API ─────────────────────────────────────────────

    def dump(self, dir: str) -> None:
        """Flush pending state to *dir*.

        Since files are written immediately on every mutation, this is
        mostly a no-op.  If *dir* differs from ``memory_dir``, the tree
        is copied there.
        """
        target = os.path.abspath(dir)
        if target != self.memory_dir and os.path.isdir(self.memory_dir):
            if os.path.isdir(target):
                shutil.rmtree(target)
            shutil.copytree(self.memory_dir, target)

    def load(self, dir: str) -> None:  # noqa: A002
        """Load memory tree from *dir* (set as the new memory_dir)."""
        dir_path = dir
        self.memory_dir = os.path.abspath(dir_path)
        self.chunker = ChatChunker(
            self.memory_dir,
            self.embedder,
            self.config.leaf_chunk_tokens,
            emb_cache=self._emb_cache,
        )
        self.compactor = Compactor(
            memory_dir=self.memory_dir,
            summarizer=self.summarizer,
            embedder=self.embedder,
            fresh_tail_count=self.config.fresh_tail_count,
            condensed_min_fanout=self.config.condensed_min_fanout,
            compact_threshold=self.config.compact_threshold,
            similarity_threshold=self.config.compact_similarity_threshold,
            emb_cache=self._emb_cache,
        )
        self.assembler = Assembler(self.memory_dir, self.embedder, emb_cache=self._emb_cache)
        self.updater = Updater(
            self.memory_dir, self.summarizer, self.embedder, emb_cache=self._emb_cache
        )
        self._rebuild_cache()
        logger.info("Loaded memory tree from %s", dir_path)

    def compact(self, force: bool = False) -> dict[str, int]:
        """Trigger compaction and rebuild the path cache.

        Args:
            force: Bypass ``should_compact()`` threshold and always run
                   ``leaf_pass``.  The lock, full depth cascade, and
                   exception handling inside ``compact_incremental`` are
                   always preserved.

        Returns:
            ``{"leaves_compacted": N, "dirs_condensed": M}``
        """
        result = self.compactor.compact_incremental(force=force)
        self._rebuild_cache()
        return result

    def get_tree_index(self) -> str:
        """Return a human-readable tree index (like ``tree`` command output)."""
        lines: list[str] = ["# Memory Tree"]
        self._render_tree(self.memory_dir, lines, indent=0)
        # Also show _fresh/
        fresh_dir = os.path.join(self.memory_dir, FRESH_DIR)
        if os.path.isdir(fresh_dir):
            fresh_files = [
                f
                for f in sorted(os.listdir(fresh_dir))
                if f.endswith(".md") and not f.startswith("_")
            ]
            if fresh_files:
                lines.append(f"\n## Fresh ({len(fresh_files)} uncompacted)")
                for f in fresh_files:
                    lines.append(f"  - {f}")
        return "\n".join(lines) + "\n"

    # ── Private helpers ───────────────────────────────────────────────────

    def _maybe_compact(self) -> None:
        """Trigger compaction in a background thread if threshold met."""
        if not self.compactor.should_compact():
            return

        def _do_compact() -> None:
            with self._compact_lock:
                try:
                    self.compactor.compact_incremental()
                except Exception:
                    logger.exception("Compaction failed")

        thread = threading.Thread(target=_do_compact, daemon=True)
        thread.start()

    def _register_path(self, path: str, item_id: str) -> None:
        self._path_cache[path] = item_id
        self._uuid_cache[item_id] = path

    def _unregister_path(self, path: str) -> None:
        item_id = self._path_cache.pop(path, None)
        if item_id:
            self._uuid_cache.pop(item_id, None)

    def _rebuild_cache(self) -> None:
        """Rebuild path↔UUID cache + embedding cache from the filesystem.

        Also removes any stale ``_embeddings.json`` files found on disk —
        embeddings are now held exclusively in memory.
        """
        self._path_cache.clear()
        self._uuid_cache.clear()
        self._emb_cache.clear()

        # Collect all .md paths, distinguishing summaries from leaves
        all_paths: list[str] = walk_tree(self.memory_dir)
        fresh_dir = os.path.join(self.memory_dir, FRESH_DIR)
        if os.path.isdir(fresh_dir):
            for fname in sorted(os.listdir(fresh_dir)):
                if fname.endswith(".md") and not fname.startswith("_"):
                    all_paths.append(os.path.join(fresh_dir, fname))

        # Register paths in UUID cache
        for path in all_paths:
            item_id = _path_to_uuid(path)
            self._register_path(path, item_id)

        # Batch-embed all keys from frontmatter
        cache_keys: list[str] = []  # embedding cache key (dir or leaf path)
        embed_texts: list[str] = []  # key phrase to embed

        for path in all_paths:
            try:
                meta, _, _ = read_md(path)
                key = meta.get("key", "")
                if not key:
                    continue
                # For _summary.md → cache key is the parent directory
                if os.path.basename(path) == SUMMARY_FILENAME:
                    cache_key = os.path.dirname(path)
                else:
                    cache_key = path
                cache_keys.append(cache_key)
                embed_texts.append(key)
            except Exception:
                logger.warning("Could not read key from %s", path)

        if embed_texts:
            try:
                vecs = self.embedder.embed(embed_texts)
                self._emb_cache.update({k: v for k, v in zip(cache_keys, vecs)})
            except Exception:
                logger.warning("Batch embed failed during _rebuild_cache; cache empty")

        # Delete stale _embeddings.json files from disk
        self._cleanup_json_index_files(self.memory_dir)
        logger.info(
            "Cache rebuilt: %d paths, %d embeddings",
            len(self._path_cache),
            len(self._emb_cache),
        )

    @staticmethod
    def _cleanup_json_index_files(root_dir: str) -> None:
        """Remove ``_embeddings.json`` and ``_embeddings.json.lock`` files."""
        for dirpath, _dirs, files in os.walk(root_dir):
            for fname in files:
                if fname in ("_embeddings.json", "_embeddings.json.lock", "_embeddings.json.tmp"):
                    try:
                        os.remove(os.path.join(dirpath, fname))
                    except OSError:
                        pass

    def _render_tree(self, dir_path: str, lines: list[str], indent: int) -> None:
        """Recursively render the directory tree."""
        entries = sorted(os.listdir(dir_path)) if os.path.isdir(dir_path) else []
        prefix = "  " * indent

        for entry in entries:
            if entry.startswith("_"):
                continue
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                summary = os.path.join(full, SUMMARY_FILENAME)
                if os.path.isfile(summary):
                    meta, _body, _edges = read_md(summary)
                    key = meta.get("key", entry)
                    lines.append(f"{prefix}- **{key}** (`{entry}/`)")
                    self._render_tree(full, lines, indent + 1)
                else:
                    lines.append(f"{prefix}- {entry}/")
            elif entry.endswith(".md"):
                try:
                    meta, _body, _edges = read_md(full)
                    key = meta.get("key", entry)
                    lines.append(f"{prefix}- {key} (`{entry}`)")
                except Exception:
                    lines.append(f"{prefix}- {entry}")
