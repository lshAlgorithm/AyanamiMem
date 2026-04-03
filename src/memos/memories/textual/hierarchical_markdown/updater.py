"""Async stale propagation and re-summarisation.

When a leaf or summary changes, ancestor ``_summary.md`` files become stale.
The updater walks upward marking ancestors, then re-summarises bottom-up so
the tree stays consistent.

The updater provides methods for the main ``HierarchicalMarkdownMemory`` class
to call — it does **not** manage its own background thread.
"""

import os

from collections import deque
from typing import Any

from memos.log import get_logger
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingCache
from memos.memories.textual.hierarchical_markdown.fs import (
    SUMMARY_FILENAME,
    list_children,
    read_md,
    write_summary,
)
from memos.memories.textual.hierarchical_markdown.summarizer import HierarchicalSummarizer


logger = get_logger(__name__)


class Updater:
    """Stale propagation and re-summarisation engine.

    Parameters:
        memory_dir: Root directory of the memory tree.
        summarizer: The :class:`HierarchicalSummarizer` for generating summaries.
        embedder: Embedder instance (must expose ``embed(texts) -> list[list[float]]``).
    """

    def __init__(
        self,
        memory_dir: str,
        summarizer: HierarchicalSummarizer,
        embedder: Any,
        emb_cache: EmbeddingCache | None = None,
    ) -> None:
        self.memory_dir = memory_dir
        self.summarizer = summarizer
        self.embedder = embedder
        self._emb_cache = emb_cache
        self._stale_queue: deque[str] = deque()

    # ── Public API ────────────────────────────────────────────────────────

    def mark_stale(self, changed_path: str) -> None:
        """Walk upward from *changed_path*, marking ancestor summaries stale.

        For each ancestor directory between *changed_path* and ``memory_dir``,
        if a ``_summary.md`` exists, its ``stale`` frontmatter field is set to
        ``true`` and the directory is enqueued for re-summarisation.

        Args:
            changed_path: Absolute path to the file that was created/modified.
        """
        # Determine the directory of the changed file
        current = os.path.dirname(changed_path) if os.path.isfile(changed_path) else changed_path

        abs_memory = os.path.abspath(self.memory_dir)

        while True:
            abs_current = os.path.abspath(current)

            # Stop if we've gone above the memory root
            if not abs_current.startswith(abs_memory):
                break

            summary_path = os.path.join(current, SUMMARY_FILENAME)
            if os.path.isfile(summary_path):
                self._set_stale_flag(summary_path, stale=True)
                if current not in self._stale_queue:
                    self._stale_queue.append(current)
                    logger.debug("Marked stale: %s", summary_path)

            # Move up one level
            parent = os.path.dirname(current)
            if parent == current:
                break  # filesystem root
            current = parent

    def process_stale(self) -> int:
        """Process one stale node: re-read children, re-summarise, update.

        Returns the number of nodes processed (0 or 1).

        If the processed node's parent is also stale, it will be processed
        on the next call to ``process_stale`` (already in the queue from
        ``mark_stale``).
        """
        if not self._stale_queue:
            return 0

        dir_path = self._stale_queue.popleft()
        summary_path = os.path.join(dir_path, SUMMARY_FILENAME)

        if not os.path.isfile(summary_path):
            logger.warning("Stale node has no _summary.md: %s", dir_path)
            return 0

        # Read current summary for context
        old_meta, old_body, old_edges = read_md(summary_path)
        if not old_meta.get("stale", False):
            logger.debug("Node no longer stale, skipping: %s", dir_path)
            return 0

        depth = old_meta.get("depth", 1)

        # Collect children paths
        children_paths = self._collect_children_paths(dir_path)
        if not children_paths:
            logger.warning("No children found for stale node: %s", dir_path)
            self._set_stale_flag(summary_path, stale=False)
            return 1

        # Re-summarise
        try:
            new_body, new_key, new_tags = self.summarizer.summarize(
                children_paths, depth=depth, previous_summary=old_body
            )
        except Exception:
            logger.exception("Re-summarisation failed for %s", dir_path)
            return 0

        # Rebuild children links
        children_links = list_children(dir_path)

        # Compute updated metadata
        total_descendants = self._count_all_descendants(dir_path)
        new_meta: dict[str, Any] = {
            **old_meta,
            "key": new_key,
            "stale": False,
            "token_count": len(new_body) // 4,
            "descendant_count": total_descendants,
            "tags": new_tags,
        }

        # Preserve non-Children edges from the old summary
        extra_edges: dict[str, list[tuple[str, str]]] = {
            k: v for k, v in old_edges.items() if k != "Children"
        }

        write_summary(dir_path, new_body, new_meta, children_links, extra_edges or None)

        # Update in-memory cache with the new summary key vector
        if self._emb_cache is not None:
            try:
                vec = self.embedder.embed([new_key])[0]
                self._emb_cache.update({dir_path: vec})
            except Exception:
                logger.warning("Failed to update embedding for %s", dir_path)

        logger.info("Re-summarised stale node: %s", dir_path)
        return 1

    def run_update_loop(self) -> int:
        """Process all stale nodes, lowest depth first.

        Re-sorts the queue by depth (ascending) so that leaf-level summaries
        are updated before their parents, ensuring consistent roll-up.

        Returns the total number of nodes processed.
        """
        if not self._stale_queue:
            return 0

        # Sort by depth: process deepest (most specific) nodes first
        # so parent re-summaries see updated children
        sorted_dirs = self._sort_by_depth(list(self._stale_queue))
        self._stale_queue = deque(sorted_dirs)

        total_processed = 0
        max_iterations = len(self._stale_queue) * 2  # Safety bound

        iterations = 0
        while self._stale_queue and iterations < max_iterations:
            processed = self.process_stale()
            total_processed += processed
            iterations += 1

        if iterations >= max_iterations:
            logger.warning(
                "Update loop hit iteration limit (%d), %d items remain in queue",
                max_iterations,
                len(self._stale_queue),
            )

        logger.info("Update loop processed %d stale nodes", total_processed)
        return total_processed

    @property
    def pending_count(self) -> int:
        """Number of stale nodes waiting to be processed."""
        return len(self._stale_queue)

    # ── Private helpers ───────────────────────────────────────────────────

    def _set_stale_flag(self, summary_path: str, *, stale: bool) -> None:
        """Rewrite a ``_summary.md`` file with the ``stale`` flag set/cleared."""
        try:
            meta, body, edges = read_md(summary_path)
        except Exception:
            logger.warning("Could not read summary for stale update: %s", summary_path)
            return

        meta["stale"] = stale
        dir_path = os.path.dirname(summary_path)

        # Reconstruct children links from edges
        children_links = edges.get("Children", [])
        extra_edges = {k: v for k, v in edges.items() if k != "Children"}

        write_summary(dir_path, body, meta, children_links, extra_edges or None)

    def _collect_children_paths(self, dir_path: str) -> list[str]:
        """Collect absolute paths to all child ``.md`` files in *dir_path*.

        Returns paths to:
        - Subdirectory ``_summary.md`` files
        - Leaf ``.md`` files directly in the directory
        """
        paths: list[str] = []
        if not os.path.isdir(dir_path):
            return paths

        for entry in sorted(os.listdir(dir_path)):
            if entry.startswith("_"):
                continue
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                summary = os.path.join(full, SUMMARY_FILENAME)
                if os.path.isfile(summary):
                    paths.append(summary)
            elif entry.endswith(".md"):
                paths.append(full)

        return paths

    def _count_all_descendants(self, dir_path: str) -> int:
        """Recursively count all leaf ``.md`` files under *dir_path*."""
        count = 0
        if not os.path.isdir(dir_path):
            return 0
        for entry in os.listdir(dir_path):
            if entry.startswith("_"):
                continue
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                count += self._count_all_descendants(full)
            elif entry.endswith(".md"):
                count += 1
        return count

    def _sort_by_depth(self, dirs: list[str]) -> list[str]:
        """Sort directories by their depth in the tree (deepest first).

        Deeper directories (more specific summaries) should be updated before
        shallower ones so that parent re-summaries see fresh content.
        """
        abs_root = os.path.abspath(self.memory_dir)

        def _depth(d: str) -> int:
            rel = os.path.relpath(os.path.abspath(d), abs_root)
            if rel == ".":
                return 0
            return rel.count(os.sep) + 1

        return sorted(dirs, key=_depth, reverse=True)
