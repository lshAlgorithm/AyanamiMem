"""In-memory embedding cache: absolute path → 384-dim vector.

Replaces the old disk-based ``_embeddings.json`` approach entirely.
Vectors are derived from the ``key`` field of each ``.md`` file and
held in a single thread-safe dict owned by
:class:`~memos.memories.textual.hierarchical_markdown_memory.HierarchicalMarkdownMemory`.

No files are written.  On every :py:meth:`load` the cache is rebuilt
from the ``.md`` files on disk in a single batch embed call.
"""

import threading

from typing import Any

import numpy as np

from memos.log import get_logger


logger = get_logger(__name__)


class EmbeddingCache:
    """Thread-safe in-memory map: ``absolute_path → float32 vector``.

    Keys are:
    - Full path to a leaf ``.md`` file  (e.g. ``…/_fresh/001-hello.md``)
    - Full path to a condensed directory (e.g. ``…/01-japan-trip/``)
      — the vector encodes that directory's ``_summary.md`` ``key`` field.

    The *directory* path (without trailing slash) is used for condensed
    nodes, **not** the ``_summary.md`` path, so callers can work with
    ``os.path.join`` results directly.
    """

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # ── Write ─────────────────────────────────────────────────────────────

    def update(self, entries: dict[str, Any]) -> None:
        """Merge *entries* into the cache.  Thread-safe.

        Values may be ``list[float]`` or ``np.ndarray``; both are
        converted to ``float32`` arrays.
        """
        with self._lock:
            for k, v in entries.items():
                self._cache[k] = np.asarray(v, dtype=np.float32)

    def remove(self, paths: list[str]) -> None:
        """Remove entries for *paths*.  Missing keys are silently ignored."""
        with self._lock:
            for p in paths:
                self._cache.pop(p, None)

    def rename_prefix(self, old_prefix: str, new_prefix: str) -> None:
        """Rename all entries whose path starts with *old_prefix*.

        Used when the compactor moves a directory: all cached paths
        inside the old directory are updated to the new location in
        a single locked pass.

        Works for both exact matches (the directory itself) and all
        descendants (files inside the directory).
        """
        old = old_prefix.rstrip("/")
        new = new_prefix.rstrip("/")
        with self._lock:
            to_rename = [k for k in self._cache if k == old or k.startswith(old + "/")]
            for old_k in to_rename:
                if old_k == old:
                    new_k = new
                else:
                    new_k = new + old_k[len(old) :]
                self._cache[new_k] = self._cache.pop(old_k)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, path: str) -> np.ndarray | None:
        """Return the vector for *path*, or ``None`` if not cached."""
        return self._cache.get(path)

    def children_of(self, dir_path: str) -> dict[str, np.ndarray]:
        """Return ``{child_name: vector}`` for all direct children of *dir_path*.

        A "direct child" has exactly one path component after *dir_path*
        — it is either a leaf filename or a condensed subdirectory name,
        not a deeper descendant.

        This is the primary read interface for the assembler's top-down
        tree walk and the compactor's clustering pass.
        """
        prefix = dir_path.rstrip("/") + "/"
        result: dict[str, np.ndarray] = {}
        # Snapshot under lock to avoid mutation during iteration
        with self._lock:
            snapshot = list(self._cache.items())
        for path, vec in snapshot:
            if path.startswith(prefix):
                rest = path[len(prefix) :]
                if rest and "/" not in rest:  # direct child only
                    result[rest] = vec
        return result

    def __len__(self) -> int:
        return len(self._cache)
