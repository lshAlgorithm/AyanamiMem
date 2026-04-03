"""Compaction engine: cluster _fresh/ leaves and build tree structure.

The compactor periodically groups accumulated leaf files from ``_fresh/``
by embedding similarity, creates subdirectories for each cluster, writes
``_summary.md`` nodes, and cascades condensation upward through the tree.

Concurrency is guarded by a simple lock file (``_fresh/.compacting``) so
that only one compaction process runs at a time.
"""

import os
import re

from datetime import datetime
from typing import Any

import numpy as np

from memos.log import get_logger
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingCache
from memos.memories.textual.hierarchical_markdown.fs import (
    FRESH_DIR,
    SUMMARY_FILENAME,
    next_seq,
    subdir_name,
    write_summary,
)
from memos.memories.textual.hierarchical_markdown.summarizer import HierarchicalSummarizer


logger = get_logger(__name__)

_SEQ_PREFIX_RE = re.compile(r"^(\d+)")
_LOCK_FILENAME = ".compacting"


# ── Similarity helpers ────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _agglomerative_cluster(
    keys: list[str],
    embeddings: dict[str, list[float] | np.ndarray],
    threshold: float = 0.5,
) -> list[list[str]]:
    """Simple agglomerative clustering using cosine similarity.

    Greedily merges the most similar pair of clusters until no pair exceeds
    *threshold*.  Returns a list of clusters (each cluster is a list of keys).
    """
    if not keys:
        return []

    # Build initial clusters: one per key that has an embedding
    clusters: list[list[str]] = []
    cluster_vecs: list[np.ndarray] = []
    no_embedding: list[str] = []

    for k in keys:
        if k in embeddings:
            clusters.append([k])
            cluster_vecs.append(np.array(embeddings[k], dtype=np.float64))
        else:
            no_embedding.append(k)

    # Greedy merge
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        best_sim = -1.0
        best_i, best_j = -1, -1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = _cosine_similarity(cluster_vecs[i], cluster_vecs[j])
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j

        if best_sim >= threshold:
            # Merge j into i
            clusters[best_i].extend(clusters[best_j])
            # Update centroid as mean
            merged_vecs = []
            for k in clusters[best_i]:
                if k in embeddings:
                    merged_vecs.append(np.array(embeddings[k], dtype=np.float64))
            if merged_vecs:
                cluster_vecs[best_i] = np.mean(merged_vecs, axis=0)
            clusters.pop(best_j)
            cluster_vecs.pop(best_j)
            changed = True

    # Attach orphan keys (no embedding) to the first cluster, or make their own
    if no_embedding:
        if clusters:
            clusters[0].extend(no_embedding)
        else:
            clusters.append(no_embedding)

    return clusters


# ── Compactor ─────────────────────────────────────────────────────────────────


class Compactor:
    """Compaction engine for the hierarchical Markdown memory tree.

    Parameters:
        memory_dir: Root directory of the memory tree.
        summarizer: The :class:`HierarchicalSummarizer` for generating summaries.
        embedder: Embedder instance (must expose ``embed(texts) -> list[list[float]]``).
        compact_threshold: Minimum non-tail leaves in ``_fresh/`` to trigger compaction.
        fresh_tail_count: Number of most-recent leaves to keep in ``_fresh/`` (never compacted).
        condensed_min_fanout: Minimum sibling count to trigger a condensation pass.
        similarity_threshold: Cosine-similarity threshold for agglomerative clustering.
    """

    def __init__(
        self,
        memory_dir: str,
        summarizer: HierarchicalSummarizer,
        embedder: Any,
        compact_threshold: int = 8,
        fresh_tail_count: int = 4,
        condensed_min_fanout: int = 6,
        similarity_threshold: float = 0.5,
        emb_cache: EmbeddingCache | None = None,
    ) -> None:
        self.memory_dir = memory_dir
        self.summarizer = summarizer
        self.embedder = embedder
        self.compact_threshold = compact_threshold
        self.fresh_tail_count = fresh_tail_count
        self.condensed_min_fanout = condensed_min_fanout
        self.similarity_threshold = similarity_threshold
        self._emb_cache = emb_cache
        self._fresh_dir = os.path.join(memory_dir, FRESH_DIR)

    # ── Public API ────────────────────────────────────────────────────────

    def should_compact(self) -> bool:
        """Return ``True`` if enough non-tail leaves exist in ``_fresh/``."""
        leaves = self._sorted_leaves()
        non_tail = len(leaves) - self.fresh_tail_count
        return non_tail >= self.compact_threshold

    def leaf_pass(self) -> int:
        """Cluster non-tail leaves and move them into subdirectories.

        Returns the number of leaves compacted.
        """
        leaves = self._sorted_leaves()
        if len(leaves) <= self.fresh_tail_count:
            return 0

        # Exclude the tail (most recent leaves stay in _fresh/)
        to_compact = leaves[: len(leaves) - self.fresh_tail_count]
        if not to_compact:
            return 0

        # Read embeddings for clustering from in-memory cache
        emb_data = self._emb_cache.children_of(self._fresh_dir) if self._emb_cache else {}

        # Cluster
        clusters = _agglomerative_cluster(to_compact, emb_data, self.similarity_threshold)

        compacted_count = 0
        moved_keys: list[str] = []

        for cluster in clusters:
            if not cluster:
                continue

            child_paths = [os.path.join(self._fresh_dir, f) for f in cluster]
            existing_paths = [p for p in child_paths if os.path.isfile(p)]
            if not existing_paths:
                continue

            # Summarize the cluster
            summary_body, key, tags = self.summarizer.summarize(existing_paths, depth=1)

            # Create subdirectory
            seq = next_seq(self.memory_dir, for_dir=True)
            dirname = subdir_name(seq, key)
            target_dir = os.path.join(self.memory_dir, dirname)
            os.makedirs(target_dir, exist_ok=True)

            # Move leaf files into the new directory
            children_links: list[tuple[str, str]] = []
            for leaf_name in cluster:
                src = os.path.join(self._fresh_dir, leaf_name)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(target_dir, leaf_name)
                os.rename(src, dst)
                # Update in-memory cache: rename old path to new path
                if self._emb_cache is not None:
                    self._emb_cache.rename_prefix(src, dst)
                children_links.append((leaf_name.rsplit(".", 1)[0], f"./{leaf_name}"))
                moved_keys.append(leaf_name)
                compacted_count += 1

            # Write _summary.md in the new directory
            now_iso = datetime.now().isoformat()
            summary_meta: dict[str, Any] = {
                "key": key,
                "depth": 1,
                "node_kind": "condensed",
                "earliest_at": now_iso,
                "latest_at": now_iso,
                "token_count": len(summary_body) // 4,
                "descendant_count": len(children_links),
                "stale": False,
                "tags": tags,
            }
            write_summary(target_dir, summary_body, summary_meta, children_links)

            # Register the new condensed dir in the cache
            if self._emb_cache is not None:
                try:
                    summary_vec = self.embedder.embed([key])[0]
                    self._emb_cache.update({target_dir: summary_vec})
                except Exception:
                    logger.warning("Failed to embed summary key for %s", dirname)

        logger.info(
            "Leaf pass compacted %d leaves into %d clusters", compacted_count, len(clusters)
        )
        return compacted_count

    def condensation_pass(self, depth: int = 1) -> int:
        """Condense sibling directories whose ``_summary.md`` has the given *depth*.

        Scans ``memory_dir`` for subdirectories whose summary ``depth`` field
        matches *depth*.  If enough exist (>= ``condensed_min_fanout``), groups
        them by embedding similarity under new parent directories.

        Returns the number of directories condensed.
        """
        # Find all direct-child dirs in memory_dir with _summary.md at the target depth
        target_dirs: list[str] = []
        parent_dir = self.memory_dir
        for entry in sorted(os.listdir(parent_dir)):
            if entry.startswith("_"):
                continue
            full = os.path.join(parent_dir, entry)
            if not os.path.isdir(full):
                continue
            summary_path = os.path.join(full, SUMMARY_FILENAME)
            if os.path.isfile(summary_path):
                from memos.memories.textual.hierarchical_markdown.fs import read_md

                meta, _body, _edges = read_md(summary_path)
                if meta.get("depth") == depth:
                    target_dirs.append(full)

        if len(target_dirs) < self.condensed_min_fanout:
            return 0

        parent_dir = os.path.dirname(target_dirs[0]) if target_dirs else self.memory_dir
        dir_names = [os.path.basename(d) for d in target_dirs]

        # Get embeddings from in-memory cache
        dir_embeddings: dict[str, np.ndarray] = {}
        if self._emb_cache is not None:
            children = self._emb_cache.children_of(parent_dir)
            for dname in dir_names:
                if dname in children:
                    dir_embeddings[dname] = children[dname]

        clusters = _agglomerative_cluster(dir_names, dir_embeddings, self.similarity_threshold)

        condensed_count = 0
        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Collect summary paths for the cluster
            child_summary_paths: list[str] = []
            for dname in cluster:
                summary_path = os.path.join(parent_dir, dname, SUMMARY_FILENAME)
                if os.path.isfile(summary_path):
                    child_summary_paths.append(summary_path)

            if not child_summary_paths:
                continue

            # Generate parent summary
            summary_body, key, tags = self.summarizer.summarize(
                child_summary_paths, depth=depth + 1
            )

            # Create new parent directory
            seq = next_seq(parent_dir, for_dir=True)
            new_dirname = subdir_name(seq, key)
            new_dir = os.path.join(parent_dir, new_dirname)
            os.makedirs(new_dir, exist_ok=True)

            # Move children into the new parent, updating cache paths
            children_links: list[tuple[str, str]] = []
            moved_dir_names: list[str] = []
            for dname in cluster:
                src = os.path.join(parent_dir, dname)
                dst = os.path.join(new_dir, dname)
                if os.path.isdir(src):
                    os.rename(src, dst)
                    if self._emb_cache is not None:
                        self._emb_cache.rename_prefix(src, dst)
                    children_links.append((dname, f"./{dname}/{SUMMARY_FILENAME}"))
                    moved_dir_names.append(dname)
                    condensed_count += 1

            # Write parent _summary.md
            now_iso = datetime.now().isoformat()
            total_descendants = sum(
                self._count_descendants(os.path.join(new_dir, d)) for d in moved_dir_names
            )
            summary_meta: dict[str, Any] = {
                "key": key,
                "depth": depth + 1,
                "node_kind": "condensed",
                "earliest_at": now_iso,
                "latest_at": now_iso,
                "token_count": len(summary_body) // 4,
                "descendant_count": total_descendants,
                "stale": False,
                "tags": tags,
            }
            write_summary(new_dir, summary_body, summary_meta, children_links)

            # Register new parent dir in cache
            if self._emb_cache is not None:
                try:
                    summary_vec = self.embedder.embed([key])[0]
                    self._emb_cache.update({new_dir: summary_vec})
                except Exception:
                    logger.warning("Failed to embed condensed key for %s", new_dirname)

        logger.info("Condensation pass at depth %d condensed %d dirs", depth, condensed_count)
        return condensed_count

    def compact_incremental(self, force: bool = False) -> dict[str, int]:
        """Run a full incremental compaction cycle.

        1. Acquire the lock file to prevent concurrent compaction.
        2. Run ``leaf_pass`` if ``should_compact()`` is ``True`` *or* ``force=True``.
        3. Cascade ``condensation_pass`` upward through all depths.
        4. Release the lock.

        Args:
            force: If ``True``, run ``leaf_pass`` even when ``should_compact()``
                   returns ``False``.  Used by explicit user-triggered compaction
                   (e.g. the ``/compact`` command) to bypass the auto-compact
                   threshold while still respecting the lock and full cascade.

        Returns:
            A dict with ``{"leaves_compacted": N, "dirs_condensed": M}`` so
            callers can report what happened without peeking at internals.
        """
        result = {"leaves_compacted": 0, "dirs_condensed": 0}
        lock_path = os.path.join(self._fresh_dir, _LOCK_FILENAME)
        if os.path.exists(lock_path):
            logger.info("Compaction already in progress (lock file exists), skipping")
            return result

        try:
            # Acquire lock
            os.makedirs(self._fresh_dir, exist_ok=True)
            with open(lock_path, "w") as f:
                f.write(datetime.now().isoformat())

            # Step 1: Leaf pass — honour threshold unless forced
            if force or self.should_compact():
                compacted = self.leaf_pass()
                result["leaves_compacted"] = compacted
                logger.info("Leaf pass completed: %d leaves compacted", compacted)
            else:
                logger.debug("Not enough leaves to compact (%s)", self._fresh_dir)

            # Step 2: Cascade condensation upward through all depths
            max_depth = self._max_depth(self.memory_dir, current_depth=0)
            for depth in range(1, max_depth + 1):
                condensed = self.condensation_pass(depth)
                if condensed == 0:
                    break  # No more condensation needed at higher depths
                result["dirs_condensed"] += condensed
                logger.info("Condensation at depth %d: %d dirs condensed", depth, condensed)

        except Exception:
            logger.exception("Compaction failed")
        finally:
            # Release lock
            if os.path.exists(lock_path):
                os.remove(lock_path)

        return result

    # ── Private helpers ───────────────────────────────────────────────────

    def _sorted_leaves(self) -> list[str]:
        """Return leaf filenames in ``_fresh/`` sorted by sequence number."""
        if not os.path.isdir(self._fresh_dir):
            return []
        leaves: list[tuple[int, str]] = []
        for entry in os.listdir(self._fresh_dir):
            if entry.startswith("_") or not entry.endswith(".md"):
                continue
            m = _SEQ_PREFIX_RE.match(entry)
            seq = int(m.group(1)) if m else 999
            leaves.append((seq, entry))
        leaves.sort(key=lambda x: x[0])
        return [name for _, name in leaves]

    def _dirs_at_depth(self, base: str, target_depth: int, current_depth: int) -> list[str]:
        """Recursively find subdirectories at a specific depth."""
        if not os.path.isdir(base):
            return []

        if current_depth == target_depth:
            # Return subdirectories at this level (those with _summary.md)
            result: list[str] = []
            for entry in sorted(os.listdir(base)):
                if entry.startswith("_"):
                    continue
                full = os.path.join(base, entry)
                if os.path.isdir(full) and os.path.isfile(os.path.join(full, SUMMARY_FILENAME)):
                    result.append(full)
            return result

        # Recurse into subdirectories
        result = []
        for entry in sorted(os.listdir(base)):
            if entry.startswith("_"):
                continue
            full = os.path.join(base, entry)
            if os.path.isdir(full):
                result.extend(self._dirs_at_depth(full, target_depth, current_depth + 1))
        return result

    def _max_depth(self, base: str, current_depth: int) -> int:
        """Find the maximum directory nesting depth in the tree."""
        if not os.path.isdir(base):
            return current_depth
        max_d = current_depth
        for entry in os.listdir(base):
            if entry.startswith("_"):
                continue
            full = os.path.join(base, entry)
            if os.path.isdir(full):
                max_d = max(max_d, self._max_depth(full, current_depth + 1))
        return max_d

    def _count_descendants(self, dir_path: str) -> int:
        """Count total leaf descendants under *dir_path*."""
        count = 0
        if not os.path.isdir(dir_path):
            return 0
        for entry in os.listdir(dir_path):
            if entry.startswith("_"):
                continue
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                count += self._count_descendants(full)
            elif entry.endswith(".md"):
                count += 1
        return count
