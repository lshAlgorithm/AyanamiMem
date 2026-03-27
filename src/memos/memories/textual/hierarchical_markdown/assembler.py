"""Top-down tree walk for search context assembly.

Given a query, the assembler selects the most relevant nodes from the
hierarchical memory tree within a token budget.  Fresh-tail leaves are
always included verbatim, then a top-down walk selects the best branches
by embedding similarity, yielding a ranked list of context snippets.
"""

import os
import re

from typing import Any

import numpy as np

from memos.log import get_logger
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingIndex
from memos.memories.textual.hierarchical_markdown.fs import (
    FRESH_DIR,
    SUMMARY_FILENAME,
    read_md,
)


logger = get_logger(__name__)

_SEQ_PREFIX_RE = re.compile(r"^(\d+)")

# Rough token estimate: 1 token ≈ 4 chars
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count from character length."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class Assembler:
    """Assemble search context from the hierarchical memory tree.

    Parameters:
        memory_dir: Root directory of the memory tree.
        embedder: Embedder instance (must expose ``embed(texts) -> list[list[float]]``).
    """

    def __init__(self, memory_dir: str, embedder: Any) -> None:
        self.memory_dir = memory_dir
        self.embedder = embedder
        self._fresh_dir = os.path.join(memory_dir, FRESH_DIR)

    def assemble(
        self,
        query: str,
        budget_tokens: int = 4096,
        fresh_tail_count: int = 4,
    ) -> list[dict[str, Any]]:
        """Assemble context for *query* within *budget_tokens*.

        The assembly strategy has two phases:

        1. **Fresh tail**: the last *fresh_tail_count* leaves from ``_fresh/``
           are always included verbatim (most recent conversation context).
        2. **Top-down walk**: starting from the memory tree root, each level's
           ``_embeddings.json`` is consulted to pick the branches most similar
           to the query.  If a node is selected, its ``_summary.md`` content
           represents the entire subtree — descendants are not traversed.

        Args:
            query: The search query text.
            budget_tokens: Maximum total tokens for the assembled context.
            fresh_tail_count: Number of most-recent fresh leaves to include.

        Returns:
            A list of dicts, each with keys:
            ``path``, ``content``, ``meta``, ``depth``, ``score``.
            Ordered by relevance (fresh tail first, then by score descending).
        """
        results: list[dict[str, Any]] = []
        remaining_budget = budget_tokens

        # ── Phase 1: Fresh tail ───────────────────────────────────────────
        tail_nodes = self._fresh_tail(fresh_tail_count)
        for node in tail_nodes:
            tokens = _estimate_tokens(node["content"])
            if tokens > remaining_budget:
                break
            remaining_budget -= tokens
            results.append(node)

        if remaining_budget <= 0:
            return results

        # ── Phase 2: Top-down walk ────────────────────────────────────────
        try:
            query_vec = np.array(self.embedder.embed([query])[0], dtype=np.float64)
        except Exception:
            logger.warning("Failed to embed query, returning fresh tail only")
            return results

        # Collect paths already included (to avoid duplicates)
        included_paths: set[str] = {r["path"] for r in results}

        walk_results = self._top_down_walk(
            self.memory_dir, query_vec, remaining_budget, included_paths, current_depth=0
        )

        # Sort walk results by score descending
        walk_results.sort(key=lambda r: r["score"], reverse=True)

        for node in walk_results:
            tokens = _estimate_tokens(node["content"])
            if tokens > remaining_budget:
                continue
            remaining_budget -= tokens
            results.append(node)
            if remaining_budget <= 0:
                break

        return results

    # ── Private helpers ───────────────────────────────────────────────────

    def _fresh_tail(self, count: int) -> list[dict[str, Any]]:
        """Read the last *count* leaf files from ``_fresh/``.

        Returns a list of result dicts, ordered oldest-to-newest.
        """
        if not os.path.isdir(self._fresh_dir):
            return []

        # Collect and sort leaves by sequence number
        leaves: list[tuple[int, str]] = []
        for entry in os.listdir(self._fresh_dir):
            if entry.startswith("_") or not entry.endswith(".md"):
                continue
            m = _SEQ_PREFIX_RE.match(entry)
            seq = int(m.group(1)) if m else 999
            leaves.append((seq, entry))
        leaves.sort(key=lambda x: x[0])

        # Take the tail
        tail = leaves[-count:] if count > 0 else []
        results: list[dict[str, Any]] = []
        for _seq, fname in tail:
            fpath = os.path.join(self._fresh_dir, fname)
            try:
                meta, body, _edges = read_md(fpath)
                results.append(
                    {
                        "path": fpath,
                        "content": body,
                        "meta": meta,
                        "depth": 0,
                        "score": 1.0,  # Fresh tail always highest priority
                    }
                )
            except Exception:
                logger.warning("Could not read fresh tail leaf: %s", fpath)

        return results

    def _top_down_walk(
        self,
        dir_path: str,
        query_vec: np.ndarray,
        budget_tokens: int,
        excluded_paths: set[str],
        current_depth: int,
    ) -> list[dict[str, Any]]:
        """Recursively walk the tree, selecting best branches by similarity.

        At each directory level:
        1. Read ``_embeddings.json`` and score each child against the query.
        2. Rank children by similarity.
        3. For the best children within budget: if a child is a directory,
           decide whether to use its ``_summary.md`` or descend further.
        4. If a child is a leaf file, include it directly.

        Returns a flat list of result dicts.
        """
        # Skip _fresh/ — already handled by fresh tail
        if os.path.basename(dir_path) == FRESH_DIR:
            return []

        idx = EmbeddingIndex(dir_path)
        emb_data = idx.read()

        if not emb_data:
            return []

        # Score each child
        scored: list[tuple[float, str]] = []
        for name, vec in emb_data.items():
            child_vec = np.array(vec, dtype=np.float64)
            sim = _cosine_similarity(query_vec, child_vec)
            scored.append((sim, name))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        remaining = budget_tokens

        for sim, name in scored:
            if remaining <= 0:
                break

            # Determine the child path
            child_path = os.path.join(dir_path, name)

            if os.path.isdir(child_path):
                # It's a subdirectory — check its _summary.md
                summary_path = os.path.join(child_path, SUMMARY_FILENAME)
                if summary_path in excluded_paths:
                    continue

                if os.path.isfile(summary_path):
                    try:
                        meta, body, _edges = read_md(summary_path)
                    except Exception:
                        logger.warning("Could not read summary: %s", summary_path)
                        continue

                    summary_tokens = _estimate_tokens(body)

                    # Decide: use summary or descend?
                    # Descend if budget allows and the directory has embeddings
                    sub_idx = EmbeddingIndex(child_path)
                    sub_emb = sub_idx.read()

                    if sub_emb and remaining > summary_tokens * 2:
                        # Budget allows deeper exploration
                        sub_results = self._top_down_walk(
                            child_path,
                            query_vec,
                            remaining,
                            excluded_paths,
                            current_depth + 1,
                        )
                        if sub_results:
                            results.extend(sub_results)
                            remaining -= sum(_estimate_tokens(r["content"]) for r in sub_results)
                            continue

                    # Use summary as-is (covers the subtree)
                    if summary_tokens <= remaining:
                        results.append(
                            {
                                "path": summary_path,
                                "content": body,
                                "meta": meta,
                                "depth": current_depth + 1,
                                "score": sim,
                            }
                        )
                        remaining -= summary_tokens
                        excluded_paths.add(summary_path)

            elif name.endswith(".md"):
                # It's a leaf file
                leaf_path = child_path
                if leaf_path in excluded_paths:
                    continue
                if not os.path.isfile(leaf_path):
                    continue

                try:
                    meta, body, _edges = read_md(leaf_path)
                except Exception:
                    logger.warning("Could not read leaf: %s", leaf_path)
                    continue

                leaf_tokens = _estimate_tokens(body)
                if leaf_tokens <= remaining:
                    results.append(
                        {
                            "path": leaf_path,
                            "content": body,
                            "meta": meta,
                            "depth": current_depth,
                            "score": sim,
                        }
                    )
                    remaining -= leaf_tokens
                    excluded_paths.add(leaf_path)

        return results
