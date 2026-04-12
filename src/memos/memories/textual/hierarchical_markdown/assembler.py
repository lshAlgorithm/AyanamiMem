"""Top-down tree walk + hybrid BM25/vector search for context assembly.

Given a query the assembler selects the most relevant nodes from the
hierarchical memory tree within a token budget.

Assembly pipeline (five phases):

1. **Fresh tail** — the most-recent ``fresh_tail_count`` leaves from
   ``_fresh/`` are always included verbatim at priority 1.0.
2. **Vector top-down walk** — starting from ``memory_dir``, each level
   uses :class:`EmbeddingCache` to pick branches by cosine similarity.
3. **BM25 leaf scan** — if a :class:`BM25Index` is provided, scan all
   indexed leaf text for keyword matches.
4. **RRF merge** — combine both ranked lists via Reciprocal Rank Fusion.
   Apply path constraint: at most one node per root-to-leaf path.
5. **MMR re-ranking** (optional) — re-order for diversity using
   Maximal Marginal Relevance if *enable_mmr* is set.
"""

import os
import re

from typing import Any

import numpy as np

from memos.log import get_logger
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingCache
from memos.memories.textual.hierarchical_markdown.fs import (
    FRESH_DIR,
    SUMMARY_FILENAME,
    read_md,
)


logger = get_logger(__name__)

_SEQ_PREFIX_RE = re.compile(r"^(\d+)")
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Assembler:
    """Assemble search context from the hierarchical memory tree."""

    def __init__(
        self,
        memory_dir: str,
        embedder: Any,
        emb_cache: EmbeddingCache | None = None,
        bm25_index: Any | None = None,  # BM25Index | None
        enable_mmr: bool = False,
        mmr_lambda: float = 0.5,
        hybrid_rrf_k: int = 60,
    ) -> None:
        self.memory_dir = memory_dir
        self.embedder = embedder
        self._emb_cache = emb_cache
        self._bm25 = bm25_index
        self.enable_mmr = enable_mmr
        self.mmr_lambda = mmr_lambda
        self.rrf_k = hybrid_rrf_k
        self._fresh_dir = os.path.join(memory_dir, FRESH_DIR)

    def assemble(
        self,
        query: str,
        budget_tokens: int = 4096,
        fresh_tail_count: int = 4,
    ) -> list[dict[str, Any]]:
        """Assemble context for *query* within *budget_tokens*.

        Returns a list of dicts with keys:
        ``path``, ``content``, ``meta``, ``depth``, ``score``.
        Fresh tail first, then hybrid results sorted by RRF score.
        """
        results: list[dict[str, Any]] = []
        remaining_budget = budget_tokens

        # ── Phase 1: Fresh tail ───────────────────────────────────────────
        fresh_nodes = self._fresh_tail(fresh_tail_count)
        fresh_paths: set[str] = set()
        for node in fresh_nodes:
            tokens = _estimate_tokens(node["content"])
            if tokens > remaining_budget:
                break
            remaining_budget -= tokens
            results.append(node)
            fresh_paths.add(node["path"])

        if remaining_budget <= 0:
            return results

        # ── Phase 2a: Vector top-down walk ────────────────────────────────
        try:
            query_vec = np.array(self.embedder.embed([query])[0], dtype=np.float64)
        except Exception:
            logger.warning("Failed to embed query; returning fresh tail only")
            return results

        excluded: set[str] = set(fresh_paths)
        vector_results = self._top_down_walk(
            self.memory_dir, query_vec, remaining_budget * 2, excluded, 0
        )
        vector_ranked: list[tuple[str, float]] = [
            (r["path"], r["score"])
            for r in sorted(vector_results, key=lambda x: x["score"], reverse=True)
        ]

        # ── Phase 2b: BM25 leaf scan ──────────────────────────────────────
        bm25_ranked: list[tuple[str, float]] = []
        if self._bm25 is not None:
            try:
                bm25_ranked = self._bm25.search(query, top_k=len(vector_ranked) * 3 + 20)
                bm25_ranked = [(p, s) for p, s in bm25_ranked if p not in fresh_paths]
            except Exception:
                logger.warning("BM25 search failed; falling back to vector-only")

        # ── Phase 3: RRF merge + path constraint ──────────────────────────
        if bm25_ranked:
            merged = _rrf_merge(vector_ranked, bm25_ranked, k=self.rrf_k)
        else:
            merged = vector_ranked  # vector-only fallback

        constrained_paths = _apply_path_constraint(merged, excluded)

        # ── Build candidate node dicts from merged paths ───────────────────
        path_to_node: dict[str, dict[str, Any]] = {r["path"]: r for r in vector_results}
        rrf_scores: dict[str, float] = dict(merged)

        candidates: list[dict[str, Any]] = []
        for path in constrained_paths:
            if path in path_to_node:
                node = dict(path_to_node[path])
                node["score"] = rrf_scores.get(path, node["score"])
                candidates.append(node)
            elif os.path.isfile(path):
                # BM25-only result — not in vector walk; load it
                try:
                    meta, body, _ = read_md(path)
                    candidates.append(
                        {
                            "path": path,
                            "content": body,
                            "meta": meta,
                            "depth": meta.get("depth", 0),
                            "score": rrf_scores.get(path, 0.0),
                        }
                    )
                except Exception:
                    logger.warning("Could not read BM25 candidate: %s", path)

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # ── Phase 4 (optional): MMR re-ranking ────────────────────────────
        if self.enable_mmr and candidates:
            candidates = _mmr(candidates, self._emb_cache, self.mmr_lambda)

        # ── Fill remaining budget ─────────────────────────────────────────
        for node in candidates:
            if remaining_budget <= 0:
                break
            tokens = _estimate_tokens(node["content"])
            if tokens > remaining_budget:
                continue
            remaining_budget -= tokens
            results.append(node)

        return results

    # ── Private helpers ───────────────────────────────────────────────────

    def _fresh_tail(self, count: int) -> list[dict[str, Any]]:
        if not os.path.isdir(self._fresh_dir):
            return []
        leaves: list[tuple[int, str]] = []
        for entry in os.listdir(self._fresh_dir):
            if entry.startswith("_") or not entry.endswith(".md"):
                continue
            m = _SEQ_PREFIX_RE.match(entry)
            leaves.append((int(m.group(1)) if m else 999, entry))
        leaves.sort(key=lambda x: x[0])
        tail = leaves[-count:] if count > 0 else []
        out: list[dict[str, Any]] = []
        for _seq, fname in tail:
            fpath = os.path.join(self._fresh_dir, fname)
            try:
                meta, body, _ = read_md(fpath)
                out.append({"path": fpath, "content": body, "meta": meta, "depth": 0, "score": 1.0})
            except Exception:
                logger.warning("Could not read fresh tail leaf: %s", fpath)
        return out

    def _top_down_walk(
        self,
        dir_path: str,
        query_vec: np.ndarray,
        budget_tokens: int,
        excluded_paths: set[str],
        current_depth: int,
    ) -> list[dict[str, Any]]:
        if os.path.basename(dir_path) == FRESH_DIR:
            return []

        emb_data = self._emb_cache.children_of(dir_path) if self._emb_cache else {}
        if not emb_data:
            return []

        scored = sorted(
            (
                (float(_cosine_similarity(query_vec, np.array(vec, dtype=np.float64))), name)
                for name, vec in emb_data.items()
            ),
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        remaining = budget_tokens

        for sim, name in scored:
            if remaining <= 0:
                break
            child_path = os.path.join(dir_path, name)

            if os.path.isdir(child_path):
                summary_path = os.path.join(child_path, SUMMARY_FILENAME)
                if summary_path in excluded_paths:
                    continue
                if not os.path.isfile(summary_path):
                    continue
                try:
                    meta, body, _ = read_md(summary_path)
                except Exception:
                    continue

                summary_tokens = _estimate_tokens(body)
                sub_emb = self._emb_cache.children_of(child_path) if self._emb_cache else {}
                if sub_emb and remaining > summary_tokens * 2:
                    sub = self._top_down_walk(
                        child_path, query_vec, remaining, excluded_paths, current_depth + 1
                    )
                    if sub:
                        results.extend(sub)
                        remaining -= sum(_estimate_tokens(r["content"]) for r in sub)
                        continue

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
                if child_path in excluded_paths or not os.path.isfile(child_path):
                    continue
                try:
                    meta, body, _ = read_md(child_path)
                except Exception:
                    continue
                leaf_tokens = _estimate_tokens(body)
                if leaf_tokens <= remaining:
                    results.append(
                        {
                            "path": child_path,
                            "content": body,
                            "meta": meta,
                            "depth": current_depth,
                            "score": sim,
                        }
                    )
                    remaining -= leaf_tokens
                    excluded_paths.add(child_path)

        return results


# ── Module-level helpers (pure functions, no state) ───────────────────────────


def _rrf_merge(
    vector_ranked: list[tuple[str, float]],
    bm25_ranked: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists.

    ``score = Σ 1/(k + rank_i)``  where rank is 1-based.
    Paths appearing in only one list get 0 from the other.
    """
    scores: dict[str, float] = {}
    for rank, (path, _) in enumerate(vector_ranked, start=1):
        scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank)
    for rank, (path, _) in enumerate(bm25_ranked, start=1):
        scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _apply_path_constraint(
    ranked: list[tuple[str, float]],
    excluded_fresh: set[str],
) -> list[str]:
    """Keep at most one node per root-to-leaf path.

    A ``_summary.md`` node "covers" its entire parent directory and all
    subdirectories within it.  Once a summary is selected, any leaf or
    child summary inside that directory is excluded — and vice versa:
    if a leaf is selected, the parent summary is excluded because it would
    redundantly cover the same content.

    Implementation:
    - ``coverage_dirs``: directories whose summary has been selected.
      Any path starting with ``dir + "/"`` is excluded.
    - ``selected_paths``: all selected paths.  If a new candidate is a
      summary for dir D, we exclude it if any already-selected path is
      inside D.
    """
    selected: list[str] = []
    selected_paths: set[str] = set()
    coverage_dirs: set[str] = set()  # dirs covered by a selected summary

    for path, _ in ranked:
        if path in excluded_fresh:
            continue

        p = path.rstrip("/")

        # Is this path inside a directory already covered by a selected summary?
        if any(p.startswith(d + "/") for d in coverage_dirs):
            continue

        # Is this path a summary?  If so, compute its coverage directory.
        if os.path.basename(p) == SUMMARY_FILENAME:
            parent_dir = os.path.dirname(p)
            # Does any already-selected path live inside this directory?
            if any(s.startswith(parent_dir + "/") for s in selected_paths):
                continue  # a child is already selected — skip this summary
            selected.append(path)
            selected_paths.add(path)
            coverage_dirs.add(parent_dir)
        else:
            selected.append(path)
            selected_paths.add(path)

    return selected


def _mmr(
    candidates: list[dict[str, Any]],
    emb_cache: EmbeddingCache | None,
    lam: float = 0.5,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance re-ranking.

    Iteratively selects: ``argmax [ λ·relevance - (1-λ)·max_sim_to_selected ]``

    Falls back to pure relevance for candidates without a cached embedding.
    """
    if not candidates:
        return candidates

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)

    while remaining:
        if not selected:
            best = max(remaining, key=lambda x: x["score"])
        else:

            def mmr_score(c: dict[str, Any]) -> float:
                if emb_cache is None:
                    return lam * c["score"]
                v = emb_cache.get(c["path"])
                if v is None:
                    return lam * c["score"]
                sims = []
                for s in selected:
                    sv = emb_cache.get(s["path"])
                    if sv is not None:
                        sims.append(_cosine_similarity(v.astype(np.float64), sv.astype(np.float64)))
                max_sim = max(sims) if sims else 0.0
                return lam * c["score"] - (1.0 - lam) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected
