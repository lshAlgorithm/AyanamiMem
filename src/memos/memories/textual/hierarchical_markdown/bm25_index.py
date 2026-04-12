"""In-memory BM25 keyword index over leaf body text.

Maintains a ``{path: tokenized_words}`` mapping and a lazily rebuilt
``BM25Okapi`` model.  The model is rebuilt only when ``search()`` is
called after any mutation (``dirty`` flag pattern), so compaction and
sequential adds don't pay rebuild cost per-mutation.

Indexed text = ``key + " " + body`` for each leaf node.  Summaries
(``_summary.md`` files) are NOT indexed — BM25 is for exact-fact
retrieval from verbatim conversation chunks; vector search covers
topic-level summary navigation.

The interface mirrors :class:`EmbeddingCache` exactly so the main class
can call ``update`` / ``remove`` / ``rename_prefix`` on both in the same
mutation hook.
"""

import re
import threading

from typing import Any

from memos.log import get_logger


logger = get_logger(__name__)

# Simple tokenizer: lowercase alphanumeric tokens only.
# Preserves numbers ("3000"), acronyms ("ros2", "c", "llm"), etc.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Thread-safe in-memory BM25Okapi index over leaf node text.

    Indexed text per node = ``key + " " + body`` so that both the
    topic label and the verbatim conversation content are searchable.

    The underlying ``BM25Okapi`` model is rebuilt lazily: mutations set a
    ``dirty`` flag, and ``search()`` triggers ``_rebuild()`` if needed.
    For typical memory sizes (<10 000 leaves) rebuild takes ~10 ms.
    """

    def __init__(self) -> None:
        self._texts: dict[str, list[str]] = {}  # path → tokenized words
        self._bm25: Any = None  # BM25Okapi | None
        self._paths: list[str] = []  # ordered list matching corpus
        self._dirty: bool = False
        self._lock = threading.Lock()

    # ── Write ─────────────────────────────────────────────────────────────

    def update(self, entries: dict[str, str]) -> None:
        """Index *entries*: ``{path: raw_text}``.

        *raw_text* should be ``key + " " + body`` for each leaf.
        Call this after writing a new leaf or after re-summarisation.
        """
        with self._lock:
            for path, text in entries.items():
                self._texts[path] = _tokenize(text)
            self._dirty = True

    def remove(self, paths: list[str]) -> None:
        """Remove *paths* from the index.  Missing paths are ignored."""
        with self._lock:
            changed = False
            for p in paths:
                if p in self._texts:
                    del self._texts[p]
                    changed = True
            if changed:
                self._dirty = True

    def rename_prefix(self, old_prefix: str, new_prefix: str) -> None:
        """Rename all entries whose path starts with *old_prefix*.

        Mirrors :meth:`EmbeddingCache.rename_prefix` so both caches can be
        updated with a single call per file move.
        """
        old = old_prefix.rstrip("/")
        new = new_prefix.rstrip("/")
        with self._lock:
            to_rename = [k for k in self._texts if k == old or k.startswith(old + "/")]
            if not to_rename:
                return
            for old_k in to_rename:
                new_k = new if old_k == old else new + old_k[len(old) :]
                self._texts[new_k] = self._texts.pop(old_k)
            self._dirty = True

    def clear(self) -> None:
        with self._lock:
            self._texts.clear()
            self._bm25 = None
            self._paths = []
            self._dirty = False

    # ── Read ──────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``[(path, bm25_score), ...]`` for the top-*k* matches.

        Triggers a model rebuild if the index has been mutated since the
        last search.  Returns an empty list if the index is empty or the
        query has no matching tokens.
        """
        with self._lock:
            if not self._texts:
                return []
            if self._dirty:
                self._rebuild()
            if self._bm25 is None:
                return []

            tokens = _tokenize(query)
            if not tokens:
                return []

            scores: list[float] = self._bm25.get_scores(tokens).tolist()
            # BM25Plus always returns non-negative scores; 0.0 means no match.
            pairs = [
                (self._paths[i], scores[i]) for i in range(len(self._paths)) if scores[i] > 0.0
            ]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:top_k]

    def __len__(self) -> int:
        return len(self._texts)

    # ── Private ───────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild the BM25Plus model from the current text dict.

        Uses ``BM25Plus`` instead of ``BM25Okapi`` to guarantee non-negative
        scores for all matched terms.  ``BM25Okapi`` can return IDF=0 when a
        term appears in exactly half the corpus (and negative IDF when it
        appears in more than half), making scores 0 or negative even for
        relevant documents.  ``BM25Plus`` uses ``idf = log((N+1)/n_q)``
        which is always ≥ log(2) > 0.

        Must be called with ``self._lock`` held.
        """
        try:
            from rank_bm25 import BM25Plus  # lazy import
        except ImportError:
            logger.warning("rank_bm25 not installed; BM25 search disabled")
            self._bm25 = None
            self._dirty = False
            return

        self._paths = list(self._texts.keys())
        corpus = [self._texts[p] for p in self._paths]
        if corpus:
            self._bm25 = BM25Plus(corpus)
        else:
            self._bm25 = None
        self._dirty = False
        logger.debug("BM25Plus index rebuilt: %d documents", len(corpus))
