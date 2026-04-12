"""Tests for BM25Index + hybrid search helpers (RRF, path constraint, MMR)."""

import threading

import pytest

from memos.memories.textual.hierarchical_markdown.assembler import (
    _apply_path_constraint,
    _rrf_merge,
)
from memos.memories.textual.hierarchical_markdown.bm25_index import BM25Index


# ── BM25Index unit tests ──────────────────────────────────────────────────────


@pytest.fixture()
def idx():
    return BM25Index()


class TestBM25Basic:
    def test_empty_search(self, idx):
        assert idx.search("anything", top_k=5) == []

    def test_update_and_search(self, idx):
        idx.update({"/mem/001.md": "Japan trip planning Kyoto temples"})
        results = idx.search("Japan", top_k=5)
        assert len(results) == 1
        assert results[0][0] == "/mem/001.md"
        assert results[0][1] > 0.0

    def test_exact_number_match(self, idx):
        idx.update(
            {
                "/mem/001.md": "budget planning Japan 3000 USD travel",
                "/mem/002.md": "machine learning Python neural network",
            }
        )
        results = idx.search("3000 USD", top_k=5)
        assert results[0][0] == "/mem/001.md"  # exact number hit ranks first

    def test_no_match_returns_empty(self, idx):
        idx.update({"/mem/001.md": "cats and dogs"})
        results = idx.search("quantum physics", top_k=5)
        # BM25 may return zero-score results filtered out
        scores = [s for _, s in results]
        assert all(s == 0.0 for s in scores) or results == []

    def test_top_k_respected(self, idx):
        for i in range(20):
            idx.update({f"/mem/{i:03d}.md": f"topic {i} keyword search test"})
        results = idx.search("keyword search", top_k=5)
        assert len(results) <= 5

    def test_ranking_order(self, idx):
        idx.update(
            {
                "/mem/a.md": "Python programming language code",
                "/mem/b.md": "Python Python Python coding language",
            }
        )
        results = idx.search("Python", top_k=5)
        paths = [p for p, _ in results]
        assert len(paths) == 2
        # b.md has higher term frequency — should rank higher with BM25Plus
        assert paths[0] == "/mem/b.md"


class TestBM25Mutations:
    def test_remove(self, idx):
        idx.update({"/mem/a.md": "Japan trip", "/mem/b.md": "ML project"})
        idx.remove(["/mem/a.md"])
        results = idx.search("Japan", top_k=5)
        assert all(p != "/mem/a.md" for p, _ in results)
        assert len(idx) == 1

    def test_remove_missing_safe(self, idx):
        idx.update({"/mem/a.md": "some text"})
        idx.remove(["/nonexistent.md"])  # must not raise
        assert len(idx) == 1

    def test_rename_prefix_file(self, idx):
        idx.update({"/mem/_fresh/001.md": "Japan trip planning"})
        idx.rename_prefix("/mem/_fresh/001.md", "/mem/01-japan/001.md")
        assert len(idx) == 1
        results = idx.search("Japan", top_k=5)
        paths = [p for p, _ in results]
        assert "/mem/01-japan/001.md" in paths

    def test_rename_prefix_dir(self, idx):
        idx.update(
            {
                "/mem/old/001.md": "Japan",
                "/mem/old/002.md": "Tokyo",
            }
        )
        idx.rename_prefix("/mem/old", "/mem/new-japan")
        assert len(idx) == 2
        results = idx.search("Japan", top_k=5)
        paths = [p for p, _ in results]
        assert all(p.startswith("/mem/new-japan/") for p in paths)
        assert not any(p.startswith("/mem/old/") for p in paths)

    def test_clear(self, idx):
        idx.update({"/a.md": "text", "/b.md": "text"})
        idx.clear()
        assert len(idx) == 0
        assert idx.search("text", top_k=5) == []

    def test_dirty_flag_triggers_rebuild(self, idx):
        idx.update({"/a.md": "alpha beta gamma"})
        r1 = idx.search("alpha", top_k=3)
        assert len(r1) == 1
        # Update again (marks dirty)
        idx.update({"/b.md": "alpha delta epsilon"})
        r2 = idx.search("alpha", top_k=3)
        assert len(r2) == 2  # both docs now in index


class TestBM25Concurrency:
    def test_concurrent_update_and_search(self, idx):
        idx.update({"/seed.md": "seed document"})
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(40):
                    idx.update({f"/w{i}.md": f"document {i} keyword"})
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(40):
                    results = idx.search("keyword", top_k=5)
                    assert isinstance(results, list)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []


# ── RRF merge tests ───────────────────────────────────────────────────────────


class TestRRFMerge:
    def test_path_in_both_lists_ranks_higher(self):
        vector = [("/a.md", 0.9), ("/b.md", 0.5)]
        bm25 = [("/b.md", 8.0), ("/c.md", 3.0)]
        merged = _rrf_merge(vector, bm25, k=60)
        paths = [p for p, _ in merged]
        # /b.md appears in both → should have higher RRF score than /c.md (bm25 only)
        assert paths.index("/b.md") < paths.index("/c.md")

    def test_path_only_in_one_list(self):
        vector = [("/a.md", 0.9)]
        bm25 = [("/b.md", 8.0)]
        merged = _rrf_merge(vector, bm25, k=60)
        assert len(merged) == 2

    def test_empty_bm25(self):
        vector = [("/a.md", 0.9), ("/b.md", 0.5)]
        merged = _rrf_merge(vector, [], k=60)
        paths = [p for p, _ in merged]
        assert paths == ["/a.md", "/b.md"]

    def test_empty_vector(self):
        bm25 = [("/b.md", 8.0), ("/c.md", 3.0)]
        merged = _rrf_merge([], bm25, k=60)
        paths = [p for p, _ in merged]
        assert paths == ["/b.md", "/c.md"]

    def test_scores_are_positive(self):
        vector = [("/a.md", 0.9)]
        bm25 = [("/a.md", 5.0), ("/b.md", 2.0)]
        merged = _rrf_merge(vector, bm25, k=60)
        assert all(s > 0 for _, s in merged)


# ── Path constraint tests ─────────────────────────────────────────────────────


class TestPathConstraint:
    BASE = "/mem"

    def test_sibling_nodes_both_included(self):
        ranked = [
            (f"{self.BASE}/01-japan/001.md", 0.9),
            (f"{self.BASE}/02-work/001.md", 0.7),
        ]
        result = _apply_path_constraint(ranked, set())
        assert len(result) == 2

    def test_leaf_excludes_parent_summary(self):
        leaf = f"{self.BASE}/01-japan/001.md"
        parent_summary = f"{self.BASE}/01-japan/_summary.md"
        # Leaf scores higher — should be selected, parent excluded
        ranked = [(leaf, 0.9), (parent_summary, 0.5)]
        result = _apply_path_constraint(ranked, set())
        assert leaf in result
        assert parent_summary not in result

    def test_parent_summary_excludes_leaf(self):
        leaf = f"{self.BASE}/01-japan/001.md"
        parent_summary = f"{self.BASE}/01-japan/_summary.md"
        # Summary scores higher — selected, leaf excluded
        ranked = [(parent_summary, 0.9), (leaf, 0.5)]
        result = _apply_path_constraint(ranked, set())
        assert parent_summary in result
        assert leaf not in result

    def test_fresh_paths_excluded_from_constraint(self):
        fresh = f"{self.BASE}/_fresh/001.md"
        normal = f"{self.BASE}/01-japan/001.md"
        ranked = [(fresh, 1.0), (normal, 0.8)]
        result = _apply_path_constraint(ranked, {fresh})
        # fresh is excluded from constraint processing — only normal appears
        assert normal in result
        assert fresh not in result

    def test_deep_ancestor_exclusion(self):
        depth2 = f"{self.BASE}/06-group/_summary.md"
        depth1 = f"{self.BASE}/06-group/01-japan/_summary.md"
        leaf = f"{self.BASE}/06-group/01-japan/001.md"
        # depth2 selected first → depth1 and leaf excluded
        ranked = [(depth2, 0.9), (depth1, 0.8), (leaf, 0.7)]
        result = _apply_path_constraint(ranked, set())
        assert result == [depth2]
