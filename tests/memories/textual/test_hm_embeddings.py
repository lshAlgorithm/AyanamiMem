"""Tests for hierarchical_markdown.embeddings — in-memory EmbeddingCache."""

import threading

import numpy as np
import pytest

from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingCache


@pytest.fixture()
def cache():
    return EmbeddingCache()


class TestBasicOperations:
    def test_empty_on_init(self, cache):
        assert len(cache) == 0

    def test_update_and_len(self, cache):
        cache.update({"/a/b/c.md": [0.1, 0.2, 0.3]})
        assert len(cache) == 1

    def test_update_list_or_array(self, cache):
        cache.update({"/a/list.md": [1.0, 2.0, 3.0]})
        cache.update({"/a/arr.md": np.array([4.0, 5.0, 6.0])})
        assert len(cache) == 2
        assert isinstance(cache.get("/a/list.md"), np.ndarray)
        assert isinstance(cache.get("/a/arr.md"), np.ndarray)

    def test_get_missing_returns_none(self, cache):
        assert cache.get("/nonexistent") is None

    def test_update_overwrites(self, cache):
        cache.update({"/a/b.md": [1.0]})
        cache.update({"/a/b.md": [9.0]})
        assert cache.get("/a/b.md")[0] == pytest.approx(9.0)

    def test_remove(self, cache):
        cache.update({"/a.md": [1.0], "/b.md": [2.0], "/c.md": [3.0]})
        cache.remove(["/b.md"])
        assert cache.get("/b.md") is None
        assert cache.get("/a.md") is not None
        assert cache.get("/c.md") is not None

    def test_remove_missing_is_safe(self, cache):
        cache.update({"/a.md": [1.0]})
        cache.remove(["/nonexistent.md"])  # must not raise
        assert len(cache) == 1

    def test_clear(self, cache):
        cache.update({"/a.md": [1.0], "/b.md": [2.0]})
        cache.clear()
        assert len(cache) == 0


class TestChildrenOf:
    def test_direct_children_only(self, cache):
        cache.update(
            {
                "/mem/06-japan": [0.1, 0.2],
                "/mem/06-japan/01-planning": [0.3, 0.4],
                "/mem/06-japan/01-planning/001.md": [0.5, 0.6],
                "/mem/07-work": [0.7, 0.8],
            }
        )
        children = cache.children_of("/mem")
        assert set(children.keys()) == {"06-japan", "07-work"}

    def test_children_of_subdir(self, cache):
        cache.update(
            {
                "/mem/06-japan/01-planning": [0.1, 0.2],
                "/mem/06-japan/02-restaurants": [0.3, 0.4],
                "/mem/06-japan/01-planning/001.md": [0.5, 0.6],
            }
        )
        children = cache.children_of("/mem/06-japan")
        assert set(children.keys()) == {"01-planning", "02-restaurants"}

    def test_children_of_empty_dir(self, cache):
        assert cache.children_of("/mem/nothing") == {}

    def test_children_returns_ndarray(self, cache):
        cache.update({"/mem/dir": [1.0, 2.0, 3.0]})
        children = cache.children_of("/mem")
        assert isinstance(children["dir"], np.ndarray)

    def test_trailing_slash_handled(self, cache):
        cache.update({"/mem/a": [0.1], "/mem/b": [0.2]})
        c1 = cache.children_of("/mem")
        c2 = cache.children_of("/mem/")
        assert set(c1.keys()) == set(c2.keys())


class TestRenamePrefix:
    def test_rename_exact_path(self, cache):
        cache.update({"/mem/old-dir": [0.1, 0.2]})
        cache.rename_prefix("/mem/old-dir", "/mem/new-dir")
        assert cache.get("/mem/old-dir") is None
        assert cache.get("/mem/new-dir") is not None

    def test_rename_dir_and_all_descendants(self, cache):
        cache.update(
            {
                "/mem/old": [0.1],
                "/mem/old/child": [0.2],
                "/mem/old/child/001.md": [0.3],
            }
        )
        cache.rename_prefix("/mem/old", "/mem/new")
        assert cache.get("/mem/old") is None
        assert cache.get("/mem/old/child") is None
        assert cache.get("/mem/old/child/001.md") is None
        assert cache.get("/mem/new") is not None
        assert cache.get("/mem/new/child") is not None
        assert cache.get("/mem/new/child/001.md") is not None

    def test_rename_does_not_affect_sibling(self, cache):
        cache.update({"/mem/old": [0.1], "/mem/older": [0.2]})
        cache.rename_prefix("/mem/old", "/mem/new")
        # "/mem/older" does NOT start with "/mem/old/" — must be untouched
        assert cache.get("/mem/older") is not None

    def test_rename_file(self, cache):
        cache.update({"/mem/_fresh/001.md": [0.5, 0.6]})
        cache.rename_prefix("/mem/_fresh/001.md", "/mem/01-japan/001.md")
        assert cache.get("/mem/_fresh/001.md") is None
        assert cache.get("/mem/01-japan/001.md") is not None


class TestConcurrency:
    def test_concurrent_writes(self, cache):
        errors: list[Exception] = []

        def writer(tid: int) -> None:
            try:
                for i in range(30):
                    cache.update({f"/mem/t{tid}-{i}": [float(tid), float(i)]})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(cache) == 4 * 30

    def test_concurrent_read_write(self, cache):
        cache.update({"/seed": [0.0]})
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    cache.update({f"/w-{i}": [float(i)]})
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(50):
                    result = cache.children_of("/")
                    assert isinstance(result, dict)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
