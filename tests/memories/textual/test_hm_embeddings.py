"""Tests for hierarchical_markdown.embeddings — async-safe embedding index."""

import os
import threading

from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingIndex


class TestBasicOperations:
    def test_read_empty(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        assert idx.read() == {}

    def test_update_and_read(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"file1.md": [0.1, 0.2, 0.3]})
        result = idx.read()
        assert "file1.md" in result
        assert result["file1.md"] == [0.1, 0.2, 0.3]

    def test_update_merges(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0, 2.0]})
        idx.update({"b.md": [3.0, 4.0]})
        result = idx.read()
        assert set(result.keys()) == {"a.md", "b.md"}

    def test_update_overwrites_existing_key(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0]})
        idx.update({"a.md": [9.0]})
        assert idx.read()["a.md"] == [9.0]

    def test_remove(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0], "b.md": [2.0], "c.md": [3.0]})
        idx.remove(["b.md"])
        result = idx.read()
        assert set(result.keys()) == {"a.md", "c.md"}

    def test_remove_nonexistent_key_safe(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0]})
        idx.remove(["nonexistent.md"])
        assert idx.read() == {"a.md": [1.0]}


class TestVersioning:
    def test_version_starts_at_zero(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        assert idx.version() == 0

    def test_version_increments(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0]})
        assert idx.version() == 1
        idx.update({"b.md": [2.0]})
        assert idx.version() == 2

    def test_remove_increments_version(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0]})
        v1 = idx.version()
        idx.remove(["a.md"])
        assert idx.version() == v1 + 1


class TestModelTracking:
    def test_model_stored(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0]}, model="all-MiniLM-L6-v2")
        meta = idx.read_meta()
        assert meta["_model"] == "all-MiniLM-L6-v2"

    def test_model_not_in_read(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0]}, model="test-model")
        result = idx.read()
        assert "_model" not in result


class TestAtomicWrite:
    def test_no_tmp_file_left(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"a.md": [1.0]})
        files = os.listdir(str(tmp_path))
        assert not any(f.endswith(".tmp") for f in files)

    def test_corrupt_file_returns_empty(self, tmp_path):
        idx = EmbeddingIndex(str(tmp_path))
        fpath = os.path.join(str(tmp_path), "_embeddings.json")
        with open(fpath, "w") as f:
            f.write("NOT VALID JSON{{{")
        assert idx.read() == {}


class TestConcurrency:
    def test_concurrent_writes(self, tmp_path):
        """Multiple threads writing to the same index should not corrupt it."""
        idx = EmbeddingIndex(str(tmp_path))
        errors = []

        def writer(thread_id):
            try:
                for i in range(20):
                    idx.update({f"t{thread_id}-{i}.md": [float(thread_id), float(i)]})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"

        result = idx.read()
        # Each thread wrote 20 entries
        assert len(result) == 80

    def test_concurrent_read_write(self, tmp_path):
        """Reads during writes should not crash or return corrupt data."""
        idx = EmbeddingIndex(str(tmp_path))
        idx.update({"seed.md": [0.0]})
        errors = []

        def writer():
            try:
                for i in range(30):
                    idx.update({f"w-{i}.md": [float(i)]})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(30):
                    data = idx.read()
                    # Must always be a valid dict
                    assert isinstance(data, dict)
            except Exception as e:
                errors.append(e)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert errors == [], f"Errors during concurrent read/write: {errors}"
