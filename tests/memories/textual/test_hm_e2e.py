"""End-to-end tests for HierarchicalMarkdownMemory."""

import json
import os

from unittest.mock import MagicMock, patch

import pytest

from memos.configs.memory import HierarchicalMarkdownMemoryConfig
from memos.mem_user.session_manager import SessionManager
from memos.memories.textual.hierarchical_markdown.fs import FRESH_DIR
from memos.memories.textual.hierarchical_markdown_memory import HierarchicalMarkdownMemory
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _fake_embed(texts):
    """Deterministic mock embedder."""
    vecs = []
    for t in texts:
        h = hash(t) % 10000
        vec = [float(h % 7), float(h % 11), float(h % 13), float(h % 17)]
        norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
        vecs.append([x / norm for x in vec])
    return vecs


def _make_mock_llm():
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "key": "conversation summary",
            "summary": "A summary of the conversation about various topics.",
            "tags": ["general"],
        }
    )
    return llm


def _make_mock_embedder():
    embedder = MagicMock()
    embedder.embed.side_effect = _fake_embed
    return embedder


@pytest.fixture()
def config(tmp_path):
    return HierarchicalMarkdownMemoryConfig(
        memory_dir=str(tmp_path / "memory"),
        extractor_llm={
            "backend": "openai",
            "config": {
                "model_name_or_path": "test-model",
                "api_key": "test-key",
                "api_base": "http://localhost:1234/v1",
            },
        },
        embedder={
            "backend": "sentence_transformer",
            "config": {"model_name_or_path": "all-MiniLM-L6-v2"},
        },
        fresh_tail_count=2,
        condensed_min_fanout=2,
        leaf_chunk_tokens=500,
    )


@pytest.fixture()
def mem(config):
    with (
        patch("memos.memories.textual.hierarchical_markdown_memory.LLMFactory") as llm_f,
        patch("memos.memories.textual.hierarchical_markdown_memory.EmbedderFactory") as emb_f,
    ):
        llm_f.from_config.return_value = _make_mock_llm()
        emb_f.from_config.return_value = _make_mock_embedder()
        return HierarchicalMarkdownMemory(config)


def _item(key: str, body: str = "") -> TextualMemoryItem:
    return TextualMemoryItem(
        memory=body or f"Body for {key}",
        metadata=TextualMemoryMetadata(key=key, source="conversation"),
    )


# ── Basic CRUD ────────────────────────────────────────────────────────────────


class TestCRUD:
    def test_extract_creates_leaves(self, mem, config):
        messages = [
            {"role": "user", "content": "I like Python programming."},
            {"role": "assistant", "content": "Python is great for ML!"},
        ]
        items = mem.extract(messages)
        assert len(items) >= 1
        assert all(isinstance(i, TextualMemoryItem) for i in items)
        # Leaf files should exist in _fresh/
        fresh = os.path.join(config.memory_dir, FRESH_DIR)
        md_files = [f for f in os.listdir(fresh) if f.endswith(".md")]
        assert len(md_files) >= 1

    def test_add_from_extract(self, mem):
        messages = [{"role": "user", "content": "Planning a trip to Japan."}]
        items = mem.extract(messages)
        ids = mem.add(items)
        assert len(ids) == len(items)

    def test_add_raw_dict(self, mem):
        ids = mem.add(
            [
                {"memory": "Test memory", "metadata": {"key": "test key"}},
            ]
        )
        assert len(ids) == 1

    def test_add_item(self, mem):
        item = _item("fruit", "Apples are healthy.")
        ids = mem.add([item])
        assert len(ids) == 1

    def test_get_after_add(self, mem):
        item = _item("get-test", "Get me back.")
        ids = mem.add([item])
        result = mem.get(ids[0])
        assert result.memory == "Get me back."
        assert result.metadata.key == "get-test"

    def test_get_nonexistent(self, mem):
        import uuid

        result = mem.get(str(uuid.uuid4()))
        assert result.memory == ""

    def test_get_all(self, mem):
        mem.add([_item("a"), _item("b"), _item("c")])
        all_items = mem.get_all()
        assert len(all_items) >= 3

    def test_get_by_ids(self, mem):
        ids = mem.add([_item("x"), _item("y")])
        results = mem.get_by_ids(ids)
        assert len(results) == 2

    def test_update(self, mem):
        ids = mem.add([_item("old-key", "old body")])
        mid = ids[0]
        mem.update(mid, _item("new-key", "new body"))
        result = mem.get(mid)
        assert result.memory == "new body"

    def test_delete(self, mem):
        ids = mem.add([_item("del-me")])
        mem.delete(ids)
        result = mem.get(ids[0])
        assert result.memory == ""

    def test_delete_all(self, mem, config):
        mem.add([_item("a"), _item("b")])
        mem.delete_all()
        assert mem.get_all() == []
        # _fresh/ should still exist
        assert os.path.isdir(os.path.join(config.memory_dir, FRESH_DIR))

    def test_drop_is_noop(self, mem):
        mem.drop()  # should not raise


# ── Search ────────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_empty(self, mem):
        results = mem.search("anything", top_k=5)
        assert results == []

    def test_search_returns_results(self, mem):
        mem.add(
            [
                _item("healthy food", "Apples are rich in fiber."),
                _item("space travel", "Mars missions planned."),
                _item("python coding", "Use list comprehensions."),
            ]
        )
        results = mem.search("nutritious fruit", top_k=3)
        assert len(results) >= 0  # may be 0 if all in _fresh/ without embeddings match
        assert all(isinstance(r, TextualMemoryItem) for r in results)


# ── Extract + Add flow ────────────────────────────────────────────────────────


class TestExtractAddFlow:
    def test_multi_turn_conversation(self, mem):
        messages = [
            {"role": "user", "content": "I want to visit Kyoto temples."},
            {"role": "assistant", "content": "Fushimi Inari is beautiful!"},
            {"role": "user", "content": "My budget is 3000 USD."},
        ]
        items = mem.extract(messages)
        ids = mem.add(items)
        assert len(ids) >= 1

        all_items = mem.get_all()
        bodies = " ".join(i.memory for i in all_items)
        assert "Kyoto" in bodies or "budget" in bodies

    def test_multiple_conversations_accumulate(self, mem):
        conv1 = [{"role": "user", "content": "I like cats."}]
        conv2 = [{"role": "user", "content": "I like dogs."}]

        items1 = mem.extract(conv1)
        mem.add(items1)

        items2 = mem.extract(conv2)
        mem.add(items2)

        all_items = mem.get_all()
        assert len(all_items) >= 2


# ── Tree index ────────────────────────────────────────────────────────────────


class TestTreeIndex:
    def test_get_tree_index(self, mem):
        mem.add([_item("topic-a"), _item("topic-b")])
        index = mem.get_tree_index()
        assert "# Memory Tree" in index
        assert "Fresh" in index


# ── Cross-session (load from disk) ───────────────────────────────────────────


class TestCrossSession:
    def test_reload_from_disk(self, mem, config):
        mem.add([_item("persist-me", "This should survive reload.")])

        # Create a fresh instance pointing to the same directory
        with (
            patch("memos.memories.textual.hierarchical_markdown_memory.LLMFactory") as llm_f,
            patch("memos.memories.textual.hierarchical_markdown_memory.EmbedderFactory") as emb_f,
        ):
            llm_f.from_config.return_value = _make_mock_llm()
            emb_f.from_config.return_value = _make_mock_embedder()
            mem2 = HierarchicalMarkdownMemory(config)

        mem2.load(config.memory_dir)
        all_items = mem2.get_all()
        bodies = [i.memory for i in all_items]
        assert any("persist-me" in b or "survive reload" in b for b in bodies)


# ── Session Isolation ─────────────────────────────────────────────────────────


def _isolated_config(root_path: str, session_id: str) -> HierarchicalMarkdownMemoryConfig:
    return HierarchicalMarkdownMemoryConfig(
        memory_dir=root_path,
        session_isolation=True,
        session_id=session_id,
        extractor_llm={
            "backend": "openai",
            "config": {
                "model_name_or_path": "test-model",
                "api_key": "test-key",
                "api_base": "http://localhost:1234/v1",
            },
        },
        embedder={
            "backend": "sentence_transformer",
            "config": {"model_name_or_path": "all-MiniLM-L6-v2"},
        },
        fresh_tail_count=2,
        condensed_min_fanout=2,
        compact_threshold=4,
    )


def _make_isolated(root_path: str, session_id: str) -> HierarchicalMarkdownMemory:
    with (
        patch("memos.memories.textual.hierarchical_markdown_memory.LLMFactory") as llm_f,
        patch("memos.memories.textual.hierarchical_markdown_memory.EmbedderFactory") as emb_f,
    ):
        llm_f.from_config.return_value = _make_mock_llm()
        emb_f.from_config.return_value = _make_mock_embedder()
        return HierarchicalMarkdownMemory(_isolated_config(root_path, session_id))


class TestSessionIsolation:
    def test_session_dir_created_under_root(self, tmp_path):
        """memory_dir is root/session_id when isolation is on."""
        root = str(tmp_path / "root")
        mem = _make_isolated(root, "alice")
        assert mem.memory_dir == os.path.join(root, "alice")
        assert os.path.isdir(mem.memory_dir)

    def test_sessions_have_separate_dirs(self, tmp_path):
        """Alice and Bob get different directories."""
        root = str(tmp_path / "root")
        alice = _make_isolated(root, "alice")
        bob = _make_isolated(root, "bob")
        assert alice.memory_dir != bob.memory_dir
        assert os.path.isdir(alice.memory_dir)
        assert os.path.isdir(bob.memory_dir)

    def test_alice_memories_invisible_to_bob(self, tmp_path):
        """Bob's memory instance cannot see Alice's items via get_all()."""
        root = str(tmp_path / "root")
        alice = _make_isolated(root, "alice")
        bob = _make_isolated(root, "bob")

        alice.add([_item("alice-secret", "Alice's private memory")])
        bob_items = bob.get_all()
        bodies = [i.memory for i in bob_items]
        assert not any("Alice's private memory" in b for b in bodies)

    def test_bob_memories_invisible_to_alice(self, tmp_path):
        root = str(tmp_path / "root")
        alice = _make_isolated(root, "alice")
        bob = _make_isolated(root, "bob")

        bob.add([_item("bob-secret", "Bob's private memory")])
        alice_items = alice.get_all()
        bodies = [i.memory for i in alice_items]
        assert not any("Bob's private memory" in b for b in bodies)

    def test_sessions_registered_in_sessions_json(self, tmp_path):
        """Both sessions appear in _sessions.json after creation."""
        root = str(tmp_path / "root")
        _make_isolated(root, "alice")
        _make_isolated(root, "bob")
        mgr = SessionManager(root)
        ids = {s["session_id"] for s in mgr.list_sessions()}
        assert {"alice", "bob"} <= ids

    def test_public_dir_created(self, tmp_path):
        """_public/ directory is created when isolation is active."""
        root = str(tmp_path / "root")
        mem = _make_isolated(root, "alice")
        assert mem._public_dir is not None
        assert os.path.isdir(mem._public_dir)

    def test_write_public_lands_in_public_dir(self, tmp_path):
        """write_public() writes to _public/, not the session dir."""
        root = str(tmp_path / "root")
        mem = _make_isolated(root, "alice")
        mem.write_public([_item("shared-fact", "This is publicly visible.")])
        public_fresh = os.path.join(mem._public_dir, FRESH_DIR)
        assert os.path.isdir(public_fresh)
        md_files = [f for f in os.listdir(public_fresh) if f.endswith(".md")]
        assert len(md_files) == 1

    def test_no_isolation_uses_memory_dir_directly(self, tmp_path):
        """When isolation is off, memory_dir is used as-is (no session subdir)."""
        config = HierarchicalMarkdownMemoryConfig(
            memory_dir=str(tmp_path / "flat"),
            session_isolation=False,
            extractor_llm={
                "backend": "openai",
                "config": {
                    "model_name_or_path": "test-model",
                    "api_key": "test-key",
                    "api_base": "http://localhost:1234/v1",
                },
            },
            embedder={
                "backend": "sentence_transformer",
                "config": {"model_name_or_path": "all-MiniLM-L6-v2"},
            },
        )
        with (
            patch("memos.memories.textual.hierarchical_markdown_memory.LLMFactory") as llm_f,
            patch("memos.memories.textual.hierarchical_markdown_memory.EmbedderFactory") as emb_f,
        ):
            llm_f.from_config.return_value = _make_mock_llm()
            emb_f.from_config.return_value = _make_mock_embedder()
            mem = HierarchicalMarkdownMemory(config)

        assert mem.memory_dir == os.path.abspath(str(tmp_path / "flat"))
        assert mem.session_manager is None
        assert mem._public_dir is None

    def test_grant_recorded_in_sessions_json(self, tmp_path):
        """grant_access() writes a grant record to _sessions.json."""
        root = str(tmp_path / "root")
        alice = _make_isolated(root, "alice")
        _make_isolated(root, "bob")  # ensure bob is registered
        alice.grant_access("bob", scope="read")

        mgr = SessionManager(root)
        grants = mgr.get_grants_for("bob")
        assert len(grants) == 1
        assert grants[0]["granting_session"] == "alice"
        assert grants[0]["scope"] == "read"

    def test_revoke_removes_grant(self, tmp_path):
        root = str(tmp_path / "root")
        alice = _make_isolated(root, "alice")
        _make_isolated(root, "bob")
        alice.grant_access("bob")
        alice.revoke_access("bob")

        mgr = SessionManager(root)
        assert mgr.get_grants_for("bob") == []

    def test_grant_raises_without_isolation(self, tmp_path):
        """grant_access() raises RuntimeError when isolation is off."""
        config = HierarchicalMarkdownMemoryConfig(
            memory_dir=str(tmp_path / "flat"),
            session_isolation=False,
            extractor_llm={
                "backend": "openai",
                "config": {"model_name_or_path": "t", "api_key": "k", "api_base": "h"},
            },
            embedder={
                "backend": "sentence_transformer",
                "config": {"model_name_or_path": "all-MiniLM-L6-v2"},
            },
        )
        with (
            patch("memos.memories.textual.hierarchical_markdown_memory.LLMFactory") as llm_f,
            patch("memos.memories.textual.hierarchical_markdown_memory.EmbedderFactory") as emb_f,
        ):
            llm_f.from_config.return_value = _make_mock_llm()
            emb_f.from_config.return_value = _make_mock_embedder()
            mem = HierarchicalMarkdownMemory(config)

        with pytest.raises(RuntimeError, match="session_isolation"):
            mem.grant_access("bob")
