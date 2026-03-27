"""Tests for the Markdown file-based textual memory backend."""

import json
import os
import uuid

from unittest.mock import MagicMock, patch

import pytest

from memos.configs.memory import MarkdownTextMemoryConfig
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
from memos.memories.textual.markdown import (
    MarkdownTextMemory,
    _parse_md,
    _serialize_md,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_embedder():
    """A mock embedder that returns deterministic 4-dim vectors."""
    embedder = MagicMock()

    # Make embed return a deterministic vector per text (hash-based for reproducibility)
    def _fake_embed(texts):
        vecs = []
        for t in texts:
            h = hash(t) % 10000
            vec = [float(h % 7), float(h % 11), float(h % 13), float(h % 17)]
            norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
            vecs.append([x / norm for x in vec])
        return vecs

    embedder.embed.side_effect = _fake_embed
    return embedder


@pytest.fixture()
def mock_llm():
    """A mock LLM that returns a canned extraction response."""
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "memory list": [
                {
                    "key": "Paris travel",
                    "value": "User plans to visit Paris next week.",
                    "tags": ["travel"],
                },
                {
                    "key": "Eiffel Tower",
                    "value": "User loves the Eiffel Tower.",
                    "tags": ["travel", "landmark"],
                },
            ]
        }
    )
    return llm


@pytest.fixture()
def config(tmp_path):
    """Build a valid MarkdownTextMemoryConfig pointing to a temp directory."""
    return MarkdownTextMemoryConfig(
        memory_dir=str(tmp_path / "md_store"),
        extractor_llm={
            "backend": "openai",
            "config": {
                "model_name_or_path": "gpt-4o-mini",
                "api_key": "test-key",
                "api_base": "http://localhost:1234/v1",
            },
        },
        embedder={
            "backend": "sentence_transformer",
            "config": {"model_name_or_path": "all-MiniLM-L6-v2"},
        },
    )


@pytest.fixture()
def memory(config, mock_embedder, mock_llm):
    """Return a MarkdownTextMemory with mocked LLM and embedder."""
    with (
        patch("memos.memories.textual.markdown.LLMFactory") as llm_factory,
        patch("memos.memories.textual.markdown.EmbedderFactory") as emb_factory,
    ):
        llm_factory.from_config.return_value = mock_llm
        emb_factory.from_config.return_value = mock_embedder
        mem = MarkdownTextMemory(config)
    return mem


def _make_item(key: str = "test key", body: str = "test body", **meta_kw) -> TextualMemoryItem:
    return TextualMemoryItem(
        memory=body,
        metadata=TextualMemoryMetadata(key=key, source="conversation", **meta_kw),
    )


# ── Serialization round-trip ──────────────────────────────────────────────────


class TestSerializationRoundTrip:
    def test_serialize_and_parse_basic(self):
        item = _make_item("fruit", "Apple is a fruit rich in fiber.")
        md = _serialize_md(item, edges=[])
        parsed_item, parsed_edges = _parse_md(md)

        assert parsed_item.id == item.id
        assert parsed_item.memory == item.memory
        assert parsed_item.metadata.key == "fruit"
        assert parsed_edges == []

    def test_serialize_and_parse_with_edges(self):
        item = _make_item("fruit", "Apple is a fruit.")
        target1 = str(uuid.uuid4())
        target2 = str(uuid.uuid4())
        edges = [
            {"type": "RELATED", "target": target1},
            {"type": "PARENT", "target": target2},
        ]
        md = _serialize_md(item, edges=edges)
        parsed_item, parsed_edges = _parse_md(md)

        assert parsed_item.id == item.id
        assert len(parsed_edges) == 2
        assert parsed_edges[0]["type"] == "RELATED"
        assert parsed_edges[0]["target"] == target1
        assert parsed_edges[1]["type"] == "PARENT"
        assert parsed_edges[1]["target"] == target2

    def test_serialize_and_parse_with_tags(self):
        item = _make_item("health", "Exercise is good.", tags=["health", "fitness"])
        md = _serialize_md(item)
        parsed_item, _ = _parse_md(md)
        assert parsed_item.metadata.tags == ["health", "fitness"]

    def test_parse_raises_on_missing_frontmatter(self):
        with pytest.raises(ValueError, match="Missing YAML frontmatter"):
            _parse_md("no frontmatter here\njust text")


# ── Init ──────────────────────────────────────────────────────────────────────


class TestInit:
    def test_init_creates_empty_index(self, memory):
        assert memory.get_all() == []
        assert memory._key_vecs is None

    def test_mode_defaults_to_sync(self, memory):
        assert memory.mode == "sync"


# ── CRUD ──────────────────────────────────────────────────────────────────────


class TestCRUD:
    def test_add_and_get(self, memory):
        item = _make_item("fruit", "Apple is rich in fiber.")
        ids = memory.add([item])
        assert len(ids) == 1

        result = memory.get(ids[0])
        assert result.memory == "Apple is rich in fiber."
        assert result.metadata.key == "fruit"

    def test_add_writes_md_file(self, memory, config):
        item = _make_item("travel", "Going to Paris.")
        ids = memory.add([item])

        fpath = os.path.join(config.memory_dir, f"{ids[0]}.md")
        assert os.path.isfile(fpath)
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert "travel" in content
        assert "Going to Paris." in content

    def test_add_dict(self, memory):
        d = {"memory": "Test from dict", "metadata": {"key": "dict test"}}
        ids = memory.add([d])
        assert len(ids) == 1
        result = memory.get(ids[0])
        assert result.memory == "Test from dict"

    def test_add_skips_duplicate(self, memory):
        item = _make_item("dup", "duplicate body")
        ids1 = memory.add([item])
        ids2 = memory.add([item])
        assert len(ids1) == 1
        assert ids2 == []
        assert len(memory.get_all()) == 1

    def test_update(self, memory):
        item = _make_item("old key", "old body")
        ids = memory.add([item])
        mid = ids[0]

        new_item = _make_item("new key", "new body")
        memory.update(mid, new_item)

        result = memory.get(mid)
        assert result.memory == "new body"
        assert result.metadata.key == "new key"

    def test_update_preserves_edges(self, memory):
        item = _make_item("key1", "body1")
        ids = memory.add([item])
        mid = ids[0]

        target_id = str(uuid.uuid4())
        memory.add_edge(mid, target_id, "RELATED")

        memory.update(mid, _make_item("key1-updated", "body1-updated"))
        edges = memory.get_edges(mid)
        assert len(edges) == 1
        assert edges[0]["target"] == target_id

    def test_get_nonexistent_returns_empty(self, memory):
        fake_id = str(uuid.uuid4())
        result = memory.get(fake_id)
        assert result.memory == ""

    def test_get_by_ids(self, memory):
        items = [_make_item("a", "body a"), _make_item("b", "body b")]
        ids = memory.add(items)
        results = memory.get_by_ids(ids)
        assert len(results) == 2
        bodies = {r.memory for r in results}
        assert bodies == {"body a", "body b"}

    def test_get_all(self, memory):
        memory.add([_make_item("x", "bx"), _make_item("y", "by"), _make_item("z", "bz")])
        assert len(memory.get_all()) == 3

    def test_delete(self, memory, config):
        item = _make_item("del", "delete me")
        ids = memory.add([item])
        mid = ids[0]
        fpath = os.path.join(config.memory_dir, f"{mid}.md")
        assert os.path.isfile(fpath)

        memory.delete([mid])
        assert memory.get(mid).memory == ""
        assert not os.path.isfile(fpath)

    def test_delete_all(self, memory, config):
        memory.add([_make_item("a", "a"), _make_item("b", "b")])
        assert len(memory.get_all()) == 2

        memory.delete_all()
        assert memory.get_all() == []

    def test_drop_is_noop(self, memory):
        memory.drop()  # should not raise


# ── Search ────────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_empty(self, memory):
        results = memory.search("anything", top_k=5)
        assert results == []

    def test_search_returns_results(self, memory):
        memory.add(
            [
                _make_item("healthy food", "Apples are healthy."),
                _make_item("space travel", "Mars missions planned for 2030."),
                _make_item("fruit nutrition", "Bananas are rich in potassium."),
            ]
        )
        results = memory.search("nutritious fruit", top_k=2)
        assert len(results) <= 2
        # Just verify we get TextualMemoryItem objects back
        assert all(isinstance(r, TextualMemoryItem) for r in results)

    def test_search_top_k_limit(self, memory):
        for i in range(10):
            memory.add([_make_item(f"key{i}", f"body{i}")])
        results = memory.search("key", top_k=3)
        assert len(results) == 3


# ── Edges ─────────────────────────────────────────────────────────────────────


class TestEdges:
    def test_add_and_get_edge(self, memory):
        item = _make_item("src", "source")
        ids = memory.add([item])
        src_id = ids[0]
        tgt_id = str(uuid.uuid4())

        memory.add_edge(src_id, tgt_id, "RELATED")
        edges = memory.get_edges(src_id)
        assert len(edges) == 1
        assert edges[0] == {"type": "RELATED", "target": tgt_id}

    def test_add_edge_no_duplicate(self, memory):
        item = _make_item("src", "source")
        ids = memory.add([item])
        src_id = ids[0]
        tgt_id = str(uuid.uuid4())

        memory.add_edge(src_id, tgt_id, "RELATED")
        memory.add_edge(src_id, tgt_id, "RELATED")
        assert len(memory.get_edges(src_id)) == 1

    def test_add_edge_different_types(self, memory):
        item = _make_item("src", "source")
        ids = memory.add([item])
        src_id = ids[0]
        tgt_id = str(uuid.uuid4())

        memory.add_edge(src_id, tgt_id, "RELATED")
        memory.add_edge(src_id, tgt_id, "PARENT")
        assert len(memory.get_edges(src_id)) == 2

    def test_remove_edge(self, memory):
        item = _make_item("src", "source")
        ids = memory.add([item])
        src_id = ids[0]
        tgt_id = str(uuid.uuid4())

        memory.add_edge(src_id, tgt_id, "RELATED")
        memory.remove_edge(src_id, tgt_id, "RELATED")
        assert memory.get_edges(src_id) == []

    def test_add_edge_nonexistent_source_raises(self, memory):
        with pytest.raises(ValueError, match="not found"):
            memory.add_edge(str(uuid.uuid4()), str(uuid.uuid4()), "RELATED")

    def test_edge_persisted_in_md_file(self, memory, config):
        item = _make_item("src", "source node")
        ids = memory.add([item])
        src_id = ids[0]
        tgt_id = str(uuid.uuid4())
        memory.add_edge(src_id, tgt_id, "PARENT")

        fpath = os.path.join(config.memory_dir, f"{src_id}.md")
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert "## Edges" in content
        assert f"PARENT: {tgt_id}" in content


# ── Extract ───────────────────────────────────────────────────────────────────


class TestExtract:
    def test_extract(self, memory, mock_llm):
        messages = [
            {"role": "user", "content": "I plan to visit Paris next week."},
            {"role": "assistant", "content": "Paris is wonderful!"},
        ]
        items = memory.extract(messages)
        assert len(items) == 2
        assert items[0].metadata.key == "Paris travel"
        assert items[0].memory == "User plans to visit Paris next week."
        assert "travel" in items[0].metadata.tags
        mock_llm.generate.assert_called_once()


# ── Load / Dump round-trip ────────────────────────────────────────────────────


class TestLoadDump:
    def test_dump_and_load(self, memory, config, mock_embedder):
        memory.add(
            [
                _make_item("k1", "body 1"),
                _make_item("k2", "body 2"),
            ]
        )
        dump_dir = config.memory_dir
        memory.dump(dump_dir)

        # Create a fresh instance and load
        with (
            patch("memos.memories.textual.markdown.LLMFactory") as llm_f,
            patch("memos.memories.textual.markdown.EmbedderFactory") as emb_f,
        ):
            llm_f.from_config.return_value = MagicMock()
            emb_f.from_config.return_value = mock_embedder
            mem2 = MarkdownTextMemory(config)

        mem2.load(dump_dir)
        assert len(mem2.get_all()) == 2
        bodies = {m.memory for m in mem2.get_all()}
        assert bodies == {"body 1", "body 2"}

    def test_load_nonexistent_dir(self, memory, caplog):
        memory.load("/nonexistent/path")
        assert "not found" in caplog.text.lower()

    def test_dump_and_load_with_edges(self, memory, config, mock_embedder):
        ids = memory.add([_make_item("e1", "edge source")])
        src_id = ids[0]
        tgt_id = str(uuid.uuid4())
        memory.add_edge(src_id, tgt_id, "RELATED")
        memory.dump(config.memory_dir)

        # Reload
        with (
            patch("memos.memories.textual.markdown.LLMFactory") as llm_f,
            patch("memos.memories.textual.markdown.EmbedderFactory") as emb_f,
        ):
            llm_f.from_config.return_value = MagicMock()
            emb_f.from_config.return_value = mock_embedder
            mem2 = MarkdownTextMemory(config)

        mem2.load(config.memory_dir)
        edges = mem2.get_edges(src_id)
        assert len(edges) == 1
        assert edges[0]["target"] == tgt_id
