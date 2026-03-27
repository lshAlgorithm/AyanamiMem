"""Cross-session persistence tests for the Markdown memory backends.

These tests verify that memories written by one ``MarkdownTextMemory`` /
``MarkdownTreeTextMemory`` instance can be read back by a **completely
separate** instance (simulating a process restart).  Each test creates
*two independent* instances that share only a filesystem directory —
there is no in-memory state transfer.
"""

import os

from unittest.mock import MagicMock, patch

from memos.configs.memory import MarkdownTextMemoryConfig
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
from memos.memories.textual.markdown import MarkdownTextMemory
from memos.memories.textual.markdown_tree import MarkdownTreeTextMemory


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _fake_embed(texts):
    """Deterministic mock embedder — same text → same vector."""
    vecs = []
    for t in texts:
        h = hash(t) % 10000
        vec = [float(h % 7), float(h % 11), float(h % 13), float(h % 17)]
        norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
        vecs.append([x / norm for x in vec])
    return vecs


def _make_embedder():
    embedder = MagicMock()
    embedder.embed.side_effect = _fake_embed
    return embedder


def _make_config(tmp_path):
    return MarkdownTextMemoryConfig(
        memory_dir=str(tmp_path / "shared_store"),
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


def _new_instance(config, cls=MarkdownTextMemory):
    """Create a completely fresh instance — no state carried over."""
    with (
        patch("memos.memories.textual.markdown.LLMFactory") as llm_f,
        patch("memos.memories.textual.markdown.EmbedderFactory") as emb_f,
    ):
        llm_f.from_config.return_value = MagicMock()
        emb_f.from_config.return_value = _make_embedder()
        return cls(config)


def _item(key, body="", **kw):
    return TextualMemoryItem(
        memory=body or f"Body for {key}",
        metadata=TextualMemoryMetadata(key=key, source="conversation", **kw),
    )


# ── Flat markdown: cross-session CRUD ─────────────────────────────────────────


class TestFlatCrossSession:
    """Two independent MarkdownTextMemory instances sharing a directory."""

    def test_session2_reads_session1_memories(self, tmp_path):
        config = _make_config(tmp_path)

        # Session 1: add + dump
        s1 = _new_instance(config)
        s1.add([_item("fruit", "Apples are healthy"), _item("sport", "Running is good")])
        s1.dump(config.memory_dir)

        # Session 2: fresh instance, load from same dir
        s2 = _new_instance(config)
        assert s2.get_all() == []  # empty before load
        s2.load(config.memory_dir)
        assert len(s2.get_all()) == 2

        keys = {m.metadata.key for m in s2.get_all()}
        assert keys == {"fruit", "sport"}

    def test_session2_can_search_session1_memories(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config)
        s1.add(
            [
                _item("healthy food", "Apples are rich in fiber"),
                _item("space travel", "Mars missions in 2030"),
            ]
        )
        s1.dump(config.memory_dir)

        s2 = _new_instance(config)
        s2.load(config.memory_dir)
        results = s2.search("nutritious fruit", top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, TextualMemoryItem) for r in results)

    def test_session2_can_update_session1_memory(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config)
        ids = s1.add([_item("old key", "old body")])
        s1.dump(config.memory_dir)
        mid = ids[0]

        s2 = _new_instance(config)
        s2.load(config.memory_dir)
        s2.update(mid, _item("new key", "new body"))
        s2.dump(config.memory_dir)

        # Session 3: verify the update persisted
        s3 = _new_instance(config)
        s3.load(config.memory_dir)
        result = s3.get(mid)
        assert result.metadata.key == "new key"
        assert result.memory == "new body"

    def test_session2_can_delete_session1_memory(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config)
        ids = s1.add([_item("keep"), _item("remove")])
        s1.dump(config.memory_dir)

        s2 = _new_instance(config)
        s2.load(config.memory_dir)
        s2.delete([ids[1]])
        s2.dump(config.memory_dir)

        s3 = _new_instance(config)
        s3.load(config.memory_dir)
        assert len(s3.get_all()) == 1
        assert s3.get_all()[0].metadata.key == "keep"

    def test_three_sessions_accumulate(self, tmp_path):
        """Three sessions each add memories — the third sees all."""
        config = _make_config(tmp_path)

        s1 = _new_instance(config)
        s1.add([_item("session1")])
        s1.dump(config.memory_dir)

        s2 = _new_instance(config)
        s2.load(config.memory_dir)
        s2.add([_item("session2")])
        s2.dump(config.memory_dir)

        s3 = _new_instance(config)
        s3.load(config.memory_dir)
        s3.add([_item("session3")])
        s3.dump(config.memory_dir)

        s4 = _new_instance(config)
        s4.load(config.memory_dir)
        keys = {m.metadata.key for m in s4.get_all()}
        assert keys == {"session1", "session2", "session3"}

    def test_edges_persist_across_sessions(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config)
        ids = s1.add([_item("src"), _item("tgt")])
        s1.add_edge(ids[0], ids[1], "RELATED")
        s1.dump(config.memory_dir)

        s2 = _new_instance(config)
        s2.load(config.memory_dir)
        edges = s2.get_edges(ids[0])
        assert len(edges) == 1
        assert edges[0]["type"] == "RELATED"
        assert edges[0]["target"] == ids[1]

    def test_md_files_are_human_readable(self, tmp_path):
        """Verify the .md files on disk contain readable YAML + body."""
        config = _make_config(tmp_path)

        s1 = _new_instance(config)
        ids = s1.add([_item("my key", "Hello world content", tags=["test", "demo"])])
        s1.dump(config.memory_dir)

        fpath = os.path.join(config.memory_dir, f"{ids[0]}.md")
        with open(fpath, encoding="utf-8") as f:
            content = f.read()

        assert "---" in content  # YAML frontmatter
        assert "my key" in content  # key in frontmatter
        assert "Hello world content" in content  # body
        assert "test" in content  # tag
        assert "demo" in content  # tag


# ── Tree: cross-session tree structure ────────────────────────────────────────


class TestTreeCrossSession:
    """Two independent MarkdownTreeTextMemory instances sharing a directory."""

    def test_tree_structure_persists(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        root_ids = s1.add([_item("Animals")])
        root = root_ids[0]
        dog = s1.add_child(root, _item("Dogs"))
        cat = s1.add_child(root, _item("Cats"))
        poodle = s1.add_child(dog, _item("Poodle"))
        s1.dump(config.memory_dir)

        # Session 2
        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)

        assert s2.get_parent(dog) == root
        assert s2.get_parent(cat) == root
        assert s2.get_parent(poodle) == dog
        assert set(s2.get_children(root)) == {dog, cat}
        assert s2.get_children(dog) == [poodle]
        assert s2.get_depth(poodle) == 2

    def test_path_to_root_persists(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        r = s1.add([_item("R")])[0]
        a = s1.add_child(r, _item("A"))
        b = s1.add_child(a, _item("B"))
        c = s1.add_child(b, _item("C"))
        s1.dump(config.memory_dir)

        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)
        path = s2.get_path_to_root(c)
        assert path == [c, b, a, r]

    def test_subtree_persists(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        r = s1.add([_item("R")])[0]
        a = s1.add_child(r, _item("A"))
        s1.add_child(r, _item("B"))  # sibling of A
        a1 = s1.add_child(a, _item("A1"))
        s1.dump(config.memory_dir)

        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)
        subtree = s2.get_subtree(a)
        assert set(subtree) == {a, a1}

    def test_session2_extends_session1_tree(self, tmp_path):
        """Session 1 creates root + child; Session 2 adds grandchild."""
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        r = s1.add([_item("Root")])[0]
        child = s1.add_child(r, _item("Child"))
        s1.dump(config.memory_dir)

        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)
        grandchild = s2.add_child(child, _item("Grandchild"))
        s2.dump(config.memory_dir)

        # Session 3 sees the full 3-level tree
        s3 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s3.load(config.memory_dir)
        assert s3.get_parent(grandchild) == child
        assert s3.get_parent(child) == r
        assert s3.get_depth(grandchild) == 2
        assert len(s3.get_all()) == 3

    def test_move_node_persists(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        ids = s1.add([_item("A"), _item("B")])
        a, b = ids
        child = s1.add_child(a, _item("C"))
        s1.dump(config.memory_dir)

        # Session 2: move C from A to B
        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)
        s2.move_node(child, b)
        s2.dump(config.memory_dir)

        # Session 3: verify
        s3 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s3.load(config.memory_dir)
        assert s3.get_parent(child) == b
        assert child not in s3.get_children(a)
        assert child in s3.get_children(b)

    def test_delete_middle_reparents_across_sessions(self, tmp_path):
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        gp = s1.add([_item("GP")])[0]
        parent = s1.add_child(gp, _item("Parent"))
        c1 = s1.add_child(parent, _item("C1"))
        c2 = s1.add_child(parent, _item("C2"))
        s1.dump(config.memory_dir)

        # Session 2: delete parent
        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)
        s2.delete([parent])
        s2.dump(config.memory_dir)

        # Session 3: children are now under grandparent
        s3 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s3.load(config.memory_dir)
        assert s3.get_parent(c1) == gp
        assert s3.get_parent(c2) == gp
        assert set(s3.get_children(gp)) == {c1, c2}
        assert len(s3.get_all()) == 3  # GP, C1, C2 (Parent deleted)

    def test_index_md_regenerated_on_load(self, tmp_path):
        """After reload, _index.md is regenerated correctly."""
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        r = s1.add([_item("Root")])[0]
        s1.add_child(r, _item("Leaf"))
        s1.dump(config.memory_dir)

        # Delete _index.md manually
        idx = os.path.join(config.memory_dir, "_index.md")
        if os.path.isfile(idx):
            os.remove(idx)

        # Session 2 regenerates it on load
        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)
        assert os.path.isfile(idx)
        with open(idx, encoding="utf-8") as f:
            content = f.read()
        assert "Root" in content
        assert "Leaf" in content

    def test_search_across_sessions(self, tmp_path):
        """Memories added in session 1 are searchable in session 2."""
        config = _make_config(tmp_path)

        s1 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s1.add(
            [
                _item("healthy eating", "Apples are rich in fiber and vitamins"),
                _item("space exploration", "Mars rover landed successfully"),
                _item("python programming", "Use list comprehensions for cleaner code"),
            ]
        )
        s1.dump(config.memory_dir)

        s2 = _new_instance(config, cls=MarkdownTreeTextMemory)
        s2.load(config.memory_dir)
        results = s2.search("nutritious food", top_k=2)
        assert len(results) > 0
        # The search should find something (exact ranking depends on mock embedder)
        assert all(isinstance(r, TextualMemoryItem) for r in results)


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestCrossSessionEdgeCases:
    def test_load_empty_dir(self, tmp_path):
        config = _make_config(tmp_path)
        os.makedirs(config.memory_dir, exist_ok=True)

        s1 = _new_instance(config)
        s1.load(config.memory_dir)
        assert s1.get_all() == []

    def test_load_dir_with_non_md_files(self, tmp_path):
        config = _make_config(tmp_path)
        os.makedirs(config.memory_dir, exist_ok=True)
        # Create some non-md files
        with open(os.path.join(config.memory_dir, "notes.txt"), "w") as f:
            f.write("random text")
        with open(os.path.join(config.memory_dir, "data.json"), "w") as f:
            f.write("{}")

        s1 = _new_instance(config)
        s1.load(config.memory_dir)
        assert s1.get_all() == []

    def test_concurrent_writes_last_dump_wins(self, tmp_path):
        """Two instances add different items, last dump wins for conflict."""
        config = _make_config(tmp_path)
        os.makedirs(config.memory_dir, exist_ok=True)

        s1 = _new_instance(config)
        s1.add([_item("from_s1")])
        s1.dump(config.memory_dir)

        # s2 starts fresh (doesn't load s1's data), adds its own
        s2 = _new_instance(config)
        s2.add([_item("from_s2")])
        s2.dump(config.memory_dir)

        # s3 loads: should see BOTH because dump writes per-id files (no overwrite)
        s3 = _new_instance(config)
        s3.load(config.memory_dir)
        keys = {m.metadata.key for m in s3.get_all()}
        assert keys == {"from_s1", "from_s2"}

    def test_same_id_different_sessions_update(self, tmp_path):
        """Updating same memory in session 2 overwrites the .md file."""
        config = _make_config(tmp_path)

        s1 = _new_instance(config)
        ids = s1.add([_item("version1", "body v1")])
        s1.dump(config.memory_dir)
        mid = ids[0]

        s2 = _new_instance(config)
        s2.load(config.memory_dir)
        s2.update(mid, _item("version2", "body v2"))
        s2.dump(config.memory_dir)

        # Read raw file
        fpath = os.path.join(config.memory_dir, f"{mid}.md")
        with open(fpath, encoding="utf-8") as f:
            raw = f.read()
        assert "version2" in raw
        assert "body v2" in raw
        assert "version1" not in raw
        assert "body v1" not in raw
