"""Tests for the Markdown tree-structured textual memory backend."""

import os
import uuid

from unittest.mock import MagicMock, patch

import pytest

from memos.configs.memory import MarkdownTextMemoryConfig
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
from memos.memories.textual.markdown_tree import MarkdownTreeTextMemory


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_embedder():
    def _fake_embed(texts):
        vecs = []
        for t in texts:
            h = hash(t) % 10000
            vec = [float(h % 7), float(h % 11), float(h % 13), float(h % 17)]
            norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
            vecs.append([x / norm for x in vec])
        return vecs

    embedder = MagicMock()
    embedder.embed.side_effect = _fake_embed
    return embedder


@pytest.fixture()
def config(tmp_path):
    return MarkdownTextMemoryConfig(
        memory_dir=str(tmp_path / "tree_store"),
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
def tree(config, mock_embedder):
    with (
        patch("memos.memories.textual.markdown.LLMFactory") as llm_f,
        patch("memos.memories.textual.markdown.EmbedderFactory") as emb_f,
    ):
        llm_f.from_config.return_value = MagicMock()
        emb_f.from_config.return_value = mock_embedder
        mem = MarkdownTreeTextMemory(config)
    return mem


def _item(key: str, body: str = "") -> TextualMemoryItem:
    return TextualMemoryItem(
        memory=body or f"Body for {key}",
        metadata=TextualMemoryMetadata(key=key, source="conversation"),
    )


# ── Basic tree operations ─────────────────────────────────────────────────────


class TestTreeBasics:
    def test_add_child(self, tree):
        parent_ids = tree.add([_item("Animals")])
        child_id = tree.add_child(parent_ids[0], _item("Dogs"))

        assert tree.get_parent(child_id) == parent_ids[0]
        assert child_id in tree.get_children(parent_ids[0])

    def test_roots(self, tree):
        ids = tree.add([_item("Root A"), _item("Root B")])
        roots = tree.get_roots()
        assert set(ids) == set(roots)

    def test_add_child_to_nonexistent_parent_raises(self, tree):
        with pytest.raises(ValueError, match="not found"):
            tree.add_child(str(uuid.uuid4()), _item("orphan"))

    def test_nested_tree(self, tree):
        """Build a 3-level tree and verify structure."""
        root_ids = tree.add([_item("Science")])
        root = root_ids[0]
        physics = tree.add_child(root, _item("Physics"))
        biology = tree.add_child(root, _item("Biology"))
        quantum = tree.add_child(physics, _item("Quantum Mechanics"))

        assert tree.get_parent(physics) == root
        assert tree.get_parent(biology) == root
        assert tree.get_parent(quantum) == physics
        assert set(tree.get_children(root)) == {physics, biology}
        assert tree.get_children(physics) == [quantum]
        assert tree.get_children(quantum) == []


# ── Tree traversal ────────────────────────────────────────────────────────────


class TestTreeTraversal:
    @pytest.fixture(autouse=True)
    def _build_tree(self, tree):
        """Build a tree for traversal tests:

        Travel
        ├── Europe
        │   ├── Paris
        │   └── London
        └── Asia
            └── Tokyo
        """
        root_ids = tree.add([_item("Travel")])
        self.root = root_ids[0]
        self.europe = tree.add_child(self.root, _item("Europe"))
        self.asia = tree.add_child(self.root, _item("Asia"))
        self.paris = tree.add_child(self.europe, _item("Paris"))
        self.london = tree.add_child(self.europe, _item("London"))
        self.tokyo = tree.add_child(self.asia, _item("Tokyo"))

    def test_get_path_to_root(self, tree):
        path = tree.get_path_to_root(self.paris)
        assert path == [self.paris, self.europe, self.root]

    def test_get_subtree(self, tree):
        subtree = tree.get_subtree(self.europe)
        assert set(subtree) == {self.europe, self.paris, self.london}

    def test_get_subtree_root(self, tree):
        full = tree.get_subtree(self.root)
        assert len(full) == 6  # root + europe + asia + paris + london + tokyo

    def test_get_depth(self, tree):
        assert tree.get_depth(self.root) == 0
        assert tree.get_depth(self.europe) == 1
        assert tree.get_depth(self.paris) == 2

    def test_get_siblings(self, tree):
        siblings = tree.get_siblings(self.paris)
        assert siblings == [self.london]

    def test_get_siblings_of_root(self, tree):
        # root has no siblings (only one root)
        assert tree.get_siblings(self.root) == []

    def test_search_subtree(self, tree):
        results = tree.search_subtree(self.europe, "city", top_k=5)
        result_ids = {r.id for r in results}
        # Should only return items in the Europe subtree
        assert result_ids <= {self.europe, self.paris, self.london}


# ── Move / reparent ──────────────────────────────────────────────────────────


class TestMoveNode:
    def test_move_node(self, tree):
        root_ids = tree.add([_item("A"), _item("B")])
        root_a, root_b = root_ids
        child = tree.add_child(root_a, _item("C"))

        # C is under A
        assert tree.get_parent(child) == root_a

        # Move C under B
        tree.move_node(child, root_b)
        assert tree.get_parent(child) == root_b
        assert child not in tree.get_children(root_a)
        assert child in tree.get_children(root_b)

    def test_move_to_root(self, tree):
        root_ids = tree.add([_item("Parent")])
        child = tree.add_child(root_ids[0], _item("Child"))
        tree.move_node(child, None)
        assert tree.get_parent(child) is None
        assert child in tree.get_roots()

    def test_move_nonexistent_raises(self, tree):
        with pytest.raises(ValueError, match="not found"):
            tree.move_node(str(uuid.uuid4()), None)


# ── Delete with re-parenting ─────────────────────────────────────────────────


class TestDeleteReparent:
    def test_delete_middle_node_reparents_children(self, tree):
        """Deleting a middle node should re-parent its children to grandparent."""
        root_ids = tree.add([_item("Grandparent")])
        gp = root_ids[0]
        parent = tree.add_child(gp, _item("Parent"))
        child1 = tree.add_child(parent, _item("Child 1"))
        child2 = tree.add_child(parent, _item("Child 2"))

        tree.delete([parent])

        # Children should now be under grandparent
        assert tree.get_parent(child1) == gp
        assert tree.get_parent(child2) == gp
        assert set(tree.get_children(gp)) == {child1, child2}

    def test_delete_root_orphans_children(self, tree):
        """Deleting a root node should make its children new roots."""
        root_ids = tree.add([_item("Root")])
        root = root_ids[0]
        c1 = tree.add_child(root, _item("C1"))
        c2 = tree.add_child(root, _item("C2"))

        tree.delete([root])
        assert tree.get_parent(c1) is None
        assert tree.get_parent(c2) is None
        assert set(tree.get_roots()) == {c1, c2}


# ── _index.md generation ─────────────────────────────────────────────────────


class TestIndexFile:
    def test_index_file_created_on_add(self, tree, config):
        tree.add([_item("Root Node")])
        fpath = os.path.join(config.memory_dir, "_index.md")
        assert os.path.isfile(fpath)

        with open(fpath, encoding="utf-8") as f:
            content = f.read()
        assert "# Memory Tree Index" in content
        assert "Root Node" in content

    def test_index_reflects_tree_structure(self, tree, config):
        root_ids = tree.add([_item("Animals")])
        root = root_ids[0]
        tree.add_child(root, _item("Cats"))
        tree.add_child(root, _item("Dogs"))

        fpath = os.path.join(config.memory_dir, "_index.md")
        with open(fpath, encoding="utf-8") as f:
            content = f.read()

        # Root should be at indent 0, children at indent 1
        assert "- **Animals**" in content
        assert "  - **Cats**" in content
        assert "  - **Dogs**" in content

    def test_index_empty_after_delete_all(self, tree, config):
        tree.add([_item("X")])
        tree.delete_all()
        fpath = os.path.join(config.memory_dir, "_index.md")
        assert not os.path.isfile(fpath)


# ── Load / dump round-trip with tree ──────────────────────────────────────────


class TestTreeLoadDump:
    def test_dump_and_load_preserves_tree(self, tree, config, mock_embedder):
        root_ids = tree.add([_item("Root")])
        root = root_ids[0]
        child = tree.add_child(root, _item("Child"))
        grandchild = tree.add_child(child, _item("Grandchild"))

        tree.dump(config.memory_dir)

        # Create fresh instance and load
        with (
            patch("memos.memories.textual.markdown.LLMFactory") as llm_f,
            patch("memos.memories.textual.markdown.EmbedderFactory") as emb_f,
        ):
            llm_f.from_config.return_value = MagicMock()
            emb_f.from_config.return_value = mock_embedder
            tree2 = MarkdownTreeTextMemory(config)

        tree2.load(config.memory_dir)

        assert tree2.get_parent(child) == root
        assert tree2.get_parent(grandchild) == child
        assert tree2.get_children(root) == [child]
        path = tree2.get_path_to_root(grandchild)
        assert path == [grandchild, child, root]
