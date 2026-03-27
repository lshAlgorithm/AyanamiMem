"""Tests for hierarchical_markdown.fs — filesystem I/O primitives."""

import os

from memos.memories.textual.hierarchical_markdown.fs import (
    leaf_filename,
    list_children,
    next_seq,
    read_md,
    slugify,
    subdir_name,
    write_leaf,
    write_md,
    write_summary,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Japan trip planning!") == "japan-trip-planning"

    def test_unicode(self):
        assert slugify("Über café résumé") == "uber-cafe-resume"

    def test_max_len(self):
        result = slugify("this is a very long title that should be truncated", max_len=15)
        assert len(result) <= 15
        assert not result.endswith("-")

    def test_empty(self):
        assert slugify("") == "node"

    def test_special_chars(self):
        assert slugify("hello@world#2024") == "hello-world-2024"

    def test_whitespace(self):
        assert slugify("  lots   of   spaces  ") == "lots-of-spaces"


class TestNaming:
    def test_leaf_filename(self):
        assert leaf_filename(3, "Budget discussion") == "003-budget-discussion.md"

    def test_leaf_filename_long_key(self):
        name = leaf_filename(1, "A" * 100)
        assert name.startswith("001-")
        assert name.endswith(".md")

    def test_subdir_name(self):
        assert subdir_name(2, "Restaurant recommendations") == "02-restaurant-recommendations"


class TestNextSeq:
    def test_empty_dir(self, tmp_path):
        assert next_seq(str(tmp_path)) == 1

    def test_nonexistent_dir(self):
        assert next_seq("/nonexistent/path") == 1

    def test_with_files(self, tmp_path):
        (tmp_path / "001-foo.md").touch()
        (tmp_path / "002-bar.md").touch()
        assert next_seq(str(tmp_path)) == 3

    def test_with_dirs(self, tmp_path):
        (tmp_path / "01-topic-a").mkdir()
        (tmp_path / "02-topic-b").mkdir()
        assert next_seq(str(tmp_path)) == 3

    def test_skips_underscored(self, tmp_path):
        (tmp_path / "_summary.md").touch()
        (tmp_path / "_fresh").mkdir()
        (tmp_path / "001-real.md").touch()
        assert next_seq(str(tmp_path)) == 2


class TestWriteAndReadMd:
    def test_roundtrip_basic(self, tmp_path):
        fpath = str(tmp_path / "test.md")
        meta = {"key": "test node", "depth": 0, "node_kind": "leaf"}
        body = "This is the body content."
        write_md(fpath, body, meta)

        read_meta, read_body, read_edges = read_md(fpath)
        assert read_meta["key"] == "test node"
        assert read_meta["depth"] == 0
        assert read_body == body
        assert read_edges == {}

    def test_roundtrip_with_edges(self, tmp_path):
        fpath = str(tmp_path / "test.md")
        meta = {"key": "parent node", "depth": 1, "node_kind": "condensed"}
        body = "Summary of children."
        edges = {
            "Children": [
                ("child-a", "./01-child-a/_summary.md"),
                ("child-b", "./02-child-b/_summary.md"),
            ],
            "Related": [("work", "../02-work/_summary.md")],
        }
        write_md(fpath, body, meta, edges)

        read_meta, read_body, read_edges = read_md(fpath)
        assert read_meta["key"] == "parent node"
        assert read_body == "Summary of children."
        assert len(read_edges["Children"]) == 2
        assert read_edges["Children"][0] == ("child-a", "./01-child-a/_summary.md")
        assert read_edges["Related"][0] == ("work", "../02-work/_summary.md")

    def test_tags_in_frontmatter(self, tmp_path):
        fpath = str(tmp_path / "test.md")
        meta = {"key": "tagged", "tags": ["travel", "japan"]}
        write_md(fpath, "Body.", meta)

        read_meta, _, _ = read_md(fpath)
        assert read_meta["tags"] == ["travel", "japan"]

    def test_no_frontmatter(self, tmp_path):
        fpath = str(tmp_path / "plain.md")
        with open(fpath, "w") as f:
            f.write("Just plain text, no frontmatter.")

        meta, body, edges = read_md(fpath)
        assert meta == {}
        assert "plain text" in body
        assert edges == {}

    def test_non_edge_section_preserved_in_body(self, tmp_path):
        fpath = str(tmp_path / "test.md")
        content = "---\nkey: test\n---\n\nBody text.\n\n## Notes\nSome notes here.\n"
        with open(fpath, "w") as f:
            f.write(content)

        _meta, body, edges = read_md(fpath)
        assert "Notes" in body
        assert "Some notes here." in body
        assert edges == {}


class TestWriteLeaf:
    def test_creates_file(self, tmp_path):
        meta = {"key": "my leaf", "depth": 0, "node_kind": "leaf"}
        fname = write_leaf(str(tmp_path), "Leaf body.", meta)
        assert fname == "001-my-leaf.md"
        assert (tmp_path / fname).is_file()

    def test_sequential(self, tmp_path):
        for i in range(3):
            write_leaf(str(tmp_path), f"Body {i}.", {"key": f"leaf {i}", "depth": 0})
        files = sorted(f for f in os.listdir(str(tmp_path)) if f.endswith(".md"))
        assert files == ["001-leaf-0.md", "002-leaf-1.md", "003-leaf-2.md"]


class TestWriteSummary:
    def test_creates_summary(self, tmp_path):
        children = [("child-a", "./001-child-a.md"), ("child-b", "./002-child-b.md")]
        meta = {"key": "parent", "depth": 1, "node_kind": "condensed"}
        write_summary(str(tmp_path), "Summary text.", meta, children)

        fpath = tmp_path / "_summary.md"
        assert fpath.is_file()

        read_meta, read_body, read_edges = read_md(str(fpath))
        assert read_meta["key"] == "parent"
        assert read_body == "Summary text."
        assert len(read_edges["Children"]) == 2


class TestListChildren:
    def test_empty(self, tmp_path):
        assert list_children(str(tmp_path)) == []

    def test_leaves_only(self, tmp_path):
        write_leaf(str(tmp_path), "Body A.", {"key": "alpha", "depth": 0})
        write_leaf(str(tmp_path), "Body B.", {"key": "beta", "depth": 0})
        children = list_children(str(tmp_path))
        assert len(children) == 2
        assert children[0][1] == "./001-alpha.md"
        assert children[1][1] == "./002-beta.md"

    def test_subdirs_with_summary(self, tmp_path):
        subdir = tmp_path / "01-topic-a"
        subdir.mkdir()
        write_summary(str(subdir), "Topic A summary.", {"key": "topic-a", "depth": 1}, [])
        children = list_children(str(tmp_path))
        assert len(children) == 1
        assert children[0][1] == "./01-topic-a/_summary.md"

    def test_mixed(self, tmp_path):
        # A leaf + a subdir
        write_leaf(str(tmp_path), "Leaf.", {"key": "leaf", "depth": 0})
        subdir = tmp_path / "02-subdir"
        subdir.mkdir()
        write_summary(str(subdir), "Sub.", {"key": "sub", "depth": 1}, [])
        children = list_children(str(tmp_path))
        assert len(children) == 2

    def test_ignores_underscored(self, tmp_path):
        (tmp_path / "_summary.md").touch()
        (tmp_path / "_fresh").mkdir()
        write_leaf(str(tmp_path), "Real.", {"key": "real", "depth": 0})
        children = list_children(str(tmp_path))
        assert len(children) == 1

    def test_nonexistent_dir(self):
        assert list_children("/nonexistent") == []
