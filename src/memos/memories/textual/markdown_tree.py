"""Tree-structured memory built on top of the Markdown file backend.

Adds hierarchical organization to :class:`MarkdownTextMemory`:

*   Every node may have a **parent** (``PARENT`` edge) and **children**
    (nodes whose ``PARENT`` edge points back).
*   Convenience methods walk the tree: ``get_children``, ``get_parent``,
    ``get_subtree``, ``get_root``, ``get_path_to_root``.
*   A dedicated ``_index.md`` file is maintained as a human-readable
    table of contents that mirrors the tree structure, updated on every
    mutation.

The on-disk layout is identical to :class:`MarkdownTextMemory` — one
``.md`` file per node with YAML frontmatter, body, and an ``## Edges``
section.  The tree relationships are encoded as regular edges
(``PARENT``, ``CHILD``, ``RELATED``, ``FOLLOWING``, etc.) so the
storage stays a flat directory of ``.md`` files that any Markdown viewer
or editor can read.

Example ``_index.md``::

    # Memory Tree Index

    - **Travel plans** (`a1b2c3`)
      - **Paris itinerary** (`d4e5f6`)
      - **Flight booking** (`7g8h9i`)
    - **Health notes** (`j0k1l2`)
"""

import os

from typing import Any

from memos.log import get_logger
from memos.memories.textual.item import TextualMemoryItem
from memos.memories.textual.markdown import MarkdownTextMemory


logger = get_logger(__name__)


class MarkdownTreeTextMemory(MarkdownTextMemory):
    """Tree-structured memory backed by Markdown files.

    Inherits **all** CRUD / search / edge operations from
    :class:`MarkdownTextMemory` and layers tree-walking helpers plus an
    auto-generated ``_index.md`` table of contents on top.
    """

    # ── Overridden mutators (refresh index after every change) ────────────

    def add(self, memories: list[TextualMemoryItem | dict[str, Any]], **kwargs: Any) -> list[str]:
        ids = super().add(memories, **kwargs)
        if ids:
            self._write_index()
        return ids

    def update(self, memory_id: str, new_memory: TextualMemoryItem | dict[str, Any]) -> None:
        super().update(memory_id, new_memory)
        self._write_index()

    def delete(self, memory_ids: list[str]) -> None:
        # Before deleting, re-parent orphaned children to grandparent (or root)
        for mid in memory_ids:
            parent_id = self.get_parent(mid)
            for child_id in self.get_children(mid):
                # Point each child to the grandparent (or detach if root)
                self.remove_edge(child_id, mid, "PARENT")
                if parent_id is not None:
                    self.add_edge(child_id, parent_id, "PARENT")
        super().delete(memory_ids)
        self._write_index()

    def delete_all(self) -> None:
        super().delete_all()
        self._delete_index()

    def load(self, dir: str) -> None:
        super().load(dir)
        # Rebuild the index after loading from disk
        self._write_index()

    # ── Tree mutation helpers ─────────────────────────────────────────────

    def add_child(
        self,
        parent_id: str,
        child: TextualMemoryItem | dict[str, Any],
    ) -> str:
        """Add *child* as a child of *parent_id*. Returns the child id."""
        if parent_id not in self._index:
            raise ValueError(f"Parent memory {parent_id} not found")
        ids = super().add([child])
        if not ids:
            raise ValueError("Failed to add child node (duplicate?)")
        child_id = ids[0]
        self.add_edge(child_id, parent_id, "PARENT")
        self._write_index()
        return child_id

    def move_node(self, node_id: str, new_parent_id: str | None) -> None:
        """Move *node_id* under *new_parent_id* (or make it a root if ``None``)."""
        if node_id not in self._index:
            raise ValueError(f"Node {node_id} not found")
        # Remove old PARENT edge
        old_parent = self.get_parent(node_id)
        if old_parent is not None:
            self.remove_edge(node_id, old_parent, "PARENT")
        # Add new PARENT edge
        if new_parent_id is not None:
            if new_parent_id not in self._index:
                raise ValueError(f"New parent {new_parent_id} not found")
            self.add_edge(node_id, new_parent_id, "PARENT")
        self._write_index()

    # ── Tree query helpers ────────────────────────────────────────────────

    def get_parent(self, node_id: str) -> str | None:
        """Return the parent ID of *node_id*, or ``None`` if it is a root."""
        for edge in self.get_edges(node_id):
            if edge["type"] == "PARENT":
                return edge["target"]
        return None

    def get_children(self, node_id: str) -> list[str]:
        """Return IDs of all direct children of *node_id*."""
        children: list[str] = []
        for mid, (_, edges) in self._index.items():
            if mid == node_id:
                continue
            for edge in edges:
                if edge["type"] == "PARENT" and edge["target"] == node_id:
                    children.append(mid)
                    break
        return children

    def get_roots(self) -> list[str]:
        """Return IDs of all root nodes (nodes with no ``PARENT`` edge)."""
        return [mid for mid in self._index if self.get_parent(mid) is None]

    def get_path_to_root(self, node_id: str) -> list[str]:
        """Return the path from *node_id* up to its root (inclusive)."""
        path = [node_id]
        visited = {node_id}
        current = node_id
        while True:
            parent = self.get_parent(current)
            if parent is None or parent in visited:
                break
            path.append(parent)
            visited.add(parent)
            current = parent
        return path

    def get_subtree(self, node_id: str) -> list[str]:
        """Return IDs of *node_id* and all its descendants (BFS)."""
        result: list[str] = []
        queue = [node_id]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(self.get_children(current))
        return result

    def get_depth(self, node_id: str) -> int:
        """Return the depth of *node_id* (root = 0)."""
        return len(self.get_path_to_root(node_id)) - 1

    def get_siblings(self, node_id: str) -> list[str]:
        """Return IDs of siblings (same parent, excluding self)."""
        parent = self.get_parent(node_id)
        if parent is None:
            # Siblings are the other roots
            return [r for r in self.get_roots() if r != node_id]
        return [c for c in self.get_children(parent) if c != node_id]

    def search_subtree(
        self, root_id: str, query: str, top_k: int, **kwargs: Any
    ) -> list[TextualMemoryItem]:
        """Search only within the subtree rooted at *root_id*."""
        subtree_ids = set(self.get_subtree(root_id))
        # Run a full search then filter
        all_results = self.search(query, top_k=top_k * 3, **kwargs)
        filtered = [r for r in all_results if r.id in subtree_ids]
        return filtered[:top_k]

    # ── _index.md generation ──────────────────────────────────────────────

    def _write_index(self) -> None:
        """Write a ``_index.md`` table of contents mirroring the tree."""
        memory_dir = self.config.memory_dir
        if not memory_dir:
            return
        os.makedirs(memory_dir, exist_ok=True)

        lines = ["# Memory Tree Index", ""]
        roots = self.get_roots()

        # Sort roots by key for stable output
        def _sort_key(mid: str) -> str:
            item = self._index[mid][0]
            return (item.metadata.key or item.memory[:40]).lower()

        roots.sort(key=_sort_key)
        for root_id in roots:
            self._render_subtree(root_id, lines, indent=0)

        if not roots:
            lines.append("*(empty)*")

        fpath = os.path.join(memory_dir, "_index.md")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _render_subtree(self, node_id: str, lines: list[str], indent: int) -> None:
        """Recursively render a node and its children as indented bullets."""
        entry = self._index.get(node_id)
        if entry is None:
            return
        item, _ = entry
        key = item.metadata.key or item.memory[:60].replace("\n", " ")
        short_id = node_id[:8]
        prefix = "  " * indent
        lines.append(f"{prefix}- **{key}** (`{short_id}`)")
        children = self.get_children(node_id)
        children.sort(key=lambda cid: (self._index[cid][0].metadata.key or "").lower())
        for child_id in children:
            self._render_subtree(child_id, lines, indent + 1)

    def _delete_index(self) -> None:
        """Remove the ``_index.md`` file."""
        memory_dir = self.config.memory_dir
        if not memory_dir:
            return
        fpath = os.path.join(memory_dir, "_index.md")
        if os.path.isfile(fpath):
            os.remove(fpath)
