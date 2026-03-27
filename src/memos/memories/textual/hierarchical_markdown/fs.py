"""Filesystem I/O primitives for the hierarchical Markdown memory.

Every memory node is a ``.md`` file with YAML frontmatter, a body, and
optional Markdown-link edge sections.  This module provides the low-level
read/write/naming functions that all higher-level components build on.
"""

import os
import re
import unicodedata

from typing import Any

import yaml

from memos.log import get_logger


logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SUMMARY_FILENAME = "_summary.md"
ROOT_FILENAME = "_root.md"
FRESH_DIR = "_fresh"
SHARED_DIR = "_shared"
EMBEDDINGS_FILENAME = "_embeddings.json"

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_LINK_RE = re.compile(r"^-\s+\[([^\]]*)\]\(([^)]+)\)\s*$")
_SEQ_PREFIX_RE = re.compile(r"^(\d+)")

# Sections that contain edge links
_EDGE_SECTIONS = frozenset({"Children", "Related", "Sequence", "Shared"})


# ── Slug / naming ────────────────────────────────────────────────────────────


def slugify(text: str, max_len: int = 40) -> str:
    """Convert *text* to a filesystem-safe slug.

    >>> slugify("Japan trip planning!")
    'japan-trip-planning'
    >>> slugify("  Über café — résumé  ", max_len=15)
    'uber-cafe-resum'
    """
    # Normalize unicode → ASCII approximation
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "node"


def next_seq(dir_path: str, *, for_dir: bool = False) -> int:
    """Return the next sequence number for files or subdirectories in *dir_path*.

    Scans for ``NNN-*.md`` (files) or ``NN-*/`` (directories) and returns
    ``max + 1``.  Returns 1 if the directory is empty or doesn't exist.
    """
    if not os.path.isdir(dir_path):
        return 1
    max_seq = 0
    for entry in os.listdir(dir_path):
        if entry.startswith("_"):
            continue
        m = _SEQ_PREFIX_RE.match(entry)
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    return max_seq + 1


def leaf_filename(seq: int, key: str) -> str:
    """Build a leaf filename: ``NNN-slug.md``.

    >>> leaf_filename(3, "Budget discussion")
    '003-budget-discussion.md'
    """
    slug = slugify(key)
    return f"{seq:03d}-{slug}.md"


def subdir_name(seq: int, key: str) -> str:
    """Build a subdirectory name: ``NN-slug``.

    >>> subdir_name(2, "Restaurant recommendations")
    '02-restaurant-recommendations'
    """
    slug = slugify(key)
    return f"{seq:02d}-{slug}"


# ── Write ─────────────────────────────────────────────────────────────────────


def _render_frontmatter(meta: dict[str, Any]) -> str:
    """Render a metadata dict as YAML frontmatter block."""
    return yaml.safe_dump(
        meta, default_flow_style=False, allow_unicode=True, sort_keys=False
    ).rstrip("\n")


def _render_edges(edges: dict[str, list[tuple[str, str]]]) -> str:
    """Render edge sections as Markdown.

    *edges* maps section name → list of (label, relative_path).

    Returns a string like::

        ## Children
        - [Planning phase](./01-planning/_summary.md)
        - [Restaurants](./02-restaurants/_summary.md)

        ## Related
        - [Work project](../02-work/_summary.md)
    """
    parts: list[str] = []
    for section, links in edges.items():
        if not links:
            continue
        lines = [f"## {section}"]
        for label, path in links:
            lines.append(f"- [{label}]({path})")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def write_md(
    filepath: str,
    body: str,
    meta: dict[str, Any],
    edges: dict[str, list[tuple[str, str]]] | None = None,
) -> None:
    """Write a ``.md`` file with YAML frontmatter, body, and edge sections."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fm = _render_frontmatter(meta)
    parts = [f"---\n{fm}\n---\n", body]
    if edges:
        edge_text = _render_edges(edges)
        if edge_text:
            parts.append("\n\n" + edge_text)
    content = "\n".join(parts)
    if not content.endswith("\n"):
        content += "\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def write_leaf(
    dir_path: str,
    body: str,
    meta: dict[str, Any],
    edges: dict[str, list[tuple[str, str]]] | None = None,
) -> str:
    """Write a leaf ``.md`` file into *dir_path*.  Returns the relative path."""
    seq = next_seq(dir_path, for_dir=False)
    key = meta.get("key", "")
    fname = leaf_filename(seq, key)
    fpath = os.path.join(dir_path, fname)
    write_md(fpath, body, meta, edges)
    return fname


def write_summary(
    dir_path: str,
    body: str,
    meta: dict[str, Any],
    children: list[tuple[str, str]],
    extra_edges: dict[str, list[tuple[str, str]]] | None = None,
) -> str:
    """Write ``_summary.md`` into *dir_path* with ``## Children`` links.

    *children* is a list of (label, relative_path).
    Returns the path to the written file.
    """
    edges: dict[str, list[tuple[str, str]]] = {"Children": children}
    if extra_edges:
        edges.update(extra_edges)
    fpath = os.path.join(dir_path, SUMMARY_FILENAME)
    write_md(fpath, body, meta, edges)
    return fpath


# ── Read ──────────────────────────────────────────────────────────────────────


def read_md(filepath: str) -> tuple[dict[str, Any], str, dict[str, list[tuple[str, str]]]]:
    """Parse a ``.md`` file → ``(frontmatter, body, edges)``.

    *edges* maps section name → list of ``(label, relative_path)``.
    """
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    # 1. Frontmatter
    fm_match = _FRONTMATTER_RE.match(text)
    if not fm_match:
        return {}, text.strip(), {}
    raw_meta: dict[str, Any] = yaml.safe_load(fm_match.group(1)) or {}
    remainder = text[fm_match.end() :]

    # 2. Split body from edge sections
    sections = _SECTION_RE.split(remainder)
    # sections[0] = body text before any ## heading
    body = sections[0].strip()

    # 3. Parse edge sections
    edges: dict[str, list[tuple[str, str]]] = {}
    i = 1
    while i < len(sections) - 1:
        heading = sections[i].strip()
        content = sections[i + 1]
        if heading in _EDGE_SECTIONS:
            links: list[tuple[str, str]] = []
            for line in content.strip().splitlines():
                m = _LINK_RE.match(line.strip())
                if m:
                    links.append((m.group(1), m.group(2)))
            if links:
                edges[heading] = links
        else:
            # Non-edge section — append to body
            body += f"\n\n## {heading}\n{content.strip()}"
        i += 2

    return raw_meta, body, edges


def list_children(dir_path: str) -> list[tuple[str, str]]:
    """Return ``(label, relative_path)`` for all child nodes in *dir_path*.

    Children are:
    - Subdirectories containing ``_summary.md`` → ``(dir_slug, ./subdir/_summary.md)``
    - Leaf ``.md`` files (not starting with ``_``) → ``(leaf_slug, ./filename.md)``

    Results are sorted by sequence number.
    """
    if not os.path.isdir(dir_path):
        return []

    children: list[tuple[int, str, str]] = []  # (seq, label, rel_path)
    for entry in os.listdir(dir_path):
        if entry.startswith("_"):
            continue
        full = os.path.join(dir_path, entry)

        if os.path.isdir(full):
            summary = os.path.join(full, SUMMARY_FILENAME)
            if os.path.isfile(summary):
                m = _SEQ_PREFIX_RE.match(entry)
                seq = int(m.group(1)) if m else 999
                label = entry.split("-", 1)[1] if "-" in entry else entry
                children.append((seq, label, f"./{entry}/{SUMMARY_FILENAME}"))
        elif entry.endswith(".md"):
            m = _SEQ_PREFIX_RE.match(entry)
            seq = int(m.group(1)) if m else 999
            label = entry.rsplit(".", 1)[0]
            if "-" in label:
                label = label.split("-", 1)[1]
            children.append((seq, label, f"./{entry}"))

    children.sort(key=lambda x: x[0])
    return [(label, path) for _, label, path in children]


def count_leaves(dir_path: str) -> int:
    """Recursively count all leaf ``.md`` files under *dir_path*."""
    count = 0
    if not os.path.isdir(dir_path):
        return 0
    for entry in os.listdir(dir_path):
        if entry.startswith("_"):
            continue
        full = os.path.join(dir_path, entry)
        if os.path.isdir(full):
            count += count_leaves(full)
        elif entry.endswith(".md"):
            count += 1
    return count


def walk_tree(dir_path: str) -> list[str]:
    """Return all ``.md`` file paths under *dir_path* (recursive), sorted."""
    paths: list[str] = []
    if not os.path.isdir(dir_path):
        return paths
    for root, dirs, files in os.walk(dir_path):
        # Skip special directories
        dirs[:] = [d for d in sorted(dirs) if not d.startswith("_")]
        for f in sorted(files):
            if (f.endswith(".md") and not f.startswith("_")) or f == SUMMARY_FILENAME:
                paths.append(os.path.join(root, f))
    return paths
