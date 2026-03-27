#!/usr/bin/env python3
"""Interactive demo: Markdown-backed tree memory across sessions.

Usage:
    # Session 1: add memories from conversations
    python examples/markdown_memory_demo.py

    # Session 2: reload and search / add more
    python examples/markdown_memory_demo.py

The memory persists in .memos/demo_memory/ as human-readable .md files.
You can open them in any text editor or Markdown viewer between sessions.

Requirements:
    - OPENAI_API_KEY and OPENAI_BASE_URL set in environment (e.g. OpenRouter)
    - sentence-transformers installed (local embeddings, no API)
"""

import json
import os
import sys
import textwrap

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memos.configs.memory import MarkdownTextMemoryConfig, MemoryConfigFactory
from memos.memories.factory import MemoryFactory
from memos.memories.textual.markdown_tree import MarkdownTreeTextMemory


# ── Configuration ─────────────────────────────────────────────────────────────

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", ".memos", "demo_memory")
# Resolve to absolute path
MEMORY_DIR = os.path.abspath(MEMORY_DIR)

# Local sentence-transformer model (cached at ~/.cache/huggingface/...)
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# OpenRouter / OpenAI config from env
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")


def build_memory() -> MarkdownTreeTextMemory:
    """Build (or reload) the tree memory from disk."""
    config = MarkdownTextMemoryConfig(
        memory_dir=MEMORY_DIR,
        extractor_llm={
            "backend": "openai",
            "config": {
                "model_name_or_path": CHAT_MODEL,
                "api_key": API_KEY,
                "api_base": API_BASE,
            },
        },
        embedder={
            "backend": "sentence_transformer",
            "config": {
                "model_name_or_path": EMBEDDER_MODEL,
            },
        },
    )
    mem = MarkdownTreeTextMemory(config)

    # Reload from disk if directory already has .md files
    if os.path.isdir(MEMORY_DIR):
        md_count = sum(1 for f in os.listdir(MEMORY_DIR) if f.endswith(".md") and f != "_index.md")
        if md_count > 0:
            mem.load(MEMORY_DIR)
            print(f"✅ Loaded {len(mem.get_all())} memories from {MEMORY_DIR}")
        else:
            print(f"📁 Memory directory exists but empty: {MEMORY_DIR}")
    else:
        print(f"📁 Fresh start — will create: {MEMORY_DIR}")

    return mem


def show_tree(mem: MarkdownTreeTextMemory) -> None:
    """Print the memory tree to stdout."""
    index_path = os.path.join(MEMORY_DIR, "_index.md")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            print(f.read())
    else:
        print("(empty tree)")


def show_memory(mem: MarkdownTreeTextMemory, memory_id: str) -> None:
    """Print a single memory node."""
    item = mem.get(memory_id)
    if not item.memory:
        print(f"  ❌ Not found: {memory_id}")
        return
    print(f"  ID:     {item.id}")
    print(f"  Key:    {item.metadata.key}")
    print(f"  Tags:   {item.metadata.tags}")
    print(f"  Body:   {item.memory[:200]}")
    edges = mem.get_edges(memory_id)
    if edges:
        print(f"  Edges:  {edges}")
    parent = mem.get_parent(memory_id)
    children = mem.get_children(memory_id)
    if parent:
        print(f"  Parent: {parent[:8]}...")
    if children:
        print(f"  Children: {[c[:8] for c in children]}")
    print()


HELP = textwrap.dedent("""\
    Commands:
      add         — Paste a conversation and extract memories via LLM
      search <q>  — Search memories by query (local embeddings, no API)
      tree        — Show the memory tree (_index.md)
      list        — List all memories (brief)
      show <id>   — Show a single memory (full details)
      child <pid> — Add a child memory under parent <pid> (8-char prefix ok)
      link <s> <t> [type] — Add an edge from s → t
      save        — Force save to disk
      quit        — Exit
""")


def resolve_id(mem: MarkdownTreeTextMemory, prefix: str) -> str | None:
    """Resolve a short ID prefix to the full UUID."""
    for mid in mem._index:
        if mid.startswith(prefix):
            return mid
    return None


def cmd_add(mem: MarkdownTreeTextMemory) -> None:
    """Interactive conversation → memory extraction."""
    print("Paste conversation lines (user:/assistant:), empty line to finish:")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)

    if not lines:
        print("No input.")
        return

    # Parse into messages
    messages = []
    for line in lines:
        if line.lower().startswith("user:"):
            messages.append({"role": "user", "content": line[5:].strip()})
        elif line.lower().startswith("assistant:"):
            messages.append({"role": "assistant", "content": line[10:].strip()})
        else:
            # Assume continuation of previous message or user
            if messages:
                messages[-1]["content"] += "\n" + line
            else:
                messages.append({"role": "user", "content": line})

    print(f"\n🔍 Extracting memories from {len(messages)} messages via LLM...")
    try:
        items = mem.extract(messages)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return

    if not items:
        print("No memories extracted.")
        return

    print(f"\n✨ Extracted {len(items)} memories:")
    for i, item in enumerate(items, 1):
        print(f"  {i}. [{item.metadata.key}] {item.memory[:100]}")

    ids = mem.add(items)
    mem.dump(MEMORY_DIR)
    print(f"💾 Saved {len(ids)} memories to disk.\n")


def cmd_search(mem: MarkdownTreeTextMemory, query: str) -> None:
    """Search memories (local embeddings only, no API call)."""
    results = mem.search(query, top_k=5)
    if not results:
        print("No results found.")
        return
    print(f"\n🔎 Top {len(results)} results for '{query}':\n")
    for i, item in enumerate(results, 1):
        print(f"  {i}. [{item.metadata.key}] {item.memory[:120]}")
        print(f"     ID: {item.id[:8]}...  Tags: {item.metadata.tags}\n")


def cmd_child(mem: MarkdownTreeTextMemory, parent_prefix: str) -> None:
    """Manually add a child node under a parent."""
    pid = resolve_id(mem, parent_prefix)
    if not pid:
        print(f"❌ No memory found starting with '{parent_prefix}'")
        return
    parent = mem.get(pid)
    print(f"Parent: [{parent.metadata.key}]")
    key = input("  Child key: ").strip()
    body = input("  Child body: ").strip()
    if not key or not body:
        print("Cancelled.")
        return
    from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata

    child = TextualMemoryItem(
        memory=body,
        metadata=TextualMemoryMetadata(key=key, source="conversation"),
    )
    child_id = mem.add_child(pid, child)
    mem.dump(MEMORY_DIR)
    print(f"✅ Added child {child_id[:8]}... under {pid[:8]}...")


def cmd_link(mem: MarkdownTreeTextMemory, args: list[str]) -> None:
    """Add an edge between two nodes."""
    if len(args) < 2:
        print("Usage: link <source_prefix> <target_prefix> [edge_type]")
        return
    src = resolve_id(mem, args[0])
    tgt = resolve_id(mem, args[1])
    etype = args[2] if len(args) > 2 else "RELATED"
    if not src or not tgt:
        print(f"❌ Could not resolve IDs: {args[0]}, {args[1]}")
        return
    mem.add_edge(src, tgt, etype)
    mem.dump(MEMORY_DIR)
    print(f"✅ {etype}: {src[:8]} → {tgt[:8]}")


def main() -> None:
    if not API_KEY:
        print("⚠️  OPENAI_API_KEY not set. `add` command (LLM extraction) won't work.")
        print("   `search` still works (local embeddings only).\n")

    print("🧠 MemOS Markdown Tree Memory — Interactive Demo")
    print(f"   Storage: {MEMORY_DIR}")
    print(f"   LLM:     {CHAT_MODEL} via {API_BASE or 'default'}")
    print(f"   Embedder: {EMBEDDER_MODEL} (local)\n")

    mem = build_memory()
    print(HELP)

    while True:
        try:
            raw = input("memo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "quit" or cmd == "exit":
            mem.dump(MEMORY_DIR)
            print("💾 Saved. Bye!")
            break
        elif cmd == "add":
            cmd_add(mem)
        elif cmd == "search":
            if not arg:
                print("Usage: search <query>")
            else:
                cmd_search(mem, arg)
        elif cmd == "tree":
            show_tree(mem)
        elif cmd == "list":
            all_items = mem.get_all()
            if not all_items:
                print("(empty)")
            for item in all_items:
                parent = mem.get_parent(item.id)
                depth = mem.get_depth(item.id)
                indent = "  " * depth
                print(f"  {indent}[{item.id[:8]}] {item.metadata.key}")
        elif cmd == "show":
            mid = resolve_id(mem, arg) if arg else None
            if mid:
                show_memory(mem, mid)
            else:
                print(f"❌ Not found: {arg}")
        elif cmd == "child":
            if not arg:
                print("Usage: child <parent_id_prefix>")
            else:
                cmd_child(mem, arg)
        elif cmd == "link":
            cmd_link(mem, arg.split())
        elif cmd == "save":
            mem.dump(MEMORY_DIR)
            print(f"💾 Saved to {MEMORY_DIR}")
        elif cmd == "help":
            print(HELP)
        else:
            print(f"Unknown command: {cmd}. Type 'help' for commands.")


if __name__ == "__main__":
    main()
