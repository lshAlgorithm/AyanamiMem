#!/usr/bin/env python3
"""Non-interactive end-to-end test for markdown tree memory."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memos.configs.memory import MarkdownTextMemoryConfig
from memos.memories.textual.markdown_tree import MarkdownTreeTextMemory

MEMORY_DIR = "/tmp/memos_e2e_test"
API_KEY = os.environ["OPENAI_API_KEY"]
API_BASE = os.environ.get(
    "OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")

# Clean up previous run
import shutil

if os.path.isdir(MEMORY_DIR):
    shutil.rmtree(MEMORY_DIR)

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
        "config": {"model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"},
    },
)

# === Session 1 ===
print("=" * 60)
print("SESSION 1: Create memories from conversation")
print("=" * 60)

mem = MarkdownTreeTextMemory(config)
print(f"[1] Init OK, {len(mem.get_all())} memories")

messages = [
    {
        "role": "user",
        "content": "I am planning a trip to Japan next month. I love sushi and want to visit Kyoto temples.",
    },
    {"role": "assistant", "content": "Kyoto is wonderful! Fushimi Inari Shrine is a must-see."},
    {"role": "user", "content": "I also want to try onsen. My budget is around 3000 USD."},
]
print("[2] Extracting memories via LLM...")
items = mem.extract(messages)
print(f"[3] Extracted {len(items)} memories:")
for item in items:
    print(f"    key={item.metadata.key!r}")
    print(f"    body={item.memory[:100]!r}")
    print()

ids = mem.add(items)
print(f"[4] Added {len(ids)} memories")

# Search (local embeddings only - no API call)
results = mem.search("Japanese food", top_k=3)
print(f"\n[5] Search 'Japanese food' -> {len(results)} results:")
for r in results:
    print(f"    [{r.metadata.key}] {r.memory[:80]}")

# Build a tree: first memory is root, rest are children
if len(ids) >= 2:
    root_id = ids[0]
    for child_id in ids[1:]:
        mem.add_edge(child_id, root_id, "PARENT")
    print(f"\n[6] Built tree: root={root_id[:8]}..., {len(ids) - 1} children")

# Dump to disk
mem.dump(MEMORY_DIR)
print(f"\n[7] Saved to {MEMORY_DIR}")
print("    Files on disk:")
for f in sorted(os.listdir(MEMORY_DIR)):
    print(f"      {f}")

# Show _index.md
idx_path = os.path.join(MEMORY_DIR, "_index.md")
if os.path.isfile(idx_path):
    print("\n[8] _index.md content:")
    with open(idx_path) as f:
        print(f.read())

# Show one .md file as example
md_files = [f for f in os.listdir(MEMORY_DIR) if f.endswith(".md") and f != "_index.md"]
if md_files:
    example = os.path.join(MEMORY_DIR, md_files[0])
    print(f"[9] Example memory file ({md_files[0]}):")
    with open(example) as f:
        print(f.read())

# === Session 2: Simulate restart ===
print("=" * 60)
print("SESSION 2: Reload from disk (simulating new process)")
print("=" * 60)

mem2 = MarkdownTreeTextMemory(config)
mem2.load(MEMORY_DIR)
print(f"[10] Loaded {len(mem2.get_all())} memories from disk")

# Search again (local only)
results2 = mem2.search("budget travel", top_k=2)
print(f"\n[11] Search 'budget travel' -> {len(results2)} results:")
for r in results2:
    print(f"    [{r.metadata.key}] {r.memory[:80]}")

# Tree structure persisted?
roots = mem2.get_roots()
print(f"\n[12] Tree structure persisted:")
for root in roots:
    root_item = mem2.get(root)
    children = mem2.get_children(root)
    print(f"    Root: [{root_item.metadata.key}] ({root[:8]}...)")
    for c in children:
        c_item = mem2.get(c)
        print(f"      └── [{c_item.metadata.key}] ({c[:8]}...)")

print("\n✅ Full end-to-end test passed!")
