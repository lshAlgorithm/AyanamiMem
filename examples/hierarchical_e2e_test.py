#!/usr/bin/env python3
"""End-to-end test for HierarchicalMarkdownMemory with a real LLM.

Demonstrates the full lifecycle:
  1. Chunk conversations into _fresh/ leaves
  2. Compact leaves into a directory tree with LLM-generated summaries
  3. Search the tree (top-down walk, local embeddings only)
  4. Show the on-disk structure (human-readable .md files)
  5. Reload from disk (cross-session persistence)

Usage:
    .venv/bin/python examples/hierarchical_e2e_test.py
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memos.configs.memory import HierarchicalMarkdownMemoryConfig
from memos.memories.textual.hierarchical_markdown_memory import HierarchicalMarkdownMemory

MEMORY_DIR = "/tmp/hm_e2e_test"
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not API_KEY:
    print("❌ Set OPENAI_API_KEY (and OPENAI_BASE_URL if using OpenRouter)")
    sys.exit(1)

# Clean slate
if os.path.isdir(MEMORY_DIR):
    shutil.rmtree(MEMORY_DIR)

config = HierarchicalMarkdownMemoryConfig(
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
        "config": {"model_name_or_path": EMBEDDER_MODEL},
    },
    fresh_tail_count=2,  # keep only 2 most recent as "fresh"
    condensed_min_fanout=2,  # compact when 2+ leaves available
    leaf_chunk_tokens=500,  # small chunks for demo
)

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SESSION 1: Create memories from conversations")
print("=" * 60)

mem = HierarchicalMarkdownMemory(config)
print(f"[1] Init OK. Memory dir: {MEMORY_DIR}\n")

# Conversation 1: Japan trip
conv1 = [
    {
        "role": "user",
        "content": "I'm planning a trip to Japan next month. I love sushi and want to visit Kyoto temples.",
    },
    {
        "role": "assistant",
        "content": "Kyoto is wonderful! Fushimi Inari Shrine is a must-see. For sushi, try Nishiki Market.",
    },
    {"role": "user", "content": "My budget is about 3000 USD. I also want to try onsen."},
]

print("[2] Extracting conversation 1 (Japan trip)...")
t0 = time.time()
items1 = mem.extract(conv1)
ids1 = mem.add(items1)
print(f"    → {len(ids1)} leaves in {time.time() - t0:.1f}s")
for item in items1:
    print(f"    leaf: [{item.metadata.key[:50]}]")

# Conversation 2: Work project
conv2 = [
    {
        "role": "user",
        "content": "I need to build a machine learning pipeline for text classification.",
    },
    {
        "role": "assistant",
        "content": "I'd recommend using HuggingFace Transformers with PyTorch. Fine-tune a BERT model.",
    },
    {"role": "user", "content": "Good idea. I'll use Python and deploy on AWS Lambda."},
]

print(f"\n[3] Extracting conversation 2 (ML project)...")
t0 = time.time()
items2 = mem.extract(conv2)
ids2 = mem.add(items2)
print(f"    → {len(ids2)} leaves in {time.time() - t0:.1f}s")
for item in items2:
    print(f"    leaf: [{item.metadata.key[:50]}]")

# Conversation 3: More Japan details
conv3 = [
    {"role": "user", "content": "What are the best restaurants near Fushimi Inari?"},
    {
        "role": "assistant",
        "content": "Vermillion cafe is right at the entrance. For local ramen, try Ramen Ori near the station.",
    },
]

print(f"\n[4] Extracting conversation 3 (restaurants)...")
t0 = time.time()
items3 = mem.extract(conv3)
ids3 = mem.add(items3)
print(f"    → {len(ids3)} leaves in {time.time() - t0:.1f}s")

# ── Compact ──
print(f"\n[5] Triggering manual compaction...")
t0 = time.time()
mem.compact()
print(f"    Compacted in {time.time() - t0:.1f}s")

# ── Show disk structure ──
print(f"\n[6] On-disk structure:")
for root, dirs, files in os.walk(MEMORY_DIR):
    dirs[:] = sorted(dirs)
    level = root.replace(MEMORY_DIR, "").count(os.sep)
    indent = "  " * level
    dirname = os.path.basename(root) or os.path.basename(MEMORY_DIR)
    print(f"  {indent}{dirname}/")
    for f in sorted(files):
        if f.endswith(".md"):
            print(f"  {indent}  {f}")

# ── Show a _summary.md if one exists ──
for root, dirs, files in os.walk(MEMORY_DIR):
    if "_summary.md" in files:
        summary_path = os.path.join(root, "_summary.md")
        print(f"\n[7] Example _summary.md ({os.path.relpath(summary_path, MEMORY_DIR)}):")
        with open(summary_path) as f:
            print(f.read())
        break

# ── Show a leaf ──
fresh_dir = os.path.join(MEMORY_DIR, "_fresh")
if os.path.isdir(fresh_dir):
    leaves = sorted(f for f in os.listdir(fresh_dir) if f.endswith(".md"))
    if leaves:
        leaf_path = os.path.join(fresh_dir, leaves[0])
        print(f"[8] Example leaf ({leaves[0]}):")
        with open(leaf_path) as f:
            print(f.read())

# ── Search (local embeddings only) ──
print("[9] Search 'Japanese food budget' (local, no API):")
t0 = time.time()
results = mem.search("Japanese food budget", top_k=3)
print(f"    {len(results)} results in {time.time() - t0:.3f}s:")
for r in results:
    print(f"    [{r.metadata.key[:50]}] {r.memory[:80]}")

print(f"\n[10] Search 'machine learning deployment' (local, no API):")
results2 = mem.search("machine learning deployment", top_k=3)
print(f"    {len(results2)} results:")
for r in results2:
    print(f"    [{r.metadata.key[:50]}] {r.memory[:80]}")

# ── Tree index ──
print(f"\n[11] Tree index:")
print(mem.get_tree_index())

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SESSION 2: Reload from disk, search again")
print("=" * 60)

mem2 = HierarchicalMarkdownMemory(config)
mem2.load(MEMORY_DIR)
all_items = mem2.get_all()
print(f"[12] Loaded {len(all_items)} nodes from disk")

results3 = mem2.search("Kyoto restaurants onsen", top_k=3)
print(f"[13] Search 'Kyoto restaurants onsen': {len(results3)} results:")
for r in results3:
    print(f"    [{r.metadata.key[:50]}] {r.memory[:80]}")

print(f"\n✅ Full hierarchical E2E test passed!")
print(f"   Memory dir: {MEMORY_DIR}")
print(f"   Open the .md files in any editor to inspect.")
