#!/usr/bin/env python3
"""Deep tree E2E test — multi-level compaction with real LLM summaries.

Feeds 12 conversations across 3 topics, compacts multiple times to
build a tree at least 2 levels deep, then verifies:
  1. Leaves are correctly moved from _fresh/ into topic directories
  2. _summary.md files exist with LLM-generated keys and content
  3. Condensation groups topic directories under higher-level parents
  4. Search finds relevant results via top-down tree walk
  5. Session 2 reloads the entire tree from disk

Usage:
    .venv/bin/python examples/deep_tree_e2e_test.py
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memos.configs.memory import HierarchicalMarkdownMemoryConfig
from memos.memories.textual.hierarchical_markdown.fs import FRESH_DIR, SUMMARY_FILENAME, read_md
from memos.memories.textual.hierarchical_markdown_memory import HierarchicalMarkdownMemory

MEMORY_DIR = "/tmp/deep_tree_e2e"
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")
EMBEDDER = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not API_KEY:
    print("❌ Set OPENAI_API_KEY (and OPENAI_BASE_URL if using OpenRouter)")
    sys.exit(1)

if os.path.isdir(MEMORY_DIR):
    shutil.rmtree(MEMORY_DIR)

# ── Config: low thresholds to force compaction ────────────────────────────────
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
        "config": {"model_name_or_path": EMBEDDER},
    },
    fresh_tail_count=2,  # keep only 2 most recent
    condensed_min_fanout=2,  # group 2+ dirs
    leaf_chunk_tokens=300,  # small chunks
)

mem = HierarchicalMarkdownMemory(config)
# Override compactor thresholds for aggressive compaction
mem.compactor.compact_threshold = 3
mem.compactor.condensed_min_fanout = 2
mem.compactor.similarity_threshold = 0.3  # lower to group related topics

# ── 12 conversations across 3 topics ─────────────────────────────────────────
CONVERSATIONS = [
    # Topic A: Japan travel (4 conversations)
    [
        {
            "role": "user",
            "content": "I'm planning a 2-week trip to Japan in April. Best time for cherry blossoms?",
        },
        {
            "role": "assistant",
            "content": "Late March to mid-April is peak sakura season. Tokyo blooms first, then Kyoto about a week later.",
        },
    ],
    [
        {
            "role": "user",
            "content": "What's the best way to get around Japan? Should I get a JR Pass?",
        },
        {
            "role": "assistant",
            "content": "Definitely get a 14-day JR Pass. It covers bullet trains, local trains, and some ferries. Costs about 50000 yen.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Recommend some must-visit temples in Kyoto for a first-timer.",
        },
        {
            "role": "assistant",
            "content": "Top picks: Kinkaku-ji (Golden Pavilion), Fushimi Inari's thousand torii gates, and Arashiyama bamboo grove. Start early to avoid crowds.",
        },
    ],
    [
        {
            "role": "user",
            "content": "What's the food budget per day in Japan? I want to try authentic ramen and sushi.",
        },
        {
            "role": "assistant",
            "content": "Budget about 3000-5000 yen per day. Conveyor belt sushi is 100-300 yen per plate. Station ramen shops serve great bowls for 800 yen.",
        },
    ],
    # Topic B: Machine learning project (4 conversations)
    [
        {
            "role": "user",
            "content": "I need to build a text classifier for customer support tickets. Which model should I use?",
        },
        {
            "role": "assistant",
            "content": "For production, fine-tune DistilBERT. It's 60% smaller than BERT with 97% of the accuracy. Use HuggingFace Transformers.",
        },
    ],
    [
        {
            "role": "user",
            "content": "How should I prepare my training data? I have 50k labeled tickets in CSV format.",
        },
        {
            "role": "assistant",
            "content": "Split 80/10/10 for train/val/test. Clean HTML tags, lowercase, truncate to 512 tokens. Use stratified sampling if classes are imbalanced.",
        },
    ],
    [
        {"role": "user", "content": "What's the best way to deploy the model? I'm using AWS."},
        {
            "role": "assistant",
            "content": "Use SageMaker for managed hosting or package as a Docker container on ECS. For real-time: API Gateway + Lambda with ONNX runtime.",
        },
    ],
    [
        {
            "role": "user",
            "content": "My model accuracy dropped after last week's retraining. How do I debug this?",
        },
        {
            "role": "assistant",
            "content": "Check for data drift: compare feature distributions. Look at confusion matrix per class. Check if new training data has label noise. Roll back if needed.",
        },
    ],
    # Topic C: Home renovation (4 conversations)
    [
        {
            "role": "user",
            "content": "I want to renovate my kitchen. What's the average cost for a full remodel?",
        },
        {
            "role": "assistant",
            "content": "A full kitchen remodel costs $15,000-$50,000 depending on size and finishes. Budget $5,000 for appliances, $8,000 for cabinets, $3,000 for countertops.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Should I go with quartz or granite countertops? My kitchen gets heavy use.",
        },
        {
            "role": "assistant",
            "content": "Quartz is better for heavy use: non-porous, no sealing needed, very durable. Granite looks beautiful but needs annual sealing. Quartz costs $50-150/sqft.",
        },
    ],
    [
        {"role": "user", "content": "What color cabinets are trending? I want a modern look."},
        {
            "role": "assistant",
            "content": "Navy blue and sage green are hot right now. White Shaker remains timeless. For modern: flat-panel in matte charcoal with brass hardware.",
        },
    ],
    [
        {
            "role": "user",
            "content": "My contractor says the permits will take 3 weeks. Is that normal?",
        },
        {
            "role": "assistant",
            "content": "3 weeks is typical for residential permits. Some cities take 6-8 weeks. You can start demolition planning while waiting.",
        },
    ],
]

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: Feed 12 conversations across 3 topics")
print("=" * 70)

total_t0 = time.time()
all_ids = []

for i, conv in enumerate(CONVERSATIONS, 1):
    topic = ["Japan travel", "ML project", "Home renovation"][(i - 1) // 4]
    print(f"  [{i:2d}/12] {topic}: extracting...", end=" ", flush=True)
    t0 = time.time()
    items = mem.extract(conv)
    ids = mem.add(items)
    all_ids.extend(ids)
    print(f"{len(ids)} leaf(s) in {time.time() - t0:.1f}s")

fresh_dir = os.path.join(MEMORY_DIR, FRESH_DIR)
leaf_count = len([f for f in os.listdir(fresh_dir) if f.endswith(".md") and not f.startswith("_")])
print(f"\n  → {leaf_count} leaves in _fresh/, {len(all_ids)} total IDs")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: First compaction (leaf_pass — groups leaves into depth-1 dirs)")
print("=" * 70)

t0 = time.time()
mem.compactor.leaf_pass()
elapsed = time.time() - t0
print(f"  Leaf pass completed in {elapsed:.1f}s")

# Count what we have now
dirs_created = [
    d
    for d in sorted(os.listdir(MEMORY_DIR))
    if os.path.isdir(os.path.join(MEMORY_DIR, d)) and not d.startswith("_")
]
fresh_remaining = len(
    [f for f in os.listdir(fresh_dir) if f.endswith(".md") and not f.startswith("_")]
)
print(f"  → {len(dirs_created)} topic dirs created, {fresh_remaining} leaves still in _fresh/")
for d in dirs_created:
    summary_path = os.path.join(MEMORY_DIR, d, SUMMARY_FILENAME)
    if os.path.isfile(summary_path):
        meta, body, edges = read_md(summary_path)
        child_count = len(edges.get("Children", []))
        print(f"    📁 {d}/")
        print(f"       key: {meta.get('key', '?')[:70]}")
        print(f"       depth: {meta.get('depth')}, children: {child_count}")
        print(f"       summary: {body[:100]}...")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Second compaction (condensation_pass — groups dirs into depth-2)")
print("=" * 70)

t0 = time.time()
condensed = mem.compactor.condensation_pass(depth=1)
elapsed = time.time() - t0
print(f"  Condensation pass completed in {elapsed:.1f}s, {condensed} dirs grouped")

# Show the full tree
print(f"\n  On-disk structure:")
for root, dirs, files in os.walk(MEMORY_DIR):
    dirs[:] = sorted(dirs)
    level = root.replace(MEMORY_DIR, "").count(os.sep)
    indent = "  " * (level + 1)
    dirname = os.path.basename(root) or os.path.basename(MEMORY_DIR)
    md_count = sum(1 for f in files if f.endswith(".md"))
    has_summary = SUMMARY_FILENAME in files
    marker = "📦" if has_summary else "📁"
    extra = f" [{md_count} .md]" if md_count else ""
    print(f"  {indent}{marker} {dirname}/{extra}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Inspect LLM-generated summaries at each depth")
print("=" * 70)

summaries_found = 0
for root, dirs, files in os.walk(MEMORY_DIR):
    dirs[:] = sorted(dirs)
    if SUMMARY_FILENAME in files:
        summary_path = os.path.join(root, SUMMARY_FILENAME)
        rel_path = os.path.relpath(summary_path, MEMORY_DIR)
        meta, body, edges = read_md(summary_path)
        depth = meta.get("depth", "?")
        key = meta.get("key", "?")
        children = edges.get("Children", [])
        summaries_found += 1
        print(f"\n  [{rel_path}]")
        print(f"  depth: {depth}  key: {key[:60]}")
        print(f"  children: {len(children)}")
        print(f"  summary: {body[:200]}")
        if children:
            for label, path in children[:5]:
                print(f"    → [{label}]({path})")
            if len(children) > 5:
                print(f"    ... and {len(children) - 5} more")

print(f"\n  Total _summary.md files: {summaries_found}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: Search (top-down tree walk, local embeddings only)")
print("=" * 70)

queries = [
    "cherry blossom season Japan",
    "machine learning model deployment AWS",
    "kitchen renovation countertop material",
    "train ticket price budget",
]
for q in queries:
    t0 = time.time()
    results = mem.search(q, top_k=3)
    elapsed = time.time() - t0
    print(f"\n  🔎 '{q}' → {len(results)} results ({elapsed:.3f}s):")
    for r in results[:3]:
        key = r.metadata.key[:50] if r.metadata.key else "?"
        body_preview = r.memory[:80].replace("\n", " ")
        print(f"    [{key}] {body_preview}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: Tree index")
print("=" * 70)
print(mem.get_tree_index())

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 7: Session 2 — reload from disk, verify tree intact")
print("=" * 70)

mem2 = HierarchicalMarkdownMemory(config)
mem2.load(MEMORY_DIR)
all_items = mem2.get_all()
print(f"  Loaded {len(all_items)} nodes from disk")

results2 = mem2.search("Japan ramen sushi food", top_k=3)
print(f"  Search 'Japan ramen sushi food' → {len(results2)} results:")
for r in results2[:3]:
    key = r.metadata.key[:50] if r.metadata.key else "?"
    print(f"    [{key}] {r.memory[:80].replace(chr(10), ' ')}")

total_elapsed = time.time() - total_t0
print(f"\n✅ Deep tree E2E test completed in {total_elapsed:.1f}s")
print(f"   Memory dir: {MEMORY_DIR}")
print(f"   Summaries: {summaries_found} LLM-generated _summary.md files")
print(f"   Open the .md files in any editor to inspect.")
