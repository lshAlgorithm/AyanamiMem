<div align="center">

# AyanamiMem

**A memory system for AI agents that lives entirely on your filesystem.**

Every memory is a plain `.md` file. Edges are Markdown links. The tree is a directory tree. Nothing is hidden in a database.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Built on MemOS](https://img.shields.io/badge/Built%20on-MemOS-purple)](https://github.com/MemTensor/MemOS)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-149%20passing-brightgreen)](#testing)

</div>

---

## What is this

AyanamiMem is a new memory backend built on top of [MemOS](https://github.com/MemTensor/MemOS). It replaces databases, vector stores, and graph engines with a single principle:

**The filesystem is the memory.**

Conversations become Markdown files. Related memories are linked with `[label](./relative/path.md)`. A background compactor groups and summarises them into a RAPTOR-style hierarchy. An agent navigates it with its existing `read_file` tool — no custom tools, no new protocols.

The result is a memory that you can open in Obsidian, commit to git, diff, revert, edit by hand, and share across sessions by copying a folder.

---

## The core idea

```
Conversation arrives
    │
    ▼  chunker
_fresh/001-i-love-python.md          ← raw verbatim chunk, depth=0
_fresh/002-planning-trip-to-japan.md
...

    │  (async, background)
    ▼  compactor
04-japan-trip-planning/              ← LLM-summarised topic, depth=2
  _summary.md                        ← "Japan trip: Kyoto, onsen, 3000 USD"
  01-japan-planning-details/         ← depth-1 subtopic
    _summary.md
    001-i-m-planning-a-trip.md       ← original leaf, untouched
    002-budget-is-3000-usd.md
  02-kyoto-temples/
    _summary.md
    003-temples-recommendation.md

    │  (on every agent query)
    ▼  assembler
Top-down tree walk:
  1. Embed query locally (sentence-transformers, no API)
  2. Compare against _embeddings.json at each directory level
  3. Descend into best-matching branch
  4. Select non-overlapping nodes at the right depth
  5. Return .md file contents with Markdown links intact

    │
    ▼  LLM context
## Relevant memories

[Japan trip: Kyoto, onsen, 3000 USD]
...budget around 3000 USD. Interested in Kyoto
temples, onsen, and local restaurants.

## Children
- [Planning details](./01-japan-planning-details/_summary.md)
- [Kyoto temples](./02-kyoto-temples/_summary.md)
```

The LLM sees the summary and its child links. If it needs more detail, it calls `read_file("./01-japan-planning-details/_summary.md")`. No custom tools. Progressive drill-down, bounded by depth.

---

## What exists today

### Three memory backends (all registered in MemOS factory)

| Backend key | Class | What it does |
|---|---|---|
| `markdown_text` | `MarkdownTextMemory` | Flat `.md` files. One per memory. Edges encoded as `- TYPE: uuid`. Local cosine search on `key` field. |
| `markdown_tree_text` | `MarkdownTreeTextMemory` | Adds parent/child edges, tree-walking helpers, auto `_index.md` TOC. |
| `hierarchical_markdown` | `HierarchicalMarkdownMemory` | Full RAPTOR-style hierarchy. Filesystem = tree. Async compaction. Relative-path Markdown links. |

### The hierarchical backend in detail

The main backend lives at `src/memos/memories/textual/hierarchical_markdown_memory.py` and is composed of seven focused components:

```
src/memos/memories/textual/hierarchical_markdown/
├── fs.py           # read/write .md, slugify, sequence numbering, edge parsing
├── embeddings.py   # _embeddings.json: file-locked atomic writes, concurrent-safe
├── chunker.py      # conversation messages → leaf .md files in _fresh/
├── summarizer.py   # depth-aware LLM summarisation, 3-level escalation
├── compactor.py    # cluster leaves by embedding sim, mkdir, move, summarise, cascade
├── assembler.py    # top-down tree walk, non-overlapping node selection
└── updater.py      # mark stale upward, re-summarise bottom-up
```

### The OpenClaw plugin

`apps/memos-markdown-openclaw/` is a lifecycle plugin for [OpenClaw](https://openclaw.ai) that wires AyanamiMem into every agent conversation automatically:

- **`before_agent_start`** → searches memory by local embeddings, injects context
- **`agent_end`** → chunks conversation into leaf files, triggers compaction
- No new tools exposed to the LLM. It reads child files via `read_file` following the Markdown links already in the context.

---

## File format

Every `.md` file has the same three-part structure:

```markdown
---
key: "Japan trip: Kyoto temples, onsen, 3000 USD budget"
depth: 2
node_kind: condensed
descendant_count: 5
earliest_at: "2026-03-20T10:00:00"
latest_at: "2026-03-23T15:12:00"
stale: false
token_count: 180
session_id: "session-abc"
tags: [travel, japan, kyoto]
---

User is planning a trip to Japan next month. Budget around 3000 USD.
Interested in Kyoto temples (Fushimi Inari, Kinkaku-ji), onsen experience,
and local food including ramen and conveyor belt sushi.

## Children
- [Planning details](./01-japan-planning-details/_summary.md)
- [Kyoto temples](./02-kyoto-temples/_summary.md)

## Related
- [Work ML project](../02-work-project/_summary.md)
```

- **YAML frontmatter** → all metadata fields (depth, kind, dates, tags)
- **Body** → the memory content, verbatim for leaves, LLM-generated for summaries
- **`## Children`** → links to subdirectories (the tree structure)
- **`## Related`** → semantic cross-links to other subtrees

Edges are plain Markdown relative links. `cat`, Obsidian, VS Code, GitHub — all render them correctly.

---

## How search works

No API call during search. Everything is local.

```python
# 1. Embed the query locally
query_vec = embedder.embed(["cherry blossom season Japan"])[0]

# 2. At root: read _embeddings.json (pre-computed keys), cosine similarity
# "Japan trip: Kyoto temples, onsen, 3000 USD" → 0.71
# "ML classifier for customer support" → 0.09
# → descend into japan-trip/

# 3. At japan-trip/_embeddings.json: compare subtopics
# "Planning details" → 0.68
# "Kyoto temples" → 0.74
# → descend into kyoto-temples/

# 4. Token budget reached → return kyoto-temples/_summary.md

# 5. Non-overlapping: if kyoto-temples/_summary.md selected,
#    its parent (japan-trip/_summary.md) and its children are excluded
```

The returned node is the most specific summary that fits the token budget. The LLM can read deeper via `read_file` on the child links it sees in the body.

---

## How compaction works

The compactor runs asynchronously after new leaves are added. It builds the tree bottom-up:

**Leaf pass** — triggered when `_fresh/` accumulates enough non-tail leaves:
1. Read `_embeddings.json` for all leaves in `_fresh/`
2. Agglomerative clustering by cosine similarity
3. For each cluster: LLM summarises → creates `NN-slug/` directory → moves leaf files in → writes `_summary.md`
4. Updates parent `_embeddings.json`

**Condensation pass** — triggered when enough depth-1 dirs exist at the same level:
1. Find all dirs at target depth in `memory_dir`
2. Cluster by their summary embeddings
3. For each cluster: LLM summarises the summaries → creates new parent dir → moves subdirs in → writes parent `_summary.md`
4. Cascade upward until no more condensation possible

**Stale propagation** — when a leaf is updated:
1. Walk upward, mark each ancestor `stale: true`
2. Background re-summarisation processes stale nodes deepest-first

---

## How it persists across sessions

Files are written to disk immediately on every mutation. A new session just calls `mem.load(memory_dir)`:

```python
# Session 1
mem = HierarchicalMarkdownMemory(config)
items = mem.extract(messages)   # writes to _fresh/*.md
mem.add(items)
mem.compact()                   # builds directory tree

# Session 2 — new process, same directory
mem2 = HierarchicalMarkdownMemory(config)
mem2.load(memory_dir)           # scans all .md files, rebuilds cache
results = mem2.search("Japan")  # works immediately
```

The directory itself is the persistent state. Git tracks every change.

---

## How git tracking works

Every mutation is a file system change, so every mutation is git-trackable:

```
memory: add 3 leaves from conversation       _fresh/001-*.md, 002-*.md, 003-*.md
memory: compact 8 leaves into japan-trip/    ← mkdir + rename + new _summary.md
memory: update leaf 001-*.md                 ← diff shows changed body
memory: re-summarise _summary.md (stale)     ← diff shows updated summary
memory: share japan-trip/ with session-def   ← symlink added
```

With `git_auto_commit: true` in config, commits are created automatically. Without it, you run `git commit` manually when you want a checkpoint.

---

## Quick start

### Install

```bash
git clone https://github.com/MemTensor/MemOS
cd MemOS
uv venv && uv pip install -e ".[all]"
uv pip install sentence-transformers pyyaml
```

### Run the demo

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # or any OpenAI-compatible endpoint
export MOS_CHAT_MODEL=deepseek/deepseek-chat           # or any chat model

# Simple demo: 3 conversations, flat _fresh/ + manual compact
.venv/bin/python examples/hierarchical_e2e_test.py

# Deep tree demo: 12 conversations, forced multi-level compaction
.venv/bin/python examples/deep_tree_e2e_test.py
```

### Use in Python

```python
from memos.configs.memory import HierarchicalMarkdownMemoryConfig
from memos.memories.textual.hierarchical_markdown_memory import HierarchicalMarkdownMemory

config = HierarchicalMarkdownMemoryConfig(
    memory_dir=".memos/my_agent",
    extractor_llm={
        "backend": "openai",
        "config": {
            "model_name_or_path": "deepseek/deepseek-chat",
            "api_key": "your_key",
            "api_base": "https://openrouter.ai/api/v1",
        },
    },
    embedder={
        "backend": "sentence_transformer",
        "config": {"model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2"},
    },
    fresh_tail_count=32,       # keep 32 most recent leaves always available
    condensed_min_fanout=4,    # group 4+ dirs into a parent
    leaf_chunk_tokens=2048,    # max tokens per leaf
)

mem = HierarchicalMarkdownMemory(config)

# Load previous sessions
mem.load(".memos/my_agent")

# Extract memories from a conversation
items = mem.extract([
    {"role": "user", "content": "I'm planning a trip to Japan..."},
    {"role": "assistant", "content": "Kyoto is wonderful!"},
])
mem.add(items)               # writes .md files immediately
mem.compact()                # trigger compaction (also runs async after add)

# Search (local embeddings, no API call)
results = mem.search("Japan travel plans", top_k=5)
for r in results:
    print(r.metadata.key, "→", r.memory[:80])

# Show the tree
print(mem.get_tree_index())
```

### Use the interactive chat loop

```bash
.venv/bin/python examples/markdown_memory_chat.py
```

This is a proactive memory-augmented chat loop. On every turn it automatically:
- Searches memory (local, no API)
- Injects context into the system prompt
- After the reply: extracts and stores new memories

Special commands: `/memories`, `/tree`, `/save`, `/quit`

---

## OpenClaw plugin

### Install

```bash
cd apps/memos-markdown-openclaw
openclaw plugins install .
```

### Configure (`~/.openclaw/openclaw.json`)

```json
{
  "plugins": {
    "entries": {
      "memos-markdown-openclaw-plugin": {
        "enabled": true
      }
    }
  }
}
```

Set environment variables:
```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export MOS_CHAT_MODEL=deepseek/deepseek-chat
export MEMORY_DIR=~/.openclaw/ayanami-mem       # where .md files live
```

Restart the gateway: `openclaw gateway restart`

### What happens automatically

| OpenClaw event | What AyanamiMem does | API call? |
|---|---|---|
| Plugin starts | Spawns `bridge.py`, loads `.md` tree from disk | No |
| User sends message | Top-down tree walk → inject context into system prompt | No (local) |
| Agent replies | Chunks conversation → `_fresh/*.md` → triggers compaction | Yes (LLM, 1 call) |
| Compaction | Groups similar leaves → builds `_summary.md` hierarchy | Yes (LLM, per cluster) |
| LLM needs detail | Reads child `.md` via existing `read_file` tool | No |

---

## Testing

### Unit + integration tests (no API required)

```bash
.venv/bin/python -m pytest tests/memories/textual/ -v
# 149 tests, all passing
```

Breakdown:
- `test_hm_fs.py` — 28 tests: file I/O primitives, slugify, sequencing
- `test_hm_embeddings.py` — 15 tests: atomic writes, concurrent read/write, versioning
- `test_hm_chunker.py` — 9 tests: chunking, token limits, metadata
- `test_hm_summarizer.py` — 9 tests: LLM escalation, fallback, update mode
- `test_hm_e2e.py` — 18 tests: full CRUD, search, extract+add, cross-session reload
- `test_markdown.py` — 31 tests: flat markdown backend
- `test_markdown_tree.py` — 20 tests: tree ops, traversal, move, delete-reparent
- `test_markdown_cross_session.py` — 19 tests: cross-session persistence, concurrent writes

### Live E2E tests (requires API key)

```bash
# Simple: 3 conversations, 1 compaction cycle
.venv/bin/python examples/hierarchical_e2e_test.py

# Deep: 12 conversations across 3 topics, forces multi-level tree
.venv/bin/python examples/deep_tree_e2e_test.py
```

The deep test produces a 3-level tree with real LLM-generated summaries:
```
depth-2: "Japan cherry blossom trip planning"
  depth-1: "Japan trip planning for cherry blossom season" (4 leaves)
  depth-1: "Planning a 2-week trip to Japan"

depth-2: "AWS model deployment strategies"
  depth-1: "Model deployment on AWS" (1 leaf)
  depth-1: "AWS Model Deployment Advice"

depth-2: "Kitchen remodel cost breakdown"
  depth-1: "kitchen remodel cost inquiry" (1 leaf)
  depth-1: "kitchen remodel cost inquiry"
```

---

## Architecture

```
OpenClaw Gateway
    │
    ├─ before_agent_start  →  POST /search  →  top-down tree walk
    │                                          local cosine similarity
    │                                          returns .md content with links
    │
    ├─ Agent generates (system prompt includes memory context + child links)
    │  LLM can call read_file("./child/_summary.md") to drill deeper
    │
    └─ agent_end  →  POST /extract  →  chunker  →  _fresh/*.md
                                       compactor (async)  →  directory tree

apps/memos-markdown-openclaw/
├── bridge.py      # Python HTTP server (stdlib only)
└── index.mjs      # OpenClaw JS plugin (ES module, no npm deps)

src/memos/memories/textual/
├── hierarchical_markdown/
│   ├── fs.py          # filesystem I/O
│   ├── embeddings.py  # _embeddings.json (async-safe)
│   ├── chunker.py     # messages → _fresh/*.md
│   ├── summarizer.py  # LLM summarisation
│   ├── compactor.py   # tree construction
│   ├── assembler.py   # search / context assembly
│   └── updater.py     # stale propagation
├── hierarchical_markdown_memory.py  # main class (BaseTextMemory)
├── markdown.py                      # flat markdown backend
└── markdown_tree.py                 # simple tree backend
```

---

## Design decisions and tradeoffs

**Why Markdown files instead of a database?**

A database is opaque. You can't open it in a text editor, commit individual records to git, or share a single memory by copying a file. The filesystem gives you all of those for free. The cost is that you can't do relational queries — but we don't need them. We need cosine similarity on keys, which is a numpy operation on an in-memory array.

**Why relative-path links instead of UUIDs?**

UUIDs require a lookup table to resolve. Relative paths are self-describing — the LLM can see `[Planning details](./01-planning/_summary.md)` and know exactly what to read. It works in any Markdown renderer. No custom tool needed.

**Why search on keys only, not full content?**

Keys are short (5–15 words), descriptive, and generated by the LLM during summarisation. They are purpose-built for retrieval. Full-content embedding is noisier and slower. The key is like a human-assigned index term — exact matches and near-misses are more meaningful.

**Why async compaction?**

Compaction is slow (it calls the LLM). Blocking the user on every message would add 2–5 seconds of latency. Instead, leaves accumulate in `_fresh/` and compaction runs in a background thread. The fresh tail (configurable, default 32 leaves) is always included verbatim in context, so recent conversations are always available even before compaction.

**Why is stale propagation bottom-up?**

If you update a leaf, its parent summary is now wrong. And its grandparent summary is wrong. Bottom-up means we re-summarise the leaf's parent first (from updated children), then the grandparent (from the updated parent), and so on. Top-down would give us stale children summarised into fresh parents.

---

## What's next (from PLAN.md)

The implementation plan has three features, of which Feature 1 is complete:

| Feature | Status | Description |
|---|---|---|
| **Hierarchical Tree Text Memory** | ✅ Done | RAPTOR-style summary tree as filesystem |
| **Session-Level Memory Isolation** | 🔲 Planned | One dir per session, `_sessions.json` grant model |
| **Cross-Session Subtree Sharing** | 🔲 Planned | Symlink-based sharing, git-trackable grants |

Session isolation and subtree sharing are designed but not yet implemented. See [PLAN.md](./PLAN.md) for the full specification.

---

## Relationship to MemOS

AyanamiMem is a fork/extension of [MemOS](https://github.com/MemTensor/MemOS). It:

- **Keeps** the full MemOS API (`BaseTextMemory`, `MemoryFactory`, configs, LLM/embedder factories)
- **Adds** three new backends (`markdown_text`, `markdown_tree_text`, `hierarchical_markdown`)
- **Does not change** any existing MemOS backends (`tree_text`, `general_text`, `naive_text`, etc.)
- **Replaces** the OpenClaw plugin's storage backend with the hierarchical markdown memory

All 149 new tests pass alongside the existing MemOS test suite. The existing MemOS functionality is unaffected.

---

## License

Apache 2.0 — same as MemOS.
