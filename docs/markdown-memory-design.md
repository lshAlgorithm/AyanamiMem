# Markdown Memory for MemOS — Design Document

## Overview

We built two new MemOS memory backends and an OpenClaw plugin that store agent memories as **human-readable Markdown files** with YAML frontmatter.  Every memory is a `.md` file you can open in any text editor.  Relationships between memories are encoded as edges inside the file itself.  Retrieval is done locally via sentence-transformer cosine similarity on the `key` field — no external API, no database, no vector server.

```
~/.openclaw/memos-markdown/
├── _index.md                          ← auto-generated table of contents
├── a1b2c3d4-...-uuid.md              ← "Japan travel plans"
├── e5f6g7h8-...-uuid.md              ←   └── "Kyoto temple visit"
└── i9j0k1l2-...-uuid.md              ←   └── "Budget and onsen"
```

---

## What We Built

### Layer 1: `markdown_text` — Flat Markdown Memory

**File:** `src/memos/memories/textual/markdown.py` (~270 lines)

A new MemOS backend (registered as `"markdown_text"` in the factory) where each `TextualMemoryItem` is persisted as a single `.md` file.  Implements all 12 methods of `BaseTextMemory`.

### Layer 2: `markdown_tree_text` — Tree-Structured Markdown Memory

**File:** `src/memos/memories/textual/markdown_tree.py` (~210 lines)

Extends `MarkdownTextMemory` with hierarchical organization.  Every node can have a parent (via a `PARENT` edge) and children.  Provides tree-walking helpers (`get_parent`, `get_children`, `get_subtree`, `get_path_to_root`, `get_depth`, `get_siblings`, `search_subtree`) plus an auto-generated `_index.md` table of contents.

### Layer 3: OpenClaw Plugin

**Directory:** `apps/memos-markdown-openclaw/`

A lifecycle plugin for OpenClaw that proactively recalls and captures memories on every agent turn.  Uses a thin Python HTTP bridge (`bridge.py`) that the JS plugin spawns as a subprocess.

### Test Suite

- `tests/memories/textual/test_markdown.py` — 31 tests (CRUD, search, edges, extract, load/dump)
- `tests/memories/textual/test_markdown_tree.py` — 20 tests (tree ops, traversal, move, delete-reparent, index)
- `tests/memories/textual/test_markdown_cross_session.py` — 19 tests (cross-session persistence, accumulation, concurrent writes)

**Total: 70 tests, all passing.**

---

## Key Design Decisions

### 1. One file per memory

Every memory is a single `{uuid}.md` file.  This gives us:

- **Human readability** — open in VS Code, Obsidian, or `cat`
- **Git-friendly** — each memory is a separate diff
- **No migration** — no schema, no binary format, no database version
- **Concurrent-safe** — two processes writing different memories don't conflict (different files)

### 2. YAML frontmatter + body + edges

Each `.md` file has three sections:

```markdown
---
id: a1b2c3d4-e5f6-7890-abcd-ef1234567890
key: "Japan travel plans"
status: activated
tags: [travel, Japan, budget]
source: conversation
updated_at: '2026-03-23T14:30:00'
---

The user is planning a trip to Japan next month.
They want to visit Kyoto temples, try onsen, and
eat sushi. Budget is around 3000 USD.

## Edges
- PARENT: 00000000-0000-0000-0000-000000000000
- RELATED: e5f6g7h8-i9j0-k1l2-m3n4-o5p6q7r8s9t0
```

- **Frontmatter** maps 1:1 to `TextualMemoryMetadata` fields
- **Body** is the `memory` content (the value)
- **`## Edges`** encodes graph relationships as `- TYPE: target-uuid` lines

### 3. Key-based retrieval (no full-text search)

The `search(query, top_k)` method embeds the `query` and compares it against the `key` fields of all stored memories using cosine similarity.  We embed **keys only**, not full bodies, because:

- Keys are short and descriptive (5-15 words) — they're what you'd search for
- Embedding short strings is faster and more discriminative
- The key acts as a semantic index — like a filename for your memory

The embedding model (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) runs fully locally.  No API call is made during search.

### 4. LLM extraction for structured memory

When new conversations arrive, the `extract(messages)` method sends them to an LLM with the prompt from `SIMPLE_STRUCT_MEM_READER_PROMPT`.  The LLM returns structured JSON with `key`, `value`, `tags` for each extracted memory.  This is the **only** step that requires an API call.

### 5. Tree structure via edges (not directories)

Tree relationships are encoded as regular edges inside each `.md` file, not as filesystem directories.  This means:

- The filesystem stays flat (one directory, all `.md` files)
- Moving a node in the tree is just rewriting one edge
- The tree is rebuilt from edges on load
- `_index.md` is a generated view, not the source of truth

---

## How Isolation Works

### Per-directory isolation

Each memory store is a directory.  Different users, agents, or projects get different directories:

```
~/.openclaw/memos-markdown/          ← agent "main"
~/.openclaw/memos-markdown-agent2/   ← agent "agent2"
~/projects/myapp/.memos/             ← project-specific
```

The `memory_dir` config parameter controls where `.md` files are stored.  Two `MarkdownTreeTextMemory` instances with different `memory_dir` values are completely isolated — they share no state.

### Per-session isolation

Within a single directory, sessions are identified by the `session_id` field in each memory's metadata.  You can filter by session in application code:

```python
all_mems = mem.get_all()
session_mems = [m for m in all_mems if m.metadata.session_id == "session-42"]
```

### Agent-level isolation (OpenClaw plugin)

The OpenClaw plugin spawns one bridge process per plugin instance.  Each bridge has its own `MEMORY_DIR`.  In multi-agent setups, different agents get different directories via environment variables.

---

## How Sharing Works

### Cross-session sharing (automatic)

All memories in a directory are visible to any instance that loads from that directory.  This is the default — Session 2 sees everything from Session 1:

```python
# Session 1
mem1 = MarkdownTreeTextMemory(config)
mem1.add([item_about_japan])
mem1.dump(memory_dir)

# Session 2 (new process, same directory)
mem2 = MarkdownTreeTextMemory(config)
mem2.load(memory_dir)
mem2.search("travel")  # finds Session 1's memory
```

### Cross-agent sharing (explicit)

Two agents can share a memory directory by pointing their `memory_dir` to the same path.  Or, one agent can dump its memories and another can load them:

```python
# Agent A dumps
agent_a_mem.dump("/shared/memories/")

# Agent B loads
agent_b_mem.load("/shared/memories/")
```

Since memories are plain `.md` files, you can also share them by copying files between directories, committing them to git, or syncing via Dropbox/rsync.

### Public vs private (edge convention)

There is no built-in access control at the file level.  However, you can use tags or a naming convention:

```yaml
tags: [public, shared-knowledge]
# or
tags: [private, agent-main-only]
```

Application code can filter by these tags during search.

---

## How Search Works

### Indexing

On startup (`load`) or after any mutation (`add`, `update`, `delete`), the backend rebuilds an in-memory index:

1. Collect all `key` fields from stored memories
2. Embed them in batch via `sentence-transformers` → numpy array of shape `(N, 384)`
3. Store as `self._key_vecs` (parallel to `self._ids`)

### Query

When `search(query, top_k)` is called:

1. Embed the query string → 384-dim vector
2. Compute cosine similarity against all key vectors: `scores = keys @ query / (||keys|| * ||query||)`
3. `argsort(-scores)[:top_k]` → return the top-k `TextualMemoryItem` objects

**Cost:** One local `sentence-transformers` forward pass (~1ms for short queries).  No network, no API key, no external service.

### Subtree search

The tree backend adds `search_subtree(root_id, query, top_k)` which searches only within the descendants of a given node.  Implementation: run full search with `top_k * 3`, then filter to the subtree ID set, then truncate to `top_k`.

---

## Worked Example: A Multi-Turn Conversation

### The conversation

```
User:  I'm planning a trip to Japan next month. I love sushi.
Asst:  Japan is wonderful! Kyoto temples are a must-see.
User:  I also want to try onsen. My budget is around 3000 USD.
```

### Step 1: LLM Extraction

The `extract()` method sends the conversation to DeepSeek with the structured extraction prompt.  The LLM returns:

```json
{
  "memory list": [
    {
      "key": "Trip to Japan Planning",
      "value": "The user is planning a trip to Japan next month. They love sushi and want to visit Kyoto temples, try onsen. Budget is around 3000 USD.",
      "tags": ["travel", "Japan", "Kyoto", "sushi", "onsen", "budget"]
    },
    {
      "key": "Kyoto Temple Recommendation",
      "value": "The assistant recommended Kyoto temples as a must-see destination in Japan.",
      "tags": ["travel", "Japan", "Kyoto", "recommendation"]
    }
  ]
}
```

### Step 2: Files on disk

After `add()` + `dump()`, two `.md` files appear:

**`aa1c21c9-7e8b-4377-bcb1-7641ae93b2d1.md`:**
```markdown
---
id: aa1c21c9-7e8b-4377-bcb1-7641ae93b2d1
key: Trip to Japan Planning
status: activated
tags:
- travel
- Japan
- Kyoto
- sushi
- onsen
- budget
source: conversation
updated_at: '2026-03-23T14:31:26'
---

The user is planning a trip to Japan next month.
They love sushi and want to visit Kyoto temples,
try onsen. Budget is around 3000 USD.
```

**`5153ca21-c684-4fcf-889b-58fe796e80c5.md`:**
```markdown
---
id: 5153ca21-c684-4fcf-889b-58fe796e80c5
key: Kyoto Temple Recommendation
status: activated
tags:
- travel
- Japan
- Kyoto
- recommendation
source: conversation
updated_at: '2026-03-23T14:31:26'
---

The assistant recommended Kyoto temples as a
must-see destination in Japan.

## Edges
- PARENT: aa1c21c9-7e8b-4377-bcb1-7641ae93b2d1
```

### Step 3: Tree structure (`_index.md`)

```markdown
# Memory Tree Index

- **Trip to Japan Planning** (`aa1c21c9`)
  - **Kyoto Temple Recommendation** (`5153ca21`)
```

### Step 4: Next conversation (search + context injection)

The user starts a new session and asks:

```
User:  What restaurants should I try in Kyoto?
```

The system runs `search("What restaurants should I try in Kyoto?", top_k=5)`.  The query is embedded locally, compared against the two stored keys:

| Key | Cosine similarity |
|---|---|
| "Trip to Japan Planning" | 0.61 |
| "Kyoto Temple Recommendation" | 0.72 |

Both are returned (top-2).  The system builds a context-augmented prompt:

```
SYSTEM PROMPT:
You are a knowledgeable and helpful AI assistant.
You have access to conversation memories that help you
provide more personalized responses.

## Relevant memories from previous conversations

1. [Kyoto Temple Recommendation] The assistant recommended
   Kyoto temples as a must-see destination in Japan.

2. [Trip to Japan Planning] The user is planning a trip to
   Japan next month. They love sushi and want to visit
   Kyoto temples, try onsen. Budget is around 3000 USD.

USER: What restaurants should I try in Kyoto?
```

The LLM now has full context from the previous session and can give a personalized answer that references the user's sushi preference, budget, and itinerary.

### Step 5: After the reply — new memories extracted

The assistant replies with specific restaurant recommendations.  The `agent_end` hook extracts new memories:

```markdown
# Memory Tree Index

- **Trip to Japan Planning** (`aa1c21c9`)
  - **Kyoto Temple Recommendation** (`5153ca21`)
  - **Kyoto Restaurant Recommendations** (`b7d8e9f0`)
```

And a new file appears:

**`b7d8e9f0-1a2b-3c4d-5e6f-7a8b9c0d1e2f.md`:**
```markdown
---
id: b7d8e9f0-1a2b-3c4d-5e6f-7a8b9c0d1e2f
key: Kyoto Restaurant Recommendations
status: activated
tags:
- travel
- Japan
- Kyoto
- food
- restaurants
source: conversation
updated_at: '2026-03-23T15:12:00'
---

The assistant recommended several restaurants in Kyoto:
Nishiki Market for street food, Kikunoi for kaiseki,
and Ippudo for ramen. The user's budget-friendly options
include conveyor belt sushi near Kyoto Station.

## Edges
- PARENT: aa1c21c9-7e8b-4377-bcb1-7641ae93b2d1
```

The tree grows organically as conversations continue, always persisted as editable `.md` files.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                   OpenClaw Gateway                   │
│                                                     │
│  before_agent_start ──→ POST /search ──→ context    │
│  agent generates reply                              │
│  agent_end ──→ POST /extract ──→ .md files          │
└──────────────────────────┬──────────────────────────┘
                           │ HTTP (localhost:18832)
┌──────────────────────────▼──────────────────────────┐
│               bridge.py (Python)                     │
│                                                     │
│  MarkdownTreeTextMemory                             │
│  ├── search()   → cosine similarity on key embeds   │
│  ├── extract()  → LLM structured extraction         │
│  ├── add()      → write .md file + update index     │
│  ├── load()     → scan dir, parse all .md files     │
│  └── dump()     → persist all to disk               │
│                                                     │
│  sentence-transformers/all-MiniLM-L6-v2 (local)     │
│  DeepSeek Chat via OpenRouter (LLM extraction only) │
└──────────────────────────┬──────────────────────────┘
                           │ filesystem
┌──────────────────────────▼──────────────────────────┐
│          ~/.openclaw/memos-markdown/                 │
│                                                     │
│  _index.md              (auto-generated TOC)        │
│  aa1c21c9-....md        (Trip to Japan Planning)    │
│  5153ca21-....md        (Kyoto Temple Recommendation)│
│  b7d8e9f0-....md        (Kyoto Restaurants)         │
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

### What requires an API call

| Operation | API call? | When |
|---|---|---|
| `search(query)` | ❌ Local | Every user turn (recall) |
| `extract(messages)` | ✅ LLM | Every 1-2 agent turns (capture) |
| `add(items)` | ❌ Local | After extraction |
| `load(dir)` | ❌ Local | On startup |
| `dump(dir)` | ❌ Local | After mutations |
| `get_parent/children/subtree` | ❌ Local | On demand |

### Files in the codebase

| File | Lines | Purpose |
|---|---|---|
| `src/memos/memories/textual/markdown.py` | ~270 | Flat markdown backend |
| `src/memos/memories/textual/markdown_tree.py` | ~210 | Tree-structured backend |
| `src/memos/configs/memory.py` | +30 | `MarkdownTextMemoryConfig` class |
| `src/memos/memories/factory.py` | +4 | Factory registration |
| `apps/memos-markdown-openclaw/bridge.py` | ~170 | Python HTTP bridge |
| `apps/memos-markdown-openclaw/index.mjs` | ~230 | OpenClaw JS plugin |
| `tests/memories/textual/test_markdown.py` | ~280 | 31 unit tests |
| `tests/memories/textual/test_markdown_tree.py` | ~230 | 20 tree tests |
| `tests/memories/textual/test_markdown_cross_session.py` | ~330 | 19 cross-session tests |
| `examples/markdown_memory_demo.py` | ~200 | Interactive CLI demo |
| `examples/markdown_memory_chat.py` | ~180 | Proactive memory chat |
| `examples/e2e_test.py` | ~90 | E2E test with real LLM |
