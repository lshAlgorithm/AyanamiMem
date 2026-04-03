# Architecture

## Full data flow

```
User sends a message
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                   OpenClaw Gateway                         │
│                                                           │
│  before_agent_start                                       │
│    └─ JS plugin: POST /search  ─────────────────────┐    │
│                                                      ▼    │
│                                          Python bridge    │
│                                          Assembler        │
│                                          ├─ embed query   │
│                                          │  (local ST)    │
│                                          ├─ read          │
│                                          │  _embeddings.  │
│                                          │  json at each  │
│                                          │  tree level    │
│                                          └─ return .md    │
│                                             contents      │
│                                                │          │
│  prependContext ◄──────────────────────────────┘          │
│  (injected into system prompt)                            │
│                                                           │
│  Agent generates reply                                    │
│  (can call read_file on child links it sees in context)   │
│                                                           │
│  agent_end                                                │
│    └─ JS plugin: POST /extract  ────────────────────┐    │
│                                                      ▼    │
│                                          Python bridge    │
│                                          ChatChunker      │
│                                          ├─ split msgs    │
│                                          ├─ write .md     │
│                                          │  to _fresh/    │
│                                          └─ update        │
│                                             _embeddings.  │
│                                             json          │
│                                                           │
│                              (background thread)          │
│                                          Compactor        │
│                                          ├─ leaf_pass     │
│                                          │  cluster+move  │
│                                          └─ condensation  │
│                                             _pass         │
└───────────────────────────────────────────────────────────┘
        │
        ▼
 ~/.openclaw/ayanami-mem/
 ├── _fresh/                 ← uncompacted recent leaves
 │   ├── 001-*.md
 │   └── _embeddings.json
 ├── 01-japan-trip/
 │   ├── _summary.md         ← LLM-generated, depth=2
 │   ├── _embeddings.json
 │   ├── 01-planning/
 │   │   ├── _summary.md     ← LLM-generated, depth=1
 │   │   ├── 001-*.md        ← verbatim leaf
 │   │   └── 002-*.md
 │   └── 02-temples/
 │       ├── _summary.md
 │       └── 003-*.md
 └── 02-ml-project/
     └── ...
```

## Components

### `fs.py` — Filesystem I/O

The lowest layer. Knows nothing about LLMs or embeddings. Handles:

- `slugify(text)` → `"Japan trip planning!"` → `"japan-trip-planning"`
- `next_seq(dir)` → scans for `NNN-*.md` and `NN-*/`, returns next integer
- `leaf_filename(seq, key)` → `"003-budget-discussion.md"`
- `subdir_name(seq, key)` → `"02-kyoto-temples"`
- `write_md(path, body, meta, edges)` → YAML frontmatter + body + edge sections
- `read_md(path)` → `(frontmatter_dict, body, edges_dict)`
- `write_leaf(dir, body, meta)` → write `NNN-slug.md`, return filename
- `write_summary(dir, body, meta, children)` → write `_summary.md` with `## Children` links
- `list_children(dir)` → sorted `[(label, rel_path)]` of leaves and subdirs
- `walk_tree(dir)` → all `.md` paths recursive, sorted

### `embeddings.py` — `_embeddings.json`

One `_embeddings.json` per directory. Stores `{filename: [float, ...]}`. Used by the assembler for fast similarity search without parsing full `.md` files.

Concurrent-safe:
- `fcntl.flock()` for mutual exclusion across processes
- Atomic write via `tmp → rename` (POSIX guarantee)
- `_version` counter to detect stale reads
- Readers get either the old or new version, never a partial write

### `chunker.py` — Messages → Leaves

Takes a `MessageList`, splits into token-bounded chunks (default 2048 tokens), writes each as a leaf `.md` file in `_fresh/`. Updates `_fresh/_embeddings.json` with the key embeddings.

The key for each leaf is extracted from the first meaningful line of the chunk (stripping the `User:` / `Assistant:` prefix).

### `summarizer.py` — LLM Summarisation

Depth-aware: uses different prompts for leaf→depth-1, depth-1→depth-2, and depth-2+. Three-level escalation:

1. Normal prompt (temperature=0.2) — try first
2. Aggressive prompt (temperature=0.1, shorter target) — if summary too long
3. Deterministic truncation — if LLM fails or keeps returning long output

Returns `(summary_body, key, tags)`. The caller writes the `_summary.md`.

### `compactor.py` — Tree Construction

Two passes, called from `compact_incremental()`:

**`leaf_pass()`**:
1. Take non-tail leaves from `_fresh/` (sorted by sequence number, skip the last `fresh_tail_count`)
2. Read their embeddings from `_fresh/_embeddings.json`
3. Agglomerative clustering by cosine similarity (threshold configurable, default 0.5; lower values = more grouping)
4. For each cluster: summarise → `mkdir NN-slug/` → `os.rename` leaf files in → `write_summary` → update `_embeddings.json`

**`condensation_pass(depth)`**:
1. Find all direct-child dirs in `memory_dir` whose `_summary.md` has `depth == target_depth`
2. Read their embeddings from the parent `_embeddings.json`
3. Cluster by similarity
4. For each cluster of ≥ `condensed_min_fanout`: summarise → `mkdir` → `os.rename` subdirs in → write parent `_summary.md`

Lock file `_fresh/.compacting` prevents concurrent compaction. Crash-safe: if renamed before the summary is written, the `_fresh/` still has original files (rename is atomic per-file on POSIX).

### `assembler.py` — Search / Context Assembly

`assemble(query, budget_tokens, fresh_tail_count)`:

1. **Fresh tail**: Always include last `fresh_tail_count` leaves from `_fresh/` verbatim
2. **Top-down walk**: At each directory level, read `_embeddings.json`, compute cosine similarity of query embedding vs each child's key embedding, select best-scoring branches within token budget, descend
3. **Non-overlapping selection**: If a node is selected, its ancestors and descendants are excluded (the selected node already summarises/contains them)
4. Return `[{path, content, meta, depth, score}]`

### `updater.py` — Stale Propagation

When a leaf is updated (`mem.update()`):

1. `mark_stale(path)` walks upward to the root, setting `stale: true` in each ancestor's `_summary.md` frontmatter
2. `process_stale()` re-summarises stale nodes deepest-first (children are fresh before parents re-summarise)
3. Updates `_embeddings.json` at each level with the new summary key vector

### `HierarchicalMarkdownMemory` — Main Class

Implements `BaseTextMemory`. Wires all components together. The MemOS factory registers it as `"hierarchical_markdown"`.

Key methods:
- `extract(messages)` → delegates to `ChatChunker`, returns leaf items
- `add(items)` → writes leaves, triggers async compaction via background thread
- `search(query, top_k)` → delegates to `Assembler`
- `update(id, item)` → rewrites leaf, calls `Updater.mark_stale + process_stale`
- `delete(ids)` → removes files, marks parents stale
- `load(dir)` → scans filesystem, rebuilds path↔UUID cache
- `compact()` → synchronous `compactor.compact_incremental()`
- `get_tree_index()` → renders tree as Markdown string

## Plugin architecture

```
apps/memos-markdown-openclaw/
├── index.mjs        # OpenClaw JS lifecycle plugin
│                    # Spawns bridge.py as subprocess
│                    # before_agent_start → POST /search
│                    # agent_end → POST /extract
│
└── bridge.py        # Python HTTP server (stdlib http.server, no FastAPI)
                     # GET  /health    → {status, count}
                     # POST /search    → {results}
                     # POST /extract   → {ids, extracted}
                     # POST /add       → {ids}
                     # POST /compact   → {status}
                     # GET  /tree      → {tree}
                     # POST /shutdown  → stops server
```

The bridge is stdlib-only (no FastAPI, no uvicorn). It starts fast and has no external deps beyond `memos` itself.
