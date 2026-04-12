# AyanamiMem — OpenClaw Plugin

Session-isolated hierarchical memory with BM25+vector hybrid search.
Stores memories as **human-readable `.md` files**. No database, no Python bridge.

---

## What it does

Every conversation turn is automatically:
1. **Captured** — chunked into a leaf `.md` file in `_fresh/`
2. **Compacted** — asynchronously grouped by topic into LLM-summarised directories
3. **Recalled** — injected as context before each reply using local embeddings + BM25

Each agent gets a fully isolated session directory. Memories persist forever as plain files you can open in any editor.

```
~/.ayanamimem/sessions/
├── agent-main/              ← agent "main" session
│   ├── _fresh/              ← recent unprocessed leaves
│   └── 01-japan-trip/       ← LLM-summarised topic cluster
│       ├── _summary.md      ← summary (has ## Children links)
│       └── 001-japan-cherry-blossom-trip.md  ← verbatim leaf
├── agent-alice/             ← completely separate session
└── _public/                 ← visible to all sessions
```

---

## Install

```bash
# From the MemOS repo root:
cd apps/ayanamimem-openclaw
openclaw plugins install .

# Or install from npm (once published):
openclaw plugins install @memtensor/ayanamimem-openclaw-plugin
```

---

## Required configuration

Set these before starting the gateway:

```bash
# In ~/.openclaw/.env  OR  your shell profile:
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1   # or any OpenAI-compatible endpoint
MOS_CHAT_MODEL=deepseek/deepseek-chat           # model for memory extraction/summarisation
```

**Embeddings run locally** — `Xenova/all-MiniLM-L6-v2` via `@huggingface/transformers` (ONNX, ~23 MB download on first use, no API call needed).

---

## Optional configuration

In `~/.openclaw/openclaw.json` under `plugins.config.ayanamimem-openclaw-plugin`:

```json
{
  "memoryRoot":         "~/.ayanamimem/sessions",
  "topK":               5,
  "freshTailCount":     2,
  "compactThreshold":   3,
  "compactSimilarity":  0.3,
  "mergeThreshold":     0.7,
  "chatModel":          "deepseek/deepseek-chat",
  "embeddingModel":     "Xenova/all-MiniLM-L6-v2",
  "enableBm25":         true,
  "enableMmr":          false
}
```

All values can also be set via environment variables (`MEMORY_ROOT`, `SESSION_ID`, etc.).

---

## How it works

### Session isolation

Each agent gets its own session directory derived from `event.agentId`:

```
agent-main   → ~/.ayanamimem/sessions/agent-main/
agent-alice  → ~/.ayanamimem/sessions/agent-alice/
```

Sessions are completely isolated — no cross-contamination by default.

### Memory tree structure

Leaves start with heuristic filenames in `_fresh/`. When enough accumulate, the compactor:
1. Groups similar leaves by embedding cosine similarity
2. Calls the LLM once per cluster to generate a summary + topic key
3. Renames leaves to `NNN-<topic-key>.md`
4. Moves them into `NN-<topic-slug>/`
5. Writes `_summary.md` for the cluster
6. Cascades: if enough clusters exist, they get grouped into a depth-2 parent

The result is a self-organising tree where directory names and leaf filenames reflect the actual topics discussed.

### Search pipeline

For every `before_agent_start`:

```
Query → embed (local, ~1ms)
      → top-down tree walk using EmbeddingCache
      → BM25Plus scan of all leaf text (exact keyword match)
      → RRF merge of both ranked lists
      → path constraint (one node per root-to-leaf path)
      → optional MMR diversity re-ranking
      → inject as appendSystemContext + prependContext
```

No API call for search — 100% local embeddings + in-memory indexes.

---

## Tools available to the LLM

| Tool | Purpose |
|---|---|
| `memory_search` | Hybrid BM25+vector search with explicit query |
| `memory_tree` | Show full hierarchical tree; nodes are `read_file`-able |
| `memory_compact` | Force compaction now (normally automatic) |
| `memory_status` | Show leaf counts, tree depth, cache size |
| `memory_sessions` | List all sessions and grants |
| `memory_write_public` | Write to `_public/` — visible to all agents |
| `memory_grant` | Share this session's memories with another |
| `memory_revoke` | Revoke a previous share |

### `read_file` navigation

The `memory_tree` output includes relative paths. The LLM can follow them:

```
## Children
- [Planning phase](./01-planning/_summary.md)
- [Restaurant recommendations](./02-restaurants/_summary.md)
```

Calling `read_file("./01-planning/_summary.md")` returns the full node content — no custom tool needed.

---

## Sharing memories between sessions

```
# Agent A grants Agent B read access:
memory_grant(targetSessionId="agent-b")

# Agent B now sees Agent A's memories in search results.
# To revoke:
memory_revoke(targetSessionId="agent-b")

# To publish a memory to ALL agents:
memory_write_public(content="...", key="shared fact")
```

Grants are recorded in `_sessions.json` alongside the memory files.

---

## Compatibility with Python chat.py

The `.md` file format and embedding model (`all-MiniLM-L6-v2`) are identical between this plugin and `examples/chat.py`. Existing memory directories created by `chat.py` work without migration:

```bash
# Point the plugin at an existing chat.py memory dir:
MEMORY_ROOT=~/.ayanamimem/sessions
# The plugin loads the existing .md files and rebuilds the index on start.
```

---

## Offline mode

To avoid all HuggingFace network calls:

```bash
EMBEDDER_MODEL=~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

---

## Architecture

```
OpenClaw Gateway
    │
    ├── before_agent_start
    │   └── Assembler.assemble()
    │       ├── EmbeddingCache.childrenOf() → cosine similarity (local)
    │       └── BM25Index.search() → keyword match (local)
    │       → RRF + path constraint → inject context
    │
    └── agent_end
        ├── chunkMessages() → leaf .md files in _fresh/
        └── setImmediate → Compactor.compactIncremental()
            ├── agglomerativeCluster() → group by embedding similarity
            ├── Summarizer.summarize() → LLM call (one per cluster)
            └── rename + move + write _summary.md

src/
├── types.ts           ← AyanamiNode, SearchHit, ConvMessage, etc.
├── config.ts          ← PluginConfig, env resolution, defaults
├── fs-store.ts        ← .md read/write, slugify, nextSeq, walkTree
├── embedding.ts       ← Transformers.js local + remote fallback
├── emb-cache.ts       ← EmbeddingCache: Map<path, Float32Array>
├── bm25.ts            ← BM25Plus: 80-line pure JS, no deps
├── chunker.ts         ← Messages → leaf .md files
├── summarizer.ts      ← fetch() to LLM API
├── compactor.ts       ← Leaf clustering, move, summarise, cascade
├── assembler.ts       ← Top-down walk + BM25 + RRF + MMR
├── updater.ts         ← Mark stale, re-summarise on leaf update
├── session-manager.ts ← _sessions.json registry + grants
└── capture.ts         ← Strip OpenClaw envelope injections
```

---

## No Python required

Unlike the `memos-markdown-openclaw` bridge plugin, this plugin has **zero Python dependencies**:

| Component | Implementation |
|---|---|
| Embeddings | `@huggingface/transformers` (Transformers.js, ONNX) |
| BM25 search | Pure JavaScript, ~80 lines, no dependencies |
| Vector search | `Float32Array` cosine in a loop |
| RRF + MMR | Pure math functions, ~100 lines |
| File storage | Node.js `fs` + `js-yaml` frontmatter |
| LLM calls | `fetch()` to OpenAI-compatible endpoint |
