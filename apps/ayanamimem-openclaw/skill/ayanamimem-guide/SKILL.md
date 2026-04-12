---
name: ayanamimem-guide
description: How to use AyanamiMem — the session-isolated hierarchical memory system
version: 0.1.0
---

# AyanamiMem Memory Guide

## What is available

You have access to a persistent memory system called **AyanamiMem**. It automatically:
- Stores every conversation turn as a leaf `.md` file
- Groups related leaves into LLM-summarised topic directories
- Recalls relevant memories before each reply (you already see them above if any exist)

## When to use memory tools explicitly

Auto-recall handles most cases. Use tools when you need something specific:

### `memory_search` — find something in past conversations
```
Call when: user asks about something from a previous session, or you need to verify a past fact
Example: memory_search(query="Python project setup preferences")
```

### `memory_tree` — see the full memory structure
```
Call when: user asks "what do you remember about me?" or you want to survey all topics
Returns: a directory tree where each dir is a topic cluster
```
After `memory_tree`, you can call `read_file("<relpath>")` on any listed path to read that node's full content. Nodes include `## Children` links you can follow to drill deeper.

### `memory_compact` — force summarisation of pending leaves
```
Call when: you've just added many memories and want them organised immediately
Normally this happens automatically in the background
```

### `memory_write_public` — share a memory with all agents
```
Call when: a fact is useful to ALL agents, not just this conversation
Example: memory_write_public(content="The project uses Python 3.12", key="project Python version")
```

### `memory_sessions` — see all sessions
```
Call when: user asks about other agents or sessions, or you need to check sharing
```

### `memory_grant` / `memory_revoke` — control memory sharing
```
memory_grant(targetSessionId="alice")   → alice can now see your memories
memory_revoke(targetSessionId="alice")  → revoke alice's access
```

## How memory files are organised

```
~/.ayanamimem/sessions/
├── <agentId>/               ← your session (isolated from other agents)
│   ├── _fresh/              ← recent unprocessed conversation chunks
│   │   └── 001-<key>.md     ← verbatim conversation leaf
│   └── 01-<topic>/          ← LLM-summarised topic cluster
│       ├── _summary.md      ← summary node (has ## Children links)
│       └── 001-<key>.md     ← leaf with original conversation
└── _public/                 ← shared across all sessions
```

Each `.md` file you see in `memory_tree` can be read with `read_file(path)` to get:
- The full memory content
- `## Children` links (for summaries) you can follow to go deeper
- `## Related` links to semantically connected topics

## Memory retrieval quality

- **Auto-recall** uses vector similarity on topic keys — good for "what did we discuss about X"
- **`memory_search`** adds BM25 keyword search — better for exact names, numbers, code terms
- **`read_file` navigation** — best for deep-diving into a specific topic after finding it in the tree

## Important notes

- Your memories are **private to your agent ID** by default
- `_public/` memories are visible to all agents
- Memories persist across gateway restarts — everything is stored as `.md` files on disk
- You can tell users the path to their memory directory so they can read/edit it directly
