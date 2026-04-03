#!/usr/bin/env python3
"""Local interactive chat with AyanamiMem hierarchical memory.

Every conversation turn is chunked into a leaf .md file in _fresh/.
When enough leaves accumulate, the compactor groups similar ones into
a topic directory with an LLM-generated _summary.md, then cascades
upward into deeper summaries. Restart any time — the tree persists.

Usage:
    python examples/chat.py

Environment:
    OPENAI_API_KEY          — required
    OPENAI_BASE_URL         — required  (e.g. https://openrouter.ai/api/v1)
    MOS_CHAT_MODEL          — optional  (default: deepseek/deepseek-chat)
    MEMORY_DIR              — optional  (default: ~/.ayanamimem/chat)
    EMBEDDER_MODEL          — optional  (default: sentence-transformers/all-MiniLM-L6-v2)
                              Set to a local snapshot path to avoid network calls.
    COMPACT_THRESHOLD       — optional  (default: 3)
                              Non-tail leaves in _fresh/ before auto-compact triggers.
    COMPACT_SIMILARITY      — optional  (default: 0.3)
                              Cosine threshold for clustering leaves into topics.
    FRESH_TAIL              — optional  (default: 2)
                              Most-recent leaves always kept in _fresh/ (never compacted).
    MEMORY_TOP_K            — optional  (default: 5)

Commands during chat:
    /tree       show the current memory tree
    /fresh      show leaves pending compaction in _fresh/
    /status     show compaction thresholds and current state
    /files      list all .md files on disk
    /memories   list all nodes (leaves + summaries)
    /compact    force compaction now (bypasses threshold)
    /clear      wipe all memory (asks for confirmation)
    /quit       exit (auto-saves remaining turn)
"""

import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memos.configs.memory import HierarchicalMarkdownMemoryConfig
from memos.llms.factory import LLMFactory
from memos.memories.textual.hierarchical_markdown_memory import HierarchicalMarkdownMemory

# ── Config from environment ───────────────────────────────────────────────────

MEMORY_DIR = os.path.abspath(os.environ.get("MEMORY_DIR", os.path.expanduser("~/.ayanamimem/chat")))
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")
EMBEDDER = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COMPACT_THRESHOLD = int(os.environ.get("COMPACT_THRESHOLD", "3"))
COMPACT_SIMILARITY = float(os.environ.get("COMPACT_SIMILARITY", "0.3"))
FRESH_TAIL = int(os.environ.get("FRESH_TAIL", "2"))
TOP_K = int(os.environ.get("MEMORY_TOP_K", "5"))

# If EMBEDDER_MODEL is a local snapshot path, block all HuggingFace network calls
if os.path.isdir(EMBEDDER):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with persistent memory across conversations.\n"
    "Memories from previous sessions are provided below — use them naturally.\n"
    "Be concise. Do not mention having memory unless the user asks."
)


# ── Config builder ────────────────────────────────────────────────────────────


def _make_config() -> HierarchicalMarkdownMemoryConfig:
    llm = {
        "backend": "openai",
        "config": {
            "model_name_or_path": CHAT_MODEL,
            "api_key": API_KEY,
            "api_base": API_BASE,
        },
    }
    return HierarchicalMarkdownMemoryConfig(
        memory_dir=MEMORY_DIR,
        extractor_llm=llm,
        embedder={
            "backend": "sentence_transformer",
            "config": {"model_name_or_path": EMBEDDER},
        },
        fresh_tail_count=FRESH_TAIL,
        compact_threshold=COMPACT_THRESHOLD,
        compact_similarity_threshold=COMPACT_SIMILARITY,
        condensed_min_fanout=3,
        leaf_chunk_tokens=1024,
    )


# ── Chat class ────────────────────────────────────────────────────────────────


class Chat:
    def __init__(self) -> None:
        self.mem = HierarchicalMarkdownMemory(_make_config())
        self.llm = LLMFactory.from_config(self.mem.config.extractor_llm)
        self.history: list[dict] = []
        self.turn = 0

        # Reload existing tree from disk
        if os.path.isdir(MEMORY_DIR):
            self.mem.load(MEMORY_DIR)
            n = len(self.mem.get_all())
            if n:
                print(f"  📚 Reloaded {n} nodes from {MEMORY_DIR}")

    # ── One turn ──────────────────────────────────────────────────────────

    def say(self, user_msg: str) -> str:
        """Recall → reply → store → maybe compact."""

        # 1. Recall: top-down tree walk + fresh tail (local, no API)
        recalled = self.mem.search(user_msg, top_k=TOP_K)
        context_block = ""
        if recalled:
            lines = [f"  [{r.metadata.key}]\n  {r.memory[:200]}" for r in recalled]
            context_block = (
                "\n\n---\nMemories from past conversations:\n" + "\n\n".join(lines) + "\n---"
            )
            print(f"  🧠  recalled {len(recalled)} node(s)  (local)")

        # 2. LLM call with memory context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + context_block},
            *self.history[-16:],
            {"role": "user", "content": user_msg},
        ]
        reply = self.llm.generate(messages)

        # 3. Update history
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": reply})
        self.turn += 1

        # 4. Every 2 turns: chunk the last exchange into _fresh/ + maybe compact
        if self.turn % 2 == 0:
            self._store()

        return reply

    def _store(self) -> None:
        """Write the last 2 exchanges as a leaf in _fresh/, then show compaction state."""
        recent = self.history[-4:]
        if not recent:
            return

        try:
            items = self.mem.extract(recent)
            if not items:
                return

            ids = self.mem.add(items)  # writes to _fresh/; _maybe_compact() fires here

            # Show what happened
            leaves = self.mem.compactor._sorted_leaves()
            n_leaves = len(leaves)
            tail = self.mem.compactor.fresh_tail_count
            n_tail = min(n_leaves, tail)
            n_compact = n_leaves - tail
            threshold = self.mem.compactor.compact_threshold
            lock_path = os.path.join(self.mem.compactor._fresh_dir, ".compacting")
            compacting = os.path.exists(lock_path)

            if compacting:
                status = "⚙️  background compaction running..."
            elif n_compact >= threshold:
                status = f"⚙️  {n_compact} above threshold ({threshold}) → compacting in background"
            else:
                need = threshold - n_compact
                status = f"📋 {n_leaves} leaf/leaves in _fresh/ ({n_compact} compactable, need {need} more to auto-compact)"

            print(f"  💾  {len(ids)} leaf/leaves written  |  {status}")

        except Exception as e:
            print(f"  ⚠️  storage failed: {e}")

    # ── Commands ──────────────────────────────────────────────────────────

    def cmd_compact(self) -> None:
        """Force compaction via compact_incremental(force=True).

        Uses the same code path as auto-compact:
        - lock file prevents concurrent runs
        - leaf_pass runs regardless of threshold (force=True)
        - condensation cascades upward through ALL depths (not just depth 1)
        - lock released and cache rebuilt on exit
        """
        cmp = self.mem.compactor
        leaves_before = len(cmp._sorted_leaves())

        if leaves_before == 0:
            print("  _fresh/ is empty — nothing to compact.")
            return

        print(f"  🔨 Compacting  ({leaves_before} leaves in _fresh/, force=True)...")
        t0 = time.time()
        result = self.mem.compact(force=True)  # lock + full cascade + rebuild cache
        elapsed = time.time() - t0

        leaves_after = len(cmp._sorted_leaves())
        n_leaves = result["leaves_compacted"]
        n_dirs = result["dirs_condensed"]

        if n_leaves:
            print(f"     leaf pass:    {n_leaves} leaf/leaves → topic dir(s)")
            print(f"     _fresh/ now:  {leaves_after} leaf/leaves (tail)")
        else:
            print(f"     leaf pass:    nothing moved (all {leaves_before} within tail)")

        if n_dirs:
            print(f"     condensation: {n_dirs} dir(s) merged across all depths")

        print(f"     done in {elapsed:.1f}s")
        self.cmd_tree()

    def cmd_status(self) -> None:
        """Show compaction state, thresholds, tree depth."""
        cmp = self.mem.compactor
        leaves = cmp._sorted_leaves()
        n_leaves = len(leaves)
        tail = cmp.fresh_tail_count
        n_compact = max(0, n_leaves - tail)
        lock_path = os.path.join(cmp._fresh_dir, ".compacting")

        # Count dirs per depth
        depth_counts: dict[int, int] = {}
        for entry in os.listdir(self.mem.memory_dir):
            if entry.startswith("_"):
                continue
            full = os.path.join(self.mem.memory_dir, entry)
            if not os.path.isdir(full):
                continue
            summary = os.path.join(full, "_summary.md")
            if os.path.isfile(summary):
                try:
                    from memos.memories.textual.hierarchical_markdown.fs import read_md

                    meta, _, _ = read_md(summary)
                    d = meta.get("depth", 1)
                    depth_counts[d] = depth_counts.get(d, 0) + 1
                except Exception:
                    pass

        print()
        print(f"  _fresh/:")
        print(f"    {n_leaves} total leaf/leaves")
        print(f"    {min(n_leaves, tail)} tail  (always kept, never compacted)")
        print(f"    {n_compact} compactable  (threshold = {cmp.compact_threshold})")
        print(f"    {'⚙️  RUNNING' if os.path.exists(lock_path) else 'idle'}")
        print(f"  Thresholds:")
        print(f"    compact_threshold:  {cmp.compact_threshold} non-tail leaves")
        print(f"    condensed_min_fanout: {cmp.condensed_min_fanout} dirs")
        print(f"    similarity:         {cmp.similarity_threshold}")
        print(f"  Tree:")
        for depth in sorted(depth_counts):
            label = "condensed" if depth > 0 else "leaf dirs"
            print(f"    depth {depth}: {depth_counts[depth]} dir(s)  [{label}]")
        if not depth_counts:
            print("    (no compacted dirs yet)")
        print(f"  Cache: {len(self.mem._path_cache)} paths indexed")
        print()

    def cmd_fresh(self) -> None:
        """List leaves pending compaction in _fresh/ with their keys."""
        cmp = self.mem.compactor
        leaves = cmp._sorted_leaves()
        if not leaves:
            print("  _fresh/ is empty.")
            return

        tail = cmp.fresh_tail_count
        n_tail = min(len(leaves), tail)
        tail_set = set(leaves[-n_tail:])

        # Load key from embedded _embeddings.json if available
        from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingIndex
        from memos.memories.textual.hierarchical_markdown.fs import read_md

        emb_keys = EmbeddingIndex(cmp._fresh_dir).read_meta()

        print(
            f"\n  _fresh/  ({len(leaves)} leaves, {n_tail} tail, {max(0, len(leaves) - n_tail)} compactable):\n"
        )
        for fname in leaves:
            fpath = os.path.join(cmp._fresh_dir, fname)
            try:
                meta, _, _ = read_md(fpath)
                key = meta.get("key", "?")[:60]
            except Exception:
                key = "?"
            marker = "  tail" if fname in tail_set else "  will compact"
            print(f"    {fname:<50}  [{key}]{marker}")
        print()

    def cmd_tree(self) -> None:
        print(self.mem.get_tree_index())

    def cmd_files(self) -> None:
        for root, dirs, files in os.walk(MEMORY_DIR):
            dirs[:] = sorted(d for d in dirs)
            level = root.replace(MEMORY_DIR, "").count(os.sep)
            indent = "  " * level
            folder = os.path.basename(root) or MEMORY_DIR
            md = [f for f in sorted(files) if f.endswith(".md")]
            if md or level == 0:
                print(f"  {indent}{folder}/")
            for f in md:
                print(f"  {indent}  {f}")

    def cmd_memories(self) -> None:
        items = self.mem.get_all()
        if not items:
            print("  (none yet)")
            return
        for i, m in enumerate(items, 1):
            key = (m.metadata.key or "?")[:50]
            body = m.memory[:80].replace("\n", " ")
            print(f"  {i:3d}. [{key}]  {body}")

    def cmd_clear(self) -> None:
        confirm = input("  Wipe all memories? Type YES to confirm: ").strip()
        if confirm == "YES":
            self.mem.delete_all()
            print("  🗑️  Cleared.")
        else:
            print("  Cancelled.")

    def save_and_exit(self) -> None:
        if self.turn % 2 != 0:
            self._store()
        print(f"\n  Memory lives in {MEMORY_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

HELP = textwrap.dedent("""\
    Commands:
      /tree       show the memory tree (directories = topic summaries)
      /fresh      show leaves pending compaction in _fresh/
      /status     show thresholds, leaf counts, cache size
      /files      show all .md files on disk
      /memories   list all nodes (leaves + summaries)
      /compact    force compaction now (bypasses threshold)
      /clear      wipe all memory
      /quit       exit
""")


def main() -> None:
    if not API_KEY:
        print("❌  OPENAI_API_KEY is not set.")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════╗")
    print("║         AyanamiMem  —  local chat                ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  model:      {CHAT_MODEL}")
    print(f"  memory:     {MEMORY_DIR}")
    print(f"  embedder:   {EMBEDDER[:60]}")
    print(f"  recall:     top-{TOP_K} nodes (local search)")
    print(
        f"  compact:    after {COMPACT_THRESHOLD} non-tail leaves  "
        f"(tail={FRESH_TAIL}, similarity={COMPACT_SIMILARITY})"
    )
    print()
    print(HELP)

    chat = Chat()

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            chat.save_and_exit()
            break

        if not raw:
            continue

        if raw.startswith("/"):
            cmd = raw.lower().split()[0]
            if cmd in ("/quit", "/exit", "/q"):
                chat.save_and_exit()
                break
            elif cmd == "/tree":
                chat.cmd_tree()
            elif cmd == "/fresh":
                chat.cmd_fresh()
            elif cmd == "/status":
                chat.cmd_status()
            elif cmd == "/files":
                chat.cmd_files()
            elif cmd == "/memories":
                chat.cmd_memories()
            elif cmd == "/compact":
                chat.cmd_compact()
            elif cmd == "/clear":
                chat.cmd_clear()
            elif cmd == "/help":
                print(HELP)
            else:
                print(f"  Unknown: {raw}. Type /help.")
            continue

        t0 = time.time()
        reply = chat.say(raw)
        print(f"\nAssistant: {reply}")
        print(f"  ({time.time() - t0:.1f}s)\n")


if __name__ == "__main__":
    main()
