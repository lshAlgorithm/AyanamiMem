#!/usr/bin/env python3
"""Local interactive chat with AyanamiMem hierarchical memory.

On startup you are asked for a session name.  Each session gets its own
isolated memory directory.  The same root directory holds all sessions and
a ``_public/`` area visible to every session.

Usage:
    python examples/chat.py

Environment:
    OPENAI_API_KEY          — required
    OPENAI_BASE_URL         — required  (e.g. https://openrouter.ai/api/v1)
    MOS_CHAT_MODEL          — optional  (default: deepseek/deepseek-chat)
    MEMORY_ROOT             — optional  (default: ~/.ayanamimem/sessions)
    SESSION_ID              — optional  skip the prompt and use this session directly
    EMBEDDER_MODEL          — optional  (default: sentence-transformers/all-MiniLM-L6-v2)
                              Set to a local snapshot path to avoid network calls.
    COMPACT_THRESHOLD       — optional  (default: 3)
    COMPACT_SIMILARITY      — optional  (default: 0.3)
    FRESH_TAIL              — optional  (default: 2)
    MEMORY_TOP_K            — optional  (default: 5)

Commands during chat:
    /tree       show the memory tree for this session
    /fresh      show leaves pending compaction
    /status     show compaction state and thresholds
    /files      list all .md files in this session
    /memories   list all nodes (leaves + summaries)
    /compact    force compaction now (bypasses threshold)
    /sessions   list all sessions in this root
    /grant <s>  share your memories with session <s>
    /revoke <s> revoke share from session <s>
    /public     write last memory to _public/ (visible to all sessions)
    /clear      wipe this session's memory
    /quit       exit (auto-saves remaining turn)
"""

import os
import re
import sys
import textwrap
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memos.configs.memory import HierarchicalMarkdownMemoryConfig
from memos.llms.factory import LLMFactory
from memos.mem_user.session_manager import SessionManager
from memos.memories.textual.hierarchical_markdown.fs import FRESH_DIR, read_md
from memos.memories.textual.hierarchical_markdown_memory import HierarchicalMarkdownMemory

# ── Config from environment ───────────────────────────────────────────────────

MEMORY_ROOT = os.path.abspath(
    os.environ.get("MEMORY_ROOT", os.path.expanduser("~/.ayanamimem/sessions"))
)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")
EMBEDDER = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COMPACT_THRESHOLD = int(os.environ.get("COMPACT_THRESHOLD", "3"))
COMPACT_SIMILARITY = float(os.environ.get("COMPACT_SIMILARITY", "0.3"))
FRESH_TAIL = int(os.environ.get("FRESH_TAIL", "2"))
TOP_K = int(os.environ.get("MEMORY_TOP_K", "5"))
ENV_SESSION_ID = os.environ.get("SESSION_ID", "")

if os.path.isdir(EMBEDDER):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with persistent memory across conversations.\n"
    "Memories from previous sessions are provided below — use them naturally.\n"
    "Be concise. Do not mention having memory unless the user asks."
)

_SLUG_RE = re.compile(r"[^a-z0-9_-]")


def _slugify_session(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower().strip()).strip("-") or "default"


# ── Session selection ─────────────────────────────────────────────────────────


def _pick_session() -> str:
    """Prompt the user for a session name, showing existing sessions."""
    if ENV_SESSION_ID:
        return _slugify_session(ENV_SESSION_ID)

    os.makedirs(MEMORY_ROOT, exist_ok=True)
    mgr = SessionManager(MEMORY_ROOT)
    existing = mgr.list_sessions()

    print()
    if existing:
        print("  Existing sessions:")
        for s in sorted(existing, key=lambda x: x.get("created_at", "")):
            sid = s["session_id"]
            mem_count = _count_memories(mgr.session_dir(sid))
            active = "●" if s.get("is_active") else "○"
            print(f"    {active}  {sid:<20}  ({mem_count} memories)")
        print()
        default = existing[-1]["session_id"]
        raw = input(f"  Session name [{default}]: ").strip()
        return _slugify_session(raw) if raw else default
    else:
        print("  No sessions yet.  Enter a name to create one.")
        raw = input("  Session name [default]: ").strip()
        return _slugify_session(raw) if raw else "default"


def _count_memories(session_path: str) -> int:
    """Quick count of .md files in a session directory."""
    if not os.path.isdir(session_path):
        return 0
    count = 0
    for root, _dirs, files in os.walk(session_path):
        count += sum(1 for f in files if f.endswith(".md") and not f.startswith("_"))
    return count


# ── Config builder ────────────────────────────────────────────────────────────


def _make_config(session_id: str) -> HierarchicalMarkdownMemoryConfig:
    llm = {
        "backend": "openai",
        "config": {
            "model_name_or_path": CHAT_MODEL,
            "api_key": API_KEY,
            "api_base": API_BASE,
        },
    }
    return HierarchicalMarkdownMemoryConfig(
        memory_dir=MEMORY_ROOT,
        session_isolation=True,
        session_id=session_id,
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
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        config = _make_config(session_id)
        self.mem = HierarchicalMarkdownMemory(config)
        self.llm = LLMFactory.from_config(self.mem.config.extractor_llm)
        self.history: list[dict] = []
        self.turn = 0

        # Reload existing memories for this session
        session_path = self.mem.memory_dir
        if os.path.isdir(session_path):
            self.mem.load(session_path)
            n = len(self.mem.get_all())
            if n:
                print(f"  📚 Loaded {n} node(s) from session '{session_id}'")

    # ── One turn ──────────────────────────────────────────────────────────

    def say(self, user_msg: str) -> str:
        recalled = self.mem.search(user_msg, top_k=TOP_K)
        context_block = ""
        if recalled:
            lines = [f"  [{r.metadata.key}]\n  {r.memory[:200]}" for r in recalled]
            context_block = (
                "\n\n---\nMemories from past conversations:\n" + "\n\n".join(lines) + "\n---"
            )
            print(f"  🧠  recalled {len(recalled)} node(s)  (local)")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + context_block},
            *self.history[-16:],
            {"role": "user", "content": user_msg},
        ]
        reply = self.llm.generate(messages)

        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": reply})
        self.turn += 1

        if self.turn % 2 == 0:
            self._store()

        return reply

    def _store(self) -> None:
        recent = self.history[-4:]
        if not recent:
            return
        try:
            items = self.mem.extract(recent)
            if not items:
                return
            ids = self.mem.add(items)

            leaves = self.mem.compactor._sorted_leaves()
            n_leaves = len(leaves)
            tail = self.mem.compactor.fresh_tail_count
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
                status = f"📋 {n_leaves} leaf/leaves ({n_compact} compactable, need {need} more)"

            print(f"  💾  {len(ids)} leaf/leaves written  |  {status}")

        except Exception as e:
            print(f"  ⚠️  storage failed: {e}")

    # ── Commands ──────────────────────────────────────────────────────────

    def cmd_compact(self) -> None:
        cmp = self.mem.compactor
        leaves_before = len(cmp._sorted_leaves())
        if leaves_before == 0:
            print("  _fresh/ is empty — nothing to compact.")
            return
        print(f"  🔨 Compacting  ({leaves_before} leaves, force=True)...")
        t0 = time.time()
        result = self.mem.compact(force=True)
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
            print(f"     condensation: {n_dirs} dir(s) merged")
        print(f"     done in {elapsed:.1f}s")
        self.cmd_tree()

    def cmd_status(self) -> None:
        cmp = self.mem.compactor
        leaves = cmp._sorted_leaves()
        n_leaves = len(leaves)
        tail = cmp.fresh_tail_count
        n_compact = max(0, n_leaves - tail)
        lock_path = os.path.join(cmp._fresh_dir, ".compacting")
        depth_counts: dict[int, int] = {}
        for entry in os.listdir(self.mem.memory_dir):
            if entry.startswith("_"):
                continue
            full = os.path.join(self.mem.memory_dir, entry)
            summary = os.path.join(full, "_summary.md")
            if os.path.isdir(full) and os.path.isfile(summary):
                try:
                    meta, _, _ = read_md(summary)
                    d = meta.get("depth", 1)
                    depth_counts[d] = depth_counts.get(d, 0) + 1
                except Exception:
                    pass
        print()
        print(f"  Session:   {self.session_id}")
        print(f"  Dir:       {self.mem.memory_dir}")
        print(
            f"  _fresh/:   {n_leaves} leaves  ({min(n_leaves, tail)} tail, {n_compact} compactable)"
        )
        print(f"  Status:    {'⚙️  COMPACTING' if os.path.exists(lock_path) else 'idle'}")
        print(f"  Threshold: {cmp.compact_threshold} non-tail leaves")
        print(f"  Similarity:{cmp.similarity_threshold}  merge:{cmp.merge_threshold}")
        print(f"  Tree:")
        for depth in sorted(depth_counts):
            print(f"    depth {depth}: {depth_counts[depth]} dir(s)")
        if not depth_counts:
            print("    (no compacted dirs yet)")
        print(f"  Cache:     {len(self.mem._path_cache)} paths indexed")
        print()

    def cmd_fresh(self) -> None:
        cmp = self.mem.compactor
        leaves = cmp._sorted_leaves()
        if not leaves:
            print("  _fresh/ is empty.")
            return
        tail = cmp.fresh_tail_count
        n_tail = min(len(leaves), tail)
        tail_set = set(leaves[-n_tail:])
        print(
            f"\n  _fresh/  ({len(leaves)} leaves, "
            f"{n_tail} tail, {max(0, len(leaves) - n_tail)} compactable):\n"
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
        base = self.mem.memory_dir
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs)
            level = root.replace(base, "").count(os.sep)
            indent = "  " * level
            folder = os.path.basename(root) or base
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

    def cmd_sessions(self) -> None:
        if self.mem.session_manager is None:
            print("  Session isolation is not active.")
            return
        sessions = self.mem.session_manager.list_sessions()
        print()
        print(f"  Root:  {MEMORY_ROOT}")
        print(f"  Sessions ({len(sessions)}):")
        for s in sorted(sessions, key=lambda x: x.get("created_at", "")):
            sid = s["session_id"]
            active = "● you " if sid == self.session_id else "  "
            n = _count_memories(self.mem.session_manager.session_dir(sid))
            flag = "active" if s.get("is_active") else "inactive"
            print(f"    {active}{sid:<22}  {n:>5} memories  [{flag}]")
        grants = self.mem.session_manager.get_grants_for(self.session_id)
        if grants:
            print(f"  Grants to this session ({len(grants)}):")
            for g in grants:
                print(f"    ← from {g['granting_session']}  scope={g['scope']}")
        print()

    def cmd_grant(self, target: str) -> None:
        if not target:
            print("  Usage: /grant <session_id>")
            return
        try:
            self.mem.grant_access(target, scope="read")
            print(f"  ✅ Shared '{self.session_id}' with '{target}'  (recorded in _sessions.json)")
            print(f"     '{target}' will see your memories in /search results.")
        except RuntimeError as e:
            print(f"  ❌ {e}")

    def cmd_revoke(self, target: str) -> None:
        if not target:
            print("  Usage: /revoke <session_id>")
            return
        try:
            self.mem.revoke_access(target)
            print(f"  ✅ Revoked access for '{target}'")
        except RuntimeError as e:
            print(f"  ❌ {e}")

    def cmd_public(self) -> None:
        """Write the last memory to _public/ (visible to all sessions)."""
        all_items = self.mem.get_all()
        if not all_items:
            print("  No memories to publish.")
            return
        last = all_items[-1]
        ids = self.mem.write_public([last])
        print(f"  📢 Published to _public/: [{last.metadata.key}]  ({ids[0][:8]}...)")

    def cmd_clear(self) -> None:
        confirm = input(f"  Wipe all memories for session '{self.session_id}'? Type YES: ").strip()
        if confirm == "YES":
            self.mem.delete_all()
            print("  🗑️  Cleared.")
        else:
            print("  Cancelled.")

    def save_and_exit(self) -> None:
        if self.turn % 2 != 0:
            self._store()
        print(f"\n  Session '{self.session_id}' saved → {self.mem.memory_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

HELP = textwrap.dedent("""\
    Commands:
      /tree           show the memory tree
      /fresh          show leaves pending compaction
      /status         show session info, thresholds, cache size
      /files          list all .md files in this session
      /memories       list all nodes (leaves + summaries)
      /compact        force compaction now
      /sessions       list all sessions in this root
      /grant <s>      share your memories with session <s>
      /revoke <s>     revoke share from session <s>
      /public         copy last memory to _public/ (all sessions can see it)
      /clear          wipe this session's memory
      /quit           exit
""")


def main() -> None:
    if not API_KEY:
        print("❌  OPENAI_API_KEY is not set.")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════╗")
    print("║         AyanamiMem  —  local chat                ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  model:    {CHAT_MODEL}")
    print(f"  root:     {MEMORY_ROOT}")
    print(f"  embedder: {EMBEDDER[:60]}")
    print(f"  recall:   top-{TOP_K}  |  compact after {COMPACT_THRESHOLD} non-tail leaves")

    session_id = _pick_session()
    print(f"\n  ► Session: {session_id}\n")
    print(HELP)

    chat = Chat(session_id)

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
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

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
            elif cmd == "/sessions":
                chat.cmd_sessions()
            elif cmd == "/grant":
                chat.cmd_grant(arg)
            elif cmd == "/revoke":
                chat.cmd_revoke(arg)
            elif cmd == "/public":
                chat.cmd_public()
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
