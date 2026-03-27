#!/usr/bin/env python3
"""Proactive memory-augmented chat using Markdown tree memory.

This is a self-contained chat loop that **automatically**:

1. BEFORE each reply: searches memory for context relevant to the user's query
2. GENERATES a reply: with memories injected into the system prompt
3. AFTER each reply: extracts structured memories from the conversation
                      and stores them as .md files

All memory persists to disk — restart the script and it picks up where it left off.

Usage:
    python examples/markdown_memory_chat.py

Environment:
    OPENAI_API_KEY   — required
    OPENAI_BASE_URL  — required (e.g. https://openrouter.ai/api/v1)
    MOS_CHAT_MODEL   — optional (default: deepseek/deepseek-chat)
    EMBEDDER_MODEL   — optional (default: sentence-transformers/all-MiniLM-L6-v2)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memos.configs.memory import MarkdownTextMemoryConfig
from memos.llms.factory import LLMFactory
from memos.memories.textual.markdown_tree import MarkdownTreeTextMemory


# ── Configuration from environment ────────────────────────────────────────────

MEMORY_DIR = os.path.abspath(
    os.environ.get(
        "MEMORY_DIR", os.path.join(os.path.dirname(__file__), "..", ".memos", "chat_memory")
    )
)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("MEMORY_TOP_K", "5"))
EXTRACT_INTERVAL = int(os.environ.get("EXTRACT_INTERVAL", "2"))  # extract every N turns

BASE_SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful AI assistant. "
    "You have access to conversation memories that help you provide more personalized responses. "
    "Use the memories to understand the user's context, preferences, and past interactions. "
    "If memories are provided, reference them naturally when relevant, "
    "but don't explicitly mention having memories unless asked."
)


def build_memory_config() -> MarkdownTextMemoryConfig:
    llm_config = {
        "backend": "openai",
        "config": {
            "model_name_or_path": CHAT_MODEL,
            "api_key": API_KEY,
            "api_base": API_BASE,
        },
    }
    return MarkdownTextMemoryConfig(
        memory_dir=MEMORY_DIR,
        extractor_llm=llm_config,
        embedder={
            "backend": "sentence_transformer",
            "config": {"model_name_or_path": EMBEDDER_MODEL},
        },
    )


class MemoryChat:
    """Chat assistant with proactive memory recall and extraction."""

    def __init__(self) -> None:
        self.config = build_memory_config()
        self.mem = MarkdownTreeTextMemory(self.config)
        self.chat_llm = LLMFactory.from_config(self.config.extractor_llm)
        self.chat_history: list[dict[str, str]] = []
        self.turn_count = 0

        # Load existing memories from disk
        if os.path.isdir(MEMORY_DIR):
            md_count = sum(
                1 for f in os.listdir(MEMORY_DIR) if f.endswith(".md") and not f.startswith("_")
            )
            if md_count > 0:
                self.mem.load(MEMORY_DIR)
                print(f"📚 Loaded {len(self.mem.get_all())} memories from previous sessions")

    def chat(self, user_query: str) -> str:
        """Process one user turn: recall → reply → extract → save."""

        # ── Step 1: RECALL — search memory for relevant context ───────────
        memories = self.mem.search(user_query, top_k=TOP_K)
        memory_context = ""
        if memories:
            memory_lines = [f"  {i}. {m.memory}" for i, m in enumerate(memories, 1)]
            memory_context = "\n\n## Relevant memories:\n" + "\n".join(memory_lines)
            print(f"  🧠 Recalled {len(memories)} memories")

        # ── Step 2: GENERATE — build prompt with memories, get LLM reply ──
        system_prompt = BASE_SYSTEM_PROMPT + memory_context
        messages = [
            {"role": "system", "content": system_prompt},
            *self.chat_history[-20:],  # last 20 turns as context window
            {"role": "user", "content": user_query},
        ]

        response = self.chat_llm.generate(messages)

        # ── Step 3: REMEMBER — update chat history ────────────────────────
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": response})
        self.turn_count += 1

        # ── Step 4: EXTRACT — periodically extract memories from recent conversation
        if self.turn_count % EXTRACT_INTERVAL == 0:
            self._extract_and_store()

        return response

    def _extract_and_store(self) -> None:
        """Extract memories from recent conversation turns and save to disk."""
        # Take the last few turns for extraction
        recent = self.chat_history[-(EXTRACT_INTERVAL * 2) :]
        if not recent:
            return

        try:
            print("  💾 Extracting memories from recent conversation...")
            items = self.mem.extract(recent)
            if items:
                ids = self.mem.add(items)
                self.mem.dump(MEMORY_DIR)
                print(f"  💾 Stored {len(ids)} new memories ({len(self.mem.get_all())} total)")
        except Exception as e:
            print(f"  ⚠️  Memory extraction failed: {e}")

    def save(self) -> None:
        """Force-save all memories and extract from any remaining conversation."""
        if self.chat_history and self.turn_count % EXTRACT_INTERVAL != 0:
            # Extract from remaining un-processed turns
            self._extract_and_store()
        self.mem.dump(MEMORY_DIR)
        print(f"💾 Saved {len(self.mem.get_all())} memories to {MEMORY_DIR}")

    def show_memories(self) -> None:
        """Print all stored memories."""
        all_mems = self.mem.get_all()
        if not all_mems:
            print("  (no memories yet)")
            return
        for i, m in enumerate(all_mems, 1):
            depth = self.mem.get_depth(m.id)
            indent = "  " * depth
            print(f"  {indent}{i}. [{m.metadata.key}] {m.memory[:80]}")

    def show_tree(self) -> None:
        """Print the memory tree."""
        idx = os.path.join(MEMORY_DIR, "_index.md")
        if os.path.isfile(idx):
            with open(idx) as f:
                print(f.read())
        else:
            print("  (empty tree)")


def main() -> None:
    if not API_KEY:
        print("❌ OPENAI_API_KEY is required. Set it in your environment.")
        sys.exit(1)

    print("🧠 MemOS Proactive Memory Chat")
    print(f"   Model:   {CHAT_MODEL}")
    print(f"   Memory:  {MEMORY_DIR}")
    print(f"   Extract: every {EXTRACT_INTERVAL} turns")
    print(f"   Recall:  top-{TOP_K} memories per query")
    print()
    print("   Special commands: /memories  /tree  /save  /quit")
    print()

    chat = MemoryChat()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            chat.save()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                chat.save()
                break
            elif cmd == "/memories":
                chat.show_memories()
            elif cmd == "/tree":
                chat.show_tree()
            elif cmd == "/save":
                chat.save()
            elif cmd == "/extract":
                chat._extract_and_store()
            else:
                print(f"  Unknown command: {user_input}")
            continue

        t0 = time.time()
        response = chat.chat(user_input)
        elapsed = time.time() - t0
        print(f"\nAssistant: {response}")
        print(f"  ({elapsed:.1f}s)\n")


if __name__ == "__main__":
    main()
