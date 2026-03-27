#!/usr/bin/env python3
"""Thin HTTP bridge: expose HierarchicalMarkdownMemory over localhost.

Launched as a subprocess by the OpenClaw plugin.  Listens on
``http://127.0.0.1:<port>`` and proxies JSON requests to the Python
hierarchical markdown memory backend.

Routes:
    POST /search   {query, top_k}       → [{id, key, memory, tags}]
    POST /extract  {messages}            → [{id, key, memory, tags}]
    POST /add      {items: [{key, memory, tags}]}  → {ids}
    POST /compact                        → triggers compaction
    GET  /memories                       → [{id, key, memory, tags}]
    GET  /tree                           → {tree: "..."}
    GET  /health                         → {status: "ok", count: N}
    POST /shutdown                       → stops the server

Environment:
    MEMORY_DIR       — root dir for the memory tree (default: ~/.openclaw/memos-markdown)
    OPENAI_API_KEY   — for LLM extraction + summarisation
    OPENAI_BASE_URL  — LLM API base
    MOS_CHAT_MODEL   — LLM model name (default: deepseek/deepseek-chat)
    EMBEDDER_MODEL   — local ST model (default: sentence-transformers/all-MiniLM-L6-v2)
    BRIDGE_PORT      — listen port (default: 18832)
"""

import json
import os
import signal
import sys
import threading

from http.server import BaseHTTPRequestHandler, HTTPServer

# Ensure src/ is on path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_root, "src"))

from memos.configs.memory import HierarchicalMarkdownMemoryConfig  # noqa: E402
from memos.memories.textual.hierarchical_markdown_memory import HierarchicalMarkdownMemory  # noqa: E402
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata  # noqa: E402


# ── Config ────────────────────────────────────────────────────────────────────

MEMORY_DIR = os.environ.get(
    "MEMORY_DIR",
    os.path.expanduser("~/.openclaw/memos-markdown"),
)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
CHAT_MODEL = os.environ.get("MOS_CHAT_MODEL", "deepseek/deepseek-chat")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PORT = int(os.environ.get("BRIDGE_PORT", "18832"))


def build_memory() -> HierarchicalMarkdownMemory:
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
        fresh_tail_count=32,
        condensed_min_fanout=4,
    )
    mem = HierarchicalMarkdownMemory(config)
    # Load existing tree if present
    if os.path.isdir(MEMORY_DIR):
        mem.load(MEMORY_DIR)
        count = len(mem.get_all())
        if count:
            print(f"[bridge] Loaded {count} memories from {MEMORY_DIR}", flush=True)
    return mem


MEM: HierarchicalMarkdownMemory | None = None
_lock = threading.Lock()


def get_mem() -> HierarchicalMarkdownMemory:
    global MEM
    if MEM is None:
        MEM = build_memory()
    return MEM


def _item_to_dict(item: TextualMemoryItem) -> dict:
    return {
        "id": item.id,
        "key": item.metadata.key or "",
        "memory": item.memory,
        "tags": item.metadata.tags or [],
    }


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        # Suppress default stderr logging
        pass

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        mem = get_mem()
        if self.path == "/health":
            self._json_response({"status": "ok", "count": len(mem.get_all())})
        elif self.path == "/memories":
            items = mem.get_all()
            self._json_response({"memories": [_item_to_dict(i) for i in items]})
        elif self.path == "/tree":
            tree = mem.get_tree_index()
            self._json_response({"tree": tree})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self) -> None:
        mem = get_mem()
        try:
            body = self._read_json()
        except Exception as e:
            self._json_response({"error": f"bad json: {e}"}, 400)
            return

        if self.path == "/search":
            query = body.get("query", "")
            top_k = body.get("top_k", 5)
            with _lock:
                results = mem.search(query, top_k=top_k)
            self._json_response(
                {
                    "results": [_item_to_dict(r) for r in results],
                }
            )

        elif self.path == "/extract":
            messages = body.get("messages", [])
            if not messages:
                self._json_response({"error": "no messages"}, 400)
                return
            with _lock:
                items = mem.extract(messages)
                ids = mem.add(items)
            self._json_response(
                {
                    "extracted": [_item_to_dict(i) for i in items],
                    "ids": ids,
                }
            )

        elif self.path == "/add":
            raw_items = body.get("items", [])
            items = []
            for raw in raw_items:
                items.append(
                    TextualMemoryItem(
                        memory=raw.get("memory", ""),
                        metadata=TextualMemoryMetadata(
                            key=raw.get("key", ""),
                            source=raw.get("source", "conversation"),
                            tags=raw.get("tags", []),
                        ),
                    )
                )
            with _lock:
                ids = mem.add(items)
            self._json_response({"ids": ids})

        elif self.path == "/compact":
            with _lock:
                mem.compact()
            self._json_response({"status": "compacted"})

        elif self.path == "/shutdown":
            self._json_response({"status": "shutting down"})
            threading.Thread(target=self.server.shutdown).start()

        else:
            self._json_response({"error": "not found"}, 404)


def main() -> None:
    print(f"[bridge] Starting on http://127.0.0.1:{PORT}", flush=True)
    print(f"[bridge] Memory dir: {MEMORY_DIR}", flush=True)
    print(f"[bridge] Model: {CHAT_MODEL}", flush=True)
    print(f"[bridge] Embedder: {EMBEDDER_MODEL}", flush=True)

    # Pre-load memory and embedder
    get_mem()

    server = HTTPServer(("127.0.0.1", PORT), BridgeHandler)

    def handle_signal(sig, frame):
        print("[bridge] Shutting down...", flush=True)
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(f"[bridge] Ready", flush=True)
    server.serve_forever()
    print("[bridge] Stopped", flush=True)


if __name__ == "__main__":
    main()
