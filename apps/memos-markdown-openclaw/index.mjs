/**
 * MemOS Markdown Memory — OpenClaw Plugin
 *
 * Lightweight lifecycle plugin that stores memories as human-readable
 * Markdown files with tree structure.  Uses a Python bridge subprocess
 * for the actual memory operations (search, extract, add).
 *
 * Hooks:
 *   before_agent_start  → search memory, inject context
 *   agent_end            → extract & store memories from conversation
 *
 * Tools:
 *   memory_search  — search stored memories
 *   memory_tree    — show the memory tree index
 */

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as path from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BRIDGE_PORT = parseInt(process.env.BRIDGE_PORT || "18832", 10);
const BRIDGE_URL = `http://127.0.0.1:${BRIDGE_PORT}`;
const BRIDGE_SCRIPT = path.join(__dirname, "bridge.py");

// Find python — prefer the MemOS .venv, then system
function findPython() {
  const candidates = [
    path.join(__dirname, "..", "..", ".venv", "bin", "python"),
    path.join(__dirname, "..", "..", ".venv", "bin", "python3"),
    "python3",
    "python",
  ];
  for (const p of candidates) {
    try {
      const { execFileSync } = require("node:child_process");
      execFileSync(p, ["--version"], { stdio: "pipe", timeout: 5000 });
      return p;
    } catch { /* try next */ }
  }
  return "python3";
}

let bridgeProcess = null;
let bridgeReady = false;

async function ensureBridge(log) {
  if (bridgeReady) return true;

  // Check if bridge is already running
  try {
    const res = await fetch(`${BRIDGE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      bridgeReady = true;
      const data = await res.json();
      log.info(`memos-md: bridge already running (${data.count} memories)`);
      return true;
    }
  } catch { /* not running, start it */ }

  // Start bridge subprocess
  const pythonBin = findPython();
  log.info(`memos-md: starting bridge: ${pythonBin} ${BRIDGE_SCRIPT}`);

  bridgeProcess = spawn(pythonBin, [BRIDGE_SCRIPT], {
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, BRIDGE_PORT: String(BRIDGE_PORT) },
    detached: false,
  });

  bridgeProcess.stdout.on("data", (d) => {
    const line = d.toString().trim();
    if (line) log.info(`memos-md: ${line}`);
    if (line.includes("Ready")) bridgeReady = true;
  });
  bridgeProcess.stderr.on("data", (d) => {
    const line = d.toString().trim();
    if (line) log.warn(`memos-md: ${line}`);
  });
  bridgeProcess.on("exit", (code) => {
    log.warn(`memos-md: bridge exited (code=${code})`);
    bridgeReady = false;
    bridgeProcess = null;
  });

  // Wait for ready (up to 30s — model loading can be slow first time)
  for (let i = 0; i < 60; i++) {
    if (bridgeReady) return true;
    await new Promise((r) => setTimeout(r, 500));
    try {
      const res = await fetch(`${BRIDGE_URL}/health`, { signal: AbortSignal.timeout(1000) });
      if (res.ok) { bridgeReady = true; return true; }
    } catch { /* still starting */ }
  }
  log.warn("memos-md: bridge did not become ready in 30s");
  return false;
}

async function bridgePost(path, body) {
  const res = await fetch(`${BRIDGE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30000),
  });
  return res.json();
}

async function bridgeGet(path) {
  const res = await fetch(`${BRIDGE_URL}${path}`, { signal: AbortSignal.timeout(5000) });
  return res.json();
}

function extractText(msg) {
  if (!msg || !msg.content) return "";
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .filter((b) => b && b.type === "text" && b.text)
      .map((b) => b.text)
      .join("\n");
  }
  return "";
}

export default {
  id: "memos-markdown-openclaw-plugin",
  name: "MemOS Markdown Memory",
  description:
    "Tree-structured markdown memory with local embedding search. " +
    "Stores memories as human-readable .md files. No database required.",
  kind: "lifecycle",

  register(api) {
    const log = api.logger ?? console;

    // ── before_agent_start: RECALL ──────────────────────────────

    api.on("before_agent_start", async (event) => {
      if (!event?.prompt || event.prompt.length < 3) return;

      try {
        if (!(await ensureBridge(log))) return;

        const query = event.prompt.slice(0, 500);
        const data = await bridgePost("/search", { query, top_k: 5 });
        const results = data.results || [];
        if (results.length === 0) return;

        const lines = results.map(
          (r, i) => `${i + 1}. [${r.key}] ${r.memory.slice(0, 300)}`
        );

        const context = [
          "## Relevant memories from previous conversations",
          "",
          "Use these to provide personalized, context-aware responses:",
          "",
          ...lines,
        ].join("\n");

        log.info(`memos-md: recalled ${results.length} memories`);
        return { prependContext: context };
      } catch (err) {
        log.warn(`memos-md: recall failed: ${err}`);
      }
    });

    // ── agent_end: CAPTURE & EXTRACT ────────────────────────────

    api.on("agent_end", async (event) => {
      if (!event?.success || !event?.messages?.length) return;

      try {
        if (!(await ensureBridge(log))) return;

        // Take last user + assistant turn
        const msgs = event.messages;
        let lastUserIdx = -1;
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i]?.role === "user") { lastUserIdx = i; break; }
        }
        if (lastUserIdx < 0) return;

        const recent = msgs.slice(lastUserIdx);
        const formatted = [];
        for (const m of recent) {
          const role = m?.role;
          if (role !== "user" && role !== "assistant") continue;
          const text = extractText(m);
          if (text.trim()) formatted.push({ role, content: text.trim() });
        }
        if (formatted.length === 0) return;

        const data = await bridgePost("/extract", { messages: formatted });
        const count = data.ids?.length ?? 0;
        if (count > 0) {
          log.info(`memos-md: extracted ${count} memories from conversation`);
        }
      } catch (err) {
        log.warn(`memos-md: capture failed: ${err}`);
      }
    });

    // ── Tool: memory_search ─────────────────────────────────────

    api.registerTool(
      {
        name: "memory_search",
        label: "Memory Search",
        description:
          "Search stored conversation memories. Pass a short query (2-5 key words).",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string", description: "Short search query" },
          },
          required: ["query"],
        },
        async execute(_id, params) {
          if (!(await ensureBridge(log))) {
            return { content: [{ type: "text", text: "Memory bridge not available." }] };
          }
          const { query } = params;
          const data = await bridgePost("/search", { query, top_k: 10 });
          const results = data.results || [];
          if (results.length === 0) {
            return { content: [{ type: "text", text: "No relevant memories found." }] };
          }
          const text = results
            .map((r, i) => `${i + 1}. [${r.key}] ${r.memory.slice(0, 300)}`)
            .join("\n\n");
          return {
            content: [{ type: "text", text: `Found ${results.length} memories:\n\n${text}` }],
          };
        },
      },
      { name: "memory_search" },
    );

    // ── Tool: memory_tree ───────────────────────────────────────

    api.registerTool(
      {
        name: "memory_tree",
        label: "Memory Tree",
        description:
          "Show the memory tree structure. Displays all memories organized hierarchically.",
        parameters: { type: "object", properties: {} },
        async execute() {
          if (!(await ensureBridge(log))) {
            return { content: [{ type: "text", text: "Memory bridge not available." }] };
          }
          const data = await bridgeGet("/tree");
          return {
            content: [{ type: "text", text: data.tree || "(empty tree)" }],
          };
        },
      },
      { name: "memory_tree" },
    );

    // ── Service lifecycle ───────────────────────────────────────

    api.registerService({
      id: "memos-markdown-openclaw-plugin",
      async start() {
        const ready = await ensureBridge(log);
        if (ready) {
          log.info("╔══════════════════════════════════════════╗");
          log.info("║  MemOS Markdown Memory — Active           ║");
          log.info("║  Storage: ~/.openclaw/memos-markdown      ║");
          log.info("║  Files: human-readable .md + _index.md    ║");
          log.info("╚══════════════════════════════════════════╝");
        }
      },
      async stop() {
        if (bridgeProcess) {
          log.info("memos-md: stopping bridge");
          bridgeProcess.kill("SIGTERM");
          bridgeProcess = null;
        }
        bridgeReady = false;
        log.info("memos-md: stopped");
      },
    });
  },
};
