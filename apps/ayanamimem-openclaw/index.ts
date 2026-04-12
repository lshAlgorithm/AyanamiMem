/** AyanamiMem OpenClaw Plugin — pure TypeScript, no Python bridge.
 *
 * Registers lifecycle hooks, tools, and a service that manages
 * per-agent session-isolated hierarchical Markdown memories.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import { buildContext } from "./src/config.js";
import { Assembler } from "./src/assembler.js";
import { BM25Index } from "./src/bm25.js";
import { prepareForCapture, extractLastTurn } from "./src/capture.js";
import { chunkMessages } from "./src/chunker.js";
import { EmbeddingCache } from "./src/emb-cache.js";
import { Embedder } from "./src/embedding.js";
import { Compactor } from "./src/compactor.js";
import { FRESH_DIR, SUMMARY_FILENAME, readMd, walkTree } from "./src/fs-store.js";
import { SessionManager } from "./src/session-manager.js";
import { Summarizer } from "./src/summarizer.js";
import { Updater } from "./src/updater.js";
import type { ConvMessage, Logger, PluginConfig, SearchHit } from "./src/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Per-session memory state ──────────────────────────────────────────────────

interface SessionMemory {
  memoryDir: string;
  freshDir: string;
  embCache: EmbeddingCache;
  bm25: BM25Index | null;
  embedder: Embedder;
  summarizer: Summarizer;
  assembler: Assembler;
  compactor: Compactor;
  updater: Updater;
  /** Path → UUID (for tool lookup by chunkId). */
  pathIndex: Map<string, string>;
}

// ── Session ID slug ───────────────────────────────────────────────────────────

function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "default";
}

// ── Format search hits for injection ─────────────────────────────────────────

function formatHits(hits: SearchHit[]): string {
  const lines: string[] = [
    "## Memories from previous conversations",
    "",
    "Use these naturally. To search more memories, call `memory_search`.",
    "",
  ];
  for (let i = 0; i < hits.length; i++) {
    lines.push(`${i + 1}. [${hits[i].key}]`);
    lines.push(`   ${hits[i].content.slice(0, 300).replace(/\n/g, " ").trim()}`);
    lines.push("");
  }
  return lines.join("\n");
}

// ── Plugin registration ───────────────────────────────────────────────────────

export function register(api: {
  on: (event: string, handler: (event: Record<string, unknown>) => unknown) => void;
  registerTool: (def: Record<string, unknown>, opts?: Record<string, unknown>) => void;
  registerService: (svc: { id: string; start: () => Promise<void>; stop: () => Promise<void> }) => void;
  logger?: Logger;
  getConfig?: () => Record<string, unknown>;
}): void {
  const log: Logger = api.logger ?? {
    info:  (...a) => console.log("[AyanamiMem]", ...a),
    warn:  (...a) => console.warn("[AyanamiMem] WARN", ...a),
    error: (...a) => console.error("[AyanamiMem] ERR", ...a),
    debug: (...a) => { if (process.env.AYANAMIMEM_DEBUG) console.debug("[AyanamiMem] DBG", ...a); },
  };

  const rawCfg = api.getConfig?.() ?? {};
  const { config } = buildContext(rawCfg, log);
  const sessManager = new SessionManager(config.memoryRoot);
  const sessions = new Map<string, SessionMemory>();

  // ── Session factory ─────────────────────────────────────────────────────────

  function getOrCreateSession(sessionId: string): SessionMemory {
    const existing = sessions.get(sessionId);
    if (existing) return existing;

    const memoryDir = sessManager.sessionDir(sessionId);
    fs.mkdirSync(path.join(memoryDir, FRESH_DIR), { recursive: true });
    sessManager.registerSession(sessionId);

    const embCache   = new EmbeddingCache();
    const bm25       = config.enableBm25 ? new BM25Index() : null;
    const embedder   = new Embedder({
      model: config.embeddingModel,
      apiBase: config.chatApiBase,
      apiKey:  config.chatApiKey,
    });
    const summarizer = new Summarizer({
      chatApiBase: config.chatApiBase,
      chatApiKey:  config.chatApiKey,
      chatModel:   config.chatModel,
      targetTokens: 512,
    }, log);
    const assembler  = new Assembler(memoryDir, embedder, embCache, bm25, {
      enableBm25: config.enableBm25,
      enableMmr:  config.enableMmr,
      rrfK:       config.rrfK,
      mmrLambda:  config.mmrLambda,
    });
    const compactor  = new Compactor(memoryDir, summarizer, embedder, embCache, bm25, {
      freshTailCount:      config.freshTailCount,
      compactThreshold:    config.compactThreshold,
      compactSimilarity:   config.compactSimilarity,
      mergeThreshold:      config.mergeThreshold,
      condensedMinFanout:  config.condensedMinFanout,
    }, log);
    const updater    = new Updater(memoryDir, summarizer, embedder, embCache, log);

    const sess: SessionMemory = {
      memoryDir, freshDir: path.join(memoryDir, FRESH_DIR),
      embCache, bm25, embedder, summarizer, assembler, compactor, updater,
      pathIndex: new Map(),
    };
    sessions.set(sessionId, sess);
    return sess;
  }

  /** Resolve session ID from lifecycle event. */
  function resolveSessionId(event: Record<string, unknown>): string {
    if (config.sessionId) return slugify(config.sessionId);
    if (typeof event.agentId === "string") return slugify(event.agentId);
    if (typeof event.conversationId === "string") return slugify(event.conversationId);
    return "default";
  }

  /** Load all .md files into the in-memory caches. */
  async function loadSession(sess: SessionMemory): Promise<void> {
    const allPaths = [
      ...walkTree(sess.memoryDir),
      ...(() => {
        const fd = sess.freshDir;
        if (!fs.existsSync(fd)) return [];
        return fs.readdirSync(fd)
          .filter((f) => f.endsWith(".md") && !f.startsWith("_"))
          .map((f) => path.join(fd, f));
      })(),
    ];

    const cachePaths: string[] = [];
    const cacheTexts: string[] = [];
    const bm25Entries: Record<string, string> = {};

    for (const p of allPaths) {
      try {
        const { meta, body } = readMd(p);
        const key = String(meta.key ?? "");
        if (!key) continue;
        const cacheKey = path.basename(p) === SUMMARY_FILENAME ? path.dirname(p) : p;
        cachePaths.push(cacheKey);
        cacheTexts.push(key);
        if (meta.nodeKind === "leaf" || path.basename(p) !== SUMMARY_FILENAME) {
          bm25Entries[p] = key + " " + body;
        }
      } catch { /* skip */ }
    }

    if (cacheTexts.length > 0) {
      try {
        const vecs = await sess.embedder.embed(cacheTexts);
        const entries: Record<string, Float32Array> = {};
        for (let i = 0; i < cachePaths.length; i++) entries[cachePaths[i]] = vecs[i];
        sess.embCache.update(entries);
      } catch (e) {
        log.warn("Batch embed failed during load: %s", String(e));
      }
    }

    if (sess.bm25 && Object.keys(bm25Entries).length > 0) {
      sess.bm25.update(bm25Entries);
    }

    log.info(
      "Session '%s' loaded: %d paths, %d embeddings",
      path.basename(sess.memoryDir), allPaths.length, sess.embCache.size,
    );
  }

  // ── Lifecycle hooks ─────────────────────────────────────────────────────────

  /** before_agent_start: recall relevant memories and inject into context. */
  api.on("before_agent_start", async (event) => {
    const sessionId = resolveSessionId(event);
    const sess = getOrCreateSession(sessionId);

    const rawPrompt = typeof event.prompt === "string" ? event.prompt : "";
    if (!rawPrompt.trim()) return;
    // Strip envelope tags before searching
    const query = rawPrompt
      .replace(/^Sender \(untrusted metadata\):.*\n?/m, "")
      .replace(/<memory_context>[^]*?<\/memory_context>/gs, "")
      .trim()
      .slice(0, 500);

    try {
      const hits = await sess.assembler.assemble(query, config.topK * 500, config.freshTailCount);
      if (!hits.length) return;

      return {
        appendSystemContext:
          "You have persistent memory across conversations (AyanamiMem). " +
          "Previous conversation memories are provided below. " +
          "Use them naturally to give personalized responses. " +
          "Tools available: memory_search, memory_tree.",
        prependContext: formatHits(hits),
      };
    } catch (e) {
      log.warn("Recall failed: %s", String(e));
    }
  });

  /** agent_end: capture new conversation turn and trigger compaction if needed. */
  api.on("agent_end", async (event) => {
    if (!event.success) return;
    const messages = Array.isArray(event.messages)
      ? (event.messages as ConvMessage[])
      : [];
    if (!messages.length) return;

    const sessionId = resolveSessionId(event);
    const sess = getOrCreateSession(sessionId);

    const turn = extractLastTurn(messages);
    const prepared = turn.map(prepareForCapture).filter((m): m is ConvMessage => m !== null);
    if (!prepared.length) return;

    try {
      const chunks = chunkMessages(prepared, sessionId, sess.freshDir, config.leafChunkTokens);
      for (const chunk of chunks) {
        // Update embedding cache
        try {
          const [vec] = await sess.embedder.embed([chunk.key]);
          sess.embCache.update({ [chunk.filepath]: vec });
        } catch { /* skip embed error */ }
        // Update BM25
        if (sess.bm25) sess.bm25.update({ [chunk.filepath]: chunk.key + " " + chunk.body });
      }

      if (chunks.length > 0 && sess.compactor.shouldCompact()) {
        setImmediate(async () => {
          try {
            const result = await sess.compactor.compactIncremental();
            if (result.leavesCompacted > 0 || result.dirsCondensed > 0) {
              log.info(
                "Auto-compact session '%s': %d leaves, %d dirs",
                sessionId, result.leavesCompacted, result.dirsCondensed,
              );
            }
          } catch (e) {
            log.warn("Compaction error: %s", String(e));
          }
        });
      }
    } catch (e) {
      log.warn("Capture failed: %s", String(e));
    }
  });

  // ── Tools ───────────────────────────────────────────────────────────────────

  function currentSession(context: unknown): SessionMemory {
    const agentId = (context as Record<string, unknown>)?.agentId;
    const sessionId = typeof agentId === "string" ? slugify(agentId) : "default";
    return getOrCreateSession(sessionId);
  }

  api.registerTool({
    name: "memory_search",
    label: "Memory Search",
    description:
      "Search stored memories by semantic similarity and keywords. " +
      "Pass a short descriptive query (2-10 words). Returns the most relevant memory nodes.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Short search query" },
        topK: { type: "number", description: "Max results (default 5)" },
      },
      required: ["query"],
    },
    async execute(_id: string, params: Record<string, unknown>, context: unknown) {
      const sess = currentSession(context);
      const query = String(params.query ?? "");
      const k = Number(params.topK ?? config.topK);
      try {
        const hits = await sess.assembler.assemble(query, k * 500, 0);
        if (!hits.length) return { content: [{ type: "text", text: "No relevant memories found." }] };
        const text = hits
          .map((h, i) => `${i + 1}. [${h.key}]\n   ${h.content.slice(0, 400).replace(/\n/g, " ")}`)
          .join("\n\n");
        return { content: [{ type: "text", text: `Found ${hits.length} memories:\n\n${text}` }] };
      } catch (e) {
        return { content: [{ type: "text", text: `Search failed: ${String(e)}` }] };
      }
    },
  }, { name: "memory_search" });

  api.registerTool({
    name: "memory_tree",
    label: "Memory Tree",
    description:
      "Show the hierarchical memory tree for this session. " +
      "Directories are LLM-summarised topic clusters. Leaves are verbatim conversation chunks. " +
      "You can read any node with read_file(path) to get full content and edge links.",
    parameters: { type: "object", properties: {} },
    async execute(_id: string, _params: unknown, context: unknown) {
      const sess = currentSession(context);
      const lines = ["# Memory Tree", ""];
      renderTree(sess.memoryDir, lines, 0, sess.memoryDir);
      const freshDir = sess.freshDir;
      if (fs.existsSync(freshDir)) {
        const fresh = fs.readdirSync(freshDir).filter((f) => f.endsWith(".md") && !f.startsWith("_"));
        if (fresh.length > 0) {
          lines.push(`\n## Fresh (${fresh.length} uncompacted)`);
          for (const f of fresh) lines.push(`  - ${f}`);
        }
      }
      return { content: [{ type: "text", text: lines.join("\n") + "\n" }] };
    },
  }, { name: "memory_tree" });

  api.registerTool({
    name: "memory_compact",
    label: "Memory Compact",
    description:
      "Force compaction of pending leaves into the topic tree. " +
      "Useful after adding many memories to trigger LLM summarisation.",
    parameters: { type: "object", properties: {} },
    async execute(_id: string, _params: unknown, context: unknown) {
      const sess = currentSession(context);
      try {
        const result = await sess.compactor.compactIncremental(true);
        return {
          content: [{
            type: "text",
            text: `Compacted: ${result.leavesCompacted} leaves → dirs, ${result.dirsCondensed} dirs condensed.`,
          }],
        };
      } catch (e) {
        return { content: [{ type: "text", text: `Compact failed: ${String(e)}` }] };
      }
    },
  }, { name: "memory_compact" });

  api.registerTool({
    name: "memory_status",
    label: "Memory Status",
    description: "Show session memory statistics: leaf count, tree depth, cache size.",
    parameters: { type: "object", properties: {} },
    async execute(_id: string, _params: unknown, context: unknown) {
      const sess = currentSession(context);
      const leaves = sess.compactor.sortedLeaves();
      const tail = config.freshTailCount;
      const compactable = Math.max(0, leaves.length - tail);
      const dirDepths: Record<number, number> = {};
      if (fs.existsSync(sess.memoryDir)) {
        for (const entry of fs.readdirSync(sess.memoryDir)) {
          if (entry.startsWith("_")) continue;
          const full = path.join(sess.memoryDir, entry);
          const sum  = path.join(full, SUMMARY_FILENAME);
          if (fs.existsSync(sum)) {
            try {
              const d = Number(readMd(sum).meta.depth ?? 1);
              dirDepths[d] = (dirDepths[d] ?? 0) + 1;
            } catch { /* skip */ }
          }
        }
      }
      const treeLines = Object.entries(dirDepths)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([d, n]) => `  depth ${d}: ${n} dir(s)`);
      const text = [
        `Session dir: ${sess.memoryDir}`,
        `_fresh/: ${leaves.length} leaves (${tail} tail, ${compactable} compactable, threshold=${config.compactThreshold})`,
        `Tree:\n${treeLines.join("\n") || "  (no compacted dirs yet)"}`,
        `Cache: ${sess.embCache.size} embeddings, ${sess.bm25?.size ?? 0} BM25 docs`,
      ].join("\n");
      return { content: [{ type: "text", text }] };
    },
  }, { name: "memory_status" });

  api.registerTool({
    name: "memory_sessions",
    label: "Memory Sessions",
    description: "List all memory sessions in this root directory.",
    parameters: { type: "object", properties: {} },
    execute(_id: string, _params: unknown, context: unknown) {
      const agentId = (context as Record<string, unknown>)?.agentId;
      const currentId = typeof agentId === "string" ? slugify(agentId) : "default";
      const list = sessManager.listSessions();
      if (!list.length) return { content: [{ type: "text", text: "No sessions registered yet." }] };
      const lines = list
        .sort((a, b) => a.sessionId.localeCompare(b.sessionId))
        .map((s) => {
          const you = s.sessionId === currentId ? " ← you" : "";
          const flag = s.isActive ? "active" : "inactive";
          return `  ${s.sessionId}  [${flag}]${you}`;
        });
      const grants = sessManager.getGrantsFor(currentId);
      if (grants.length) {
        lines.push("", "Grants to this session:");
        for (const g of grants) lines.push(`  ← ${g.grantingSession}  scope=${g.scope}`);
      }
      return { content: [{ type: "text", text: `Sessions (root: ${config.memoryRoot}):\n${lines.join("\n")}` }] };
    },
  }, { name: "memory_sessions" });

  api.registerTool({
    name: "memory_write_public",
    label: "Write Public Memory",
    description:
      "Write a memory to the _public/ directory, visible to all sessions and agents.",
    parameters: {
      type: "object",
      properties: {
        content: { type: "string", description: "The memory content to publish." },
        key:     { type: "string", description: "Short descriptive key (5-15 words)." },
        tags:    { type: "array", items: { type: "string" } },
      },
      required: ["content"],
    },
    async execute(_id: string, params: Record<string, unknown>, context: unknown) {
      const sess = currentSession(context);
      const content = String(params.content ?? "");
      const key  = String(params.key  ?? content.slice(0, 60).replace(/\n/g, " "));
      const tags = Array.isArray(params.tags) ? params.tags.map(String) : [];
      const publicFresh = path.join(sessManager.publicDir(), FRESH_DIR);
      fs.mkdirSync(publicFresh, { recursive: true });
      const meta: Record<string, unknown> = {
        key, depth: 0, nodeKind: "leaf", source: "public",
        sessionId: "public", tags, stale: false, descendantCount: 1,
        tokenCount: Math.floor(content.length / 4),
        earliestAt: new Date().toISOString(),
        latestAt: new Date().toISOString(),
      };
      try {
        const { writeLeaf } = await import("./src/fs-store.js");
        writeLeaf(publicFresh, content, meta);
        try {
          const [vec] = await sess.embedder.embed([key]);
          sess.embCache.update({ [publicFresh]: vec });
        } catch { /* ok */ }
        return { content: [{ type: "text", text: `Published to _public/: [${key}]` }] };
      } catch (e) {
        return { content: [{ type: "text", text: `Failed: ${String(e)}` }] };
      }
    },
  }, { name: "memory_write_public" });

  api.registerTool({
    name: "memory_grant",
    label: "Grant Memory Access",
    description: "Share this session's memories with another session.",
    parameters: {
      type: "object",
      properties: {
        targetSessionId: { type: "string", description: "Session ID to grant access to." },
      },
      required: ["targetSessionId"],
    },
    execute(_id: string, params: Record<string, unknown>, context: unknown) {
      const agentId = (context as Record<string, unknown>)?.agentId;
      const currentId = typeof agentId === "string" ? slugify(agentId) : "default";
      const target = String(params.targetSessionId ?? "");
      if (!target) return { content: [{ type: "text", text: "targetSessionId is required." }] };
      sessManager.grantAccess(currentId, target, "read");
      return { content: [{ type: "text", text: `Granted '${currentId}' → '${target}' (recorded in _sessions.json).` }] };
    },
  }, { name: "memory_grant" });

  api.registerTool({
    name: "memory_revoke",
    label: "Revoke Memory Access",
    description: "Revoke a previously granted memory access.",
    parameters: {
      type: "object",
      properties: {
        targetSessionId: { type: "string" },
      },
      required: ["targetSessionId"],
    },
    execute(_id: string, params: Record<string, unknown>, context: unknown) {
      const agentId = (context as Record<string, unknown>)?.agentId;
      const currentId = typeof agentId === "string" ? slugify(agentId) : "default";
      const target = String(params.targetSessionId ?? "");
      sessManager.revokeAccess(currentId, target);
      return { content: [{ type: "text", text: `Revoked access for '${target}'.` }] };
    },
  }, { name: "memory_revoke" });

  // ── Service lifecycle ───────────────────────────────────────────────────────

  api.registerService({
    id: "ayanamimem-openclaw-plugin",

    async start() {
      log.info("AyanamiMem starting (root: %s)", config.memoryRoot);

      // Load all existing sessions
      for (const sess of sessManager.listSessions()) {
        try {
          const sm = getOrCreateSession(sess.sessionId);
          if (fs.existsSync(sm.memoryDir)) {
            await loadSession(sm);
          }
        } catch (e) {
          log.warn("Failed to load session '%s': %s", sess.sessionId, String(e));
        }
      }

      // Install bundled skill guide
      installSkillGuide(log);

      // Patch tools.allow in openclaw.json
      patchOpenclawConfig(log);

      log.info("╔══════════════════════════════════════════╗");
      log.info("║  AyanamiMem Active                        ║");
      log.info("║  Root: %s", config.memoryRoot);
      log.info("║  Sessions: %d loaded                      ║", sessions.size);
      log.info("╚══════════════════════════════════════════╝");
    },

    async stop() {
      log.info("AyanamiMem stopping");
    },
  });

  // ── Private helpers ─────────────────────────────────────────────────────────

  function renderTree(dir: string, lines: string[], indent: number, base: string): void {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir).sort();
    for (const entry of entries) {
      if (entry.startsWith("_")) continue;
      const full = path.join(dir, entry);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) {
        const sum = path.join(full, SUMMARY_FILENAME);
        if (fs.existsSync(sum)) {
          try {
            const key = String(readMd(sum).meta.key ?? entry);
            const relPath = path.relative(base, sum);
            lines.push(`${"  ".repeat(indent)}- **${key}** (\`${relPath}\`)`);
            renderTree(full, lines, indent + 1, base);
          } catch { lines.push(`${"  ".repeat(indent)}- ${entry}/`); }
        }
      } else if (entry.endsWith(".md")) {
        try {
          const key = String(readMd(full).meta.key ?? entry);
          const relPath = path.relative(base, full);
          lines.push(`${"  ".repeat(indent)}- ${key} (\`${relPath}\`)`);
        } catch { lines.push(`${"  ".repeat(indent)}- ${entry}`); }
      }
    }
  }
}

// ── Skill guide installation ──────────────────────────────────────────────────

function installSkillGuide(log: Logger): void {
  try {
    const skillSrc  = path.join(__dirname, "skill", "ayanamimem-guide", "SKILL.md");
    if (!fs.existsSync(skillSrc)) return;
    const home = process.env.HOME ?? process.env.USERPROFILE ?? "~";
    const dest = path.join(home, ".openclaw", "workspace", "skills", "ayanamimem-guide");
    fs.mkdirSync(dest, { recursive: true });
    fs.copyFileSync(skillSrc, path.join(dest, "SKILL.md"));
    log.info("Skill guide installed: %s", dest);
  } catch (e) {
    log.debug("Skill guide install skipped: %s", String(e));
  }
}

// ── openclaw.json patcher ─────────────────────────────────────────────────────

function patchOpenclawConfig(log: Logger): void {
  try {
    const home = process.env.HOME ?? process.env.USERPROFILE ?? "~";
    const cfgPath = path.join(home, ".openclaw", "openclaw.json");
    if (!fs.existsSync(cfgPath)) return;
    const cfg = JSON.parse(fs.readFileSync(cfgPath, "utf8")) as Record<string, unknown>;
    const tools = (cfg.tools ?? {}) as Record<string, unknown>;
    const allow = Array.isArray(tools.allow) ? tools.allow as string[] : [];
    if (!allow.includes("group:plugins")) {
      allow.push("group:plugins");
      tools.allow = allow;
      cfg.tools = tools;
      const tmp = cfgPath + ".tmp";
      fs.writeFileSync(tmp, JSON.stringify(cfg, null, 2), "utf8");
      fs.renameSync(tmp, cfgPath);
      log.info("Patched openclaw.json: added group:plugins to tools.allow");
    }
  } catch (e) {
    log.debug("openclaw.json patch skipped: %s", String(e));
  }
}
