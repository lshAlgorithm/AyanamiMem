/** Plugin configuration builder — mirrors local plugin's buildContext(). */

import * as os from "node:os";
import * as path from "node:path";
import { DEFAULTS, type Logger, type PluginConfig, type PluginContext } from "./types.js";

// ── Env resolution ────────────────────────────────────────────────────────────

/** Recursively replace ${ENV_VAR} placeholders in string values. */
function deepResolveEnv<T>(val: T): T {
  if (typeof val === "string") {
    return val.replace(/\$\{([^}]+)\}/g, (_, k) => process.env[k] ?? "") as unknown as T;
  }
  if (Array.isArray(val)) return val.map(deepResolveEnv) as unknown as T;
  if (val && typeof val === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(val as Record<string, unknown>)) {
      out[k] = deepResolveEnv(v);
    }
    return out as unknown as T;
  }
  return val;
}

// ── Tilde expansion ───────────────────────────────────────────────────────────

function expandHome(p: string): string {
  if (p.startsWith("~/") || p === "~") {
    return path.join(os.homedir(), p.slice(2));
  }
  return p;
}

// ── Fallback logger ───────────────────────────────────────────────────────────

const consoleLog: Logger = {
  info:  (m, ...a) => console.log(`[AyanamiMem] ${m}`, ...a),
  warn:  (m, ...a) => console.warn(`[AyanamiMem] WARN ${m}`, ...a),
  error: (m, ...a) => console.error(`[AyanamiMem] ERROR ${m}`, ...a),
  debug: (m, ...a) => {
    if (process.env.AYANAMIEMEM_DEBUG) console.debug(`[AyanamiMem] DEBUG ${m}`, ...a);
  },
};

// ── Builder ───────────────────────────────────────────────────────────────────

export function buildContext(pluginCfg: Record<string, unknown>, apiLog?: Logger): PluginContext {
  const resolved = deepResolveEnv(pluginCfg) as Record<string, unknown>;
  const log: Logger = apiLog ?? consoleLog;

  const chatApiBase = (resolved.chatApiBase as string | undefined)
    ?? process.env.OPENAI_BASE_URL
    ?? process.env.OPENAI_API_BASE
    ?? "";
  const chatApiKey = (resolved.chatApiKey as string | undefined)
    ?? process.env.OPENAI_API_KEY
    ?? "";
  const chatModel = (resolved.chatModel as string | undefined)
    ?? process.env.MOS_CHAT_MODEL
    ?? DEFAULTS.chatModel;
  const embeddingModel = (resolved.embeddingModel as string | undefined)
    ?? process.env.EMBEDDER_MODEL
    ?? DEFAULTS.embeddingModel;

  const memoryRootRaw = (resolved.memoryRoot as string | undefined)
    ?? process.env.MEMORY_ROOT
    ?? DEFAULTS.memoryRoot;
  const memoryRoot = path.resolve(expandHome(memoryRootRaw));

  const config: PluginConfig = {
    memoryRoot,
    sessionId: (resolved.sessionId as string | undefined) ?? process.env.SESSION_ID,
    topK: num(resolved.topK, DEFAULTS.topK),
    freshTailCount: num(resolved.freshTailCount, DEFAULTS.freshTailCount),
    compactThreshold: num(resolved.compactThreshold, DEFAULTS.compactThreshold),
    compactSimilarity: num(resolved.compactSimilarity, DEFAULTS.compactSimilarity),
    mergeThreshold: num(resolved.mergeThreshold, DEFAULTS.mergeThreshold),
    condensedMinFanout: num(resolved.condensedMinFanout, DEFAULTS.condensedMinFanout),
    leafChunkTokens: num(resolved.leafChunkTokens, DEFAULTS.leafChunkTokens),
    chatModel,
    chatApiBase,
    chatApiKey,
    embeddingModel,
    enableBm25: bool(resolved.enableBm25, DEFAULTS.enableBm25),
    enableMmr: bool(resolved.enableMmr, DEFAULTS.enableMmr),
    rrfK: num(resolved.rrfK, DEFAULTS.rrfK),
    mmrLambda: num(resolved.mmrLambda, DEFAULTS.mmrLambda),
  };

  return { config, log };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function num(val: unknown, def: number): number {
  const n = Number(val);
  return Number.isFinite(n) ? n : def;
}

function bool(val: unknown, def: boolean): boolean {
  if (val === undefined || val === null) return def;
  if (typeof val === "boolean") return val;
  if (val === "true") return true;
  if (val === "false") return false;
  return def;
}
