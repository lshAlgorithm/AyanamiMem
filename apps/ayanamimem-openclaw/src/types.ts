/** Core type definitions for AyanamiMem. */

// ── Node types ────────────────────────────────────────────────────────────────

export type NodeKind = "leaf" | "condensed";

export interface NodeMeta {
  key: string;
  depth: number;
  nodeKind: NodeKind;
  sessionId: string;
  source: string;
  tags: string[];
  stale: boolean;
  descendantCount: number;
  tokenCount: number;
  earliestAt: string;
  latestAt: string;
}

/** A single memory node — leaf or condensed summary. */
export interface AyanamiNode {
  /** Absolute filesystem path to the .md file. */
  path: string;
  /** Topic label (5-15 words, LLM-generated for condensed; heuristic for fresh leaves). */
  key: string;
  /** Full body text (verbatim conversation for leaf; LLM summary for condensed). */
  body: string;
  meta: NodeMeta;
  edges: Record<string, [string, string][]>;   // sectionName → [(label, relPath)]
}

// ── Search types ──────────────────────────────────────────────────────────────

export interface SearchHit {
  path: string;
  content: string;
  key: string;
  score: number;
  depth: number;
  meta: NodeMeta;
}

export interface RankedCandidate {
  path: string;
  vectorScore: number | null;
  bm25Score: number | null;
  rrfScore: number;
}

// ── Conversation types ────────────────────────────────────────────────────────

export type MessageRole = "user" | "assistant" | "system" | "tool";

export interface ConvMessage {
  role: MessageRole;
  content: string | unknown;
}

// ── Session types ─────────────────────────────────────────────────────────────

export interface Session {
  sessionId: string;
  userId: string;
  createdAt: string;
  isActive: boolean;
}

export interface Grant {
  grantingSession: string;
  granteeSession: string;
  scope: "read";
  subtreePath?: string;
  grantedAt: string;
}

export interface SessionsFile {
  sessions: Record<string, Session>;
  grants: Grant[];
}

// ── Config types ──────────────────────────────────────────────────────────────

export interface PluginConfig {
  memoryRoot: string;
  sessionId?: string;
  topK: number;
  freshTailCount: number;
  compactThreshold: number;
  compactSimilarity: number;
  mergeThreshold: number;
  condensedMinFanout: number;
  leafChunkTokens: number;
  chatModel: string;
  chatApiBase: string;
  chatApiKey: string;
  embeddingModel: string;
  enableBm25: boolean;
  enableMmr: boolean;
  rrfK: number;
  mmrLambda: number;
}

export interface PluginContext {
  config: PluginConfig;
  log: Logger;
}

// ── Logger ────────────────────────────────────────────────────────────────────

export interface Logger {
  info(msg: string, ...args: unknown[]): void;
  warn(msg: string, ...args: unknown[]): void;
  error(msg: string, ...args: unknown[]): void;
  debug(msg: string, ...args: unknown[]): void;
}

// ── Compactor result ──────────────────────────────────────────────────────────

export interface CompactResult {
  leavesCompacted: number;
  dirsCondensed: number;
}

// ── Defaults ──────────────────────────────────────────────────────────────────

export const DEFAULTS: Readonly<PluginConfig> = {
  memoryRoot: "~/.ayanamimem/sessions",
  topK: 5,
  freshTailCount: 2,
  compactThreshold: 3,
  compactSimilarity: 0.3,
  mergeThreshold: 0.70,
  condensedMinFanout: 3,
  leafChunkTokens: 1024,
  chatModel: "deepseek/deepseek-chat",
  chatApiBase: "",
  chatApiKey: "",
  embeddingModel: "Xenova/all-MiniLM-L6-v2",
  enableBm25: true,
  enableMmr: false,
  rrfK: 60,
  mmrLambda: 0.5,
};
