/** Conversation message chunker — writes leaf .md files to _fresh/.
 *
 * Port of Python src/.../hierarchical_markdown/chunker.py
 * No LLM call — uses a fast heuristic key from the first user sentence.
 * The compactor renames leaves to the LLM cluster key after grouping.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { FRESH_DIR, leafFilename, nextSeq, writeMd, estimateTokens } from "./fs-store.js";
import type { ConvMessage } from "./types.js";

// ── Format ────────────────────────────────────────────────────────────────────

function formatMessage(msg: ConvMessage): string {
  const role = msg.role;
  let content = msg.content;
  if (typeof content !== "string") {
    // Handle multi-part content blocks
    if (Array.isArray(content)) {
      content = (content as Array<{ type: string; text?: string } | string>)
        .map((b) => (typeof b === "string" ? b : b.type === "text" ? (b.text ?? "") : ""))
        .join("\n");
    } else {
      content = String(content ?? "");
    }
  }
  const label =
    role === "user" ? "User"
    : role === "assistant" ? "Assistant"
    : role === "system" ? "System"
    : role.charAt(0).toUpperCase() + role.slice(1);
  return `${label}: ${content}`;
}

// ── Heuristic key ─────────────────────────────────────────────────────────────

export function extractKey(chunkText: string): string {
  for (const line of chunkText.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let content = trimmed;
    if (content.includes(": ")) content = content.slice(content.indexOf(": ") + 2);
    if (content.length > 80) content = content.slice(0, 77) + "...";
    return content;
  }
  return "conversation chunk";
}

// ── Chunker ───────────────────────────────────────────────────────────────────

interface ChunkResult {
  filename: string;
  filepath: string;
  key: string;
  body: string;
}

export function chunkMessages(
  messages: ConvMessage[],
  sessionId: string,
  freshDir: string,
  leafChunkTokens = 1024,
): ChunkResult[] {
  if (messages.length === 0) return [];
  fs.mkdirSync(freshDir, { recursive: true });

  const chunks = splitMessages(messages, leafChunkTokens);
  const results: ChunkResult[] = [];
  const now = new Date().toISOString();

  for (const chunkText of chunks) {
    const key = extractKey(chunkText);
    const seq = nextSeq(freshDir);
    const fname = leafFilename(seq, key);
    const fpath = path.join(freshDir, fname);

    const meta: Record<string, unknown> = {
      key,
      depth: 0,
      nodeKind: "leaf",
      sessionId,
      source: "conversation",
      tags: [],
      stale: false,
      descendantCount: 1,
      tokenCount: estimateTokens(chunkText),
      earliestAt: now,
      latestAt: now,
    };

    writeMd(fpath, chunkText, meta);
    results.push({ filename: fname, filepath: fpath, key, body: chunkText });
  }

  return results;
}

// ── Split ─────────────────────────────────────────────────────────────────────

function splitMessages(messages: ConvMessage[], maxTokens: number): string[] {
  const chunks: string[] = [];
  let currentLines: string[] = [];
  let currentTokens = 0;

  for (const msg of messages) {
    const line = formatMessage(msg);
    const lineTokens = estimateTokens(line);

    if (currentTokens + lineTokens > maxTokens && currentLines.length > 0) {
      chunks.push(currentLines.join("\n\n"));
      currentLines = [];
      currentTokens = 0;
    }

    currentLines.push(line);
    currentTokens += lineTokens;
  }

  if (currentLines.length > 0) {
    chunks.push(currentLines.join("\n\n"));
  }

  return chunks;
}
