/** Strip OpenClaw envelope injections from messages before storage.
 *
 * Port of Python logic + local plugin's src/capture/index.ts
 * Prevents our own injected memory context from being re-stored as fake content.
 */

import type { ConvMessage } from "./types.js";

// Tags our plugin injects — strip before re-storing
const INJECTION_PATTERNS = [
  // AyanamiMem recall block
  /\n*## User's conversation history[^]*?(?=\n(?:User:|Assistant:|$))/g,
  /<memory_context>[^]*?<\/memory_context>/gs,
  /<memos_system_instruction>[^]*?<\/memos_system_instruction>/gs,
  /\[STORED_MEMORY\][^]*?\[\/STORED_MEMORY\]/gs,
  // OpenClaw envelope tags
  /^Sender \(untrusted metadata\):.*\n?/gm,
  /^Conversation info:.*\n?/gm,
  /^\[message_id:[^\]]+\]\s*/gm,
  /^\[\[reply_to_current\]\]\s*/gm,
];

export function stripMemoryInjection(content: string): string {
  let s = content;
  for (const pattern of INJECTION_PATTERNS) {
    s = s.replace(pattern, "");
  }
  return s.trim();
}

/** Full message-level stripping pipeline for capture. */
export function prepareForCapture(msg: ConvMessage): ConvMessage | null {
  // Skip system messages
  if (msg.role === "system") return null;

  // Get text content
  let content = "";
  if (typeof msg.content === "string") {
    content = msg.content;
  } else if (Array.isArray(msg.content)) {
    content = (msg.content as Array<{ type: string; text?: string } | string>)
      .map((b) => (typeof b === "string" ? b : b.type === "text" ? (b.text ?? "") : ""))
      .join("\n");
  }

  // Skip empty
  if (!content.trim()) return null;

  // Skip tool results from our own tools (prevent self-ingestion loops)
  const ownToolNames = [
    "memory_search", "memory_tree", "memory_get", "memory_compact",
    "memory_status", "memory_sessions", "memory_write_public",
    "memory_grant", "memory_revoke",
  ];
  if (msg.role === "tool") {
    if (ownToolNames.some((t) => content.includes(t))) return null;
  }

  // Strip injected context
  content = stripMemoryInjection(content);
  if (msg.role === "assistant") {
    // Strip chain-of-thought reasoning blocks
    content = content.replace(/<think>[^]*?<\/think>/gs, "").trim();
  }

  if (!content.trim()) return null;
  return { role: msg.role, content };
}

/** Extract the last user+assistant exchange from a message list. */
export function extractLastTurn(messages: ConvMessage[]): ConvMessage[] {
  let lastUserIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") {
      lastUserIdx = i;
      break;
    }
  }
  if (lastUserIdx < 0) return [];
  return messages.slice(lastUserIdx).filter((m) => m.role === "user" || m.role === "assistant");
}
