/** LLM-based summarisation via OpenAI-compatible API.
 *
 * Port of Python src/.../hierarchical_markdown/summarizer.py
 * All prompts are TS string constants matching hierarchical_markdown_prompts.py.
 */

import * as fs from "node:fs";
import { readMd } from "./fs-store.js";
import type { Logger } from "./types.js";

// ── Prompt templates ──────────────────────────────────────────────────────────

const LEAF_KEY_PROMPT = `Given this conversation chunk, produce a short descriptive key (5-15 words) and 1-5 tags.

<conversation>
{{conversation}}
</conversation>

Return ONLY valid JSON: {"key": "...", "tags": ["..."]}`;

const LEAF_SUMMARY_PROMPT = `You are a memory summariser. Below is a raw conversation chunk.

<conversation>
{{conversation}}
</conversation>

{{previous_context}}

Produce a concise narrative summary (max {{target_tokens}} tokens) capturing key facts, decisions, preferences, and action items. Also produce a short key (5-15 words) and 1-5 tags.

Return ONLY valid JSON: {"key": "...", "summary": "...", "tags": ["..."]}`;

const CONDENSED_D1_PROMPT = `You are a memory summariser. Below are summaries of related conversation segments.

<summaries>
{{children_summaries}}
</summaries>

{{previous_context}}

Produce a thematic summary (max {{target_tokens}} tokens) preserving key facts, names, numbers, and decisions. Also produce a key (5-15 words) and 1-5 tags.

Return ONLY valid JSON: {"key": "...", "summary": "...", "tags": ["..."]}`;

const CONDENSED_D2_PLUS_PROMPT = `You are a memory summariser working at high abstraction. Below are summaries of broad topic groups.

<summaries>
{{children_summaries}}
</summaries>

{{previous_context}}

Produce an abstract overview (max {{target_tokens}} tokens) identifying major themes and patterns. Also produce a key (5-15 words) and 1-5 tags.

Return ONLY valid JSON: {"key": "...", "summary": "...", "tags": ["..."]}`;

const AGGRESSIVE_SUMMARY_PROMPT = `You are a memory summariser under strict space constraints. Compress the following.

<content>
{{content}}
</content>

Keep ONLY concrete facts (names, numbers, dates), firm decisions, preferences, and action items. Max {{target_tokens}} tokens.

Return ONLY valid JSON: {"key": "...", "summary": "...", "tags": ["..."]}`;

// ── JSON extraction ───────────────────────────────────────────────────────────

function parseJson(text: string): Record<string, unknown> {
  const start = text.indexOf("{");
  const end   = text.lastIndexOf("}");
  if (start < 0 || end < 0) return {};
  try {
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    return {};
  }
}

// ── Summarizer class ──────────────────────────────────────────────────────────

interface SummarizerCfg {
  chatApiBase: string;
  chatApiKey: string;
  chatModel: string;
  targetTokens?: number;
}

export class Summarizer {
  private cfg: SummarizerCfg;
  private log: Logger;
  private targetTokens: number;

  constructor(cfg: SummarizerCfg, log: Logger) {
    this.cfg = cfg;
    this.log = log;
    this.targetTokens = cfg.targetTokens ?? 512;
  }

  /** Lightweight: extract just a key and tags from raw text (no full summary). */
  async extractKey(text: string): Promise<{ key: string; tags: string[] }> {
    const prompt = LEAF_KEY_PROMPT.replace("{{conversation}}", text.slice(0, 2000));
    try {
      const resp = await this._call(prompt);
      const data = parseJson(resp);
      return {
        key:  String(data.key  ?? text.slice(0, 60).replace(/\n/g, " ")),
        tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
      };
    } catch {
      return { key: text.slice(0, 60).replace(/\n/g, " "), tags: [] };
    }
  }

  /** Summarise children .md files into a condensed node. */
  async summarize(
    childPaths: string[],
    depth: number,
    prevSummary?: string,
  ): Promise<{ summary: string; key: string; tags: string[] }> {
    const childrenText = this._readChildren(childPaths);

    const prevCtx = prevSummary
      ? `The previous summary was:\n<previous>\n${prevSummary}\n</previous>\nUpdate it based on the new children.`
      : "";

    const template = prevSummary
      ? CONDENSED_D1_PROMPT
      : depth === 1 ? LEAF_SUMMARY_PROMPT
      : depth === 2 ? CONDENSED_D1_PROMPT
      : CONDENSED_D2_PLUS_PROMPT;

    // Level 1: normal summarisation
    let result = await this._runPrompt(template, childrenText, prevCtx, 0.2);
    if (result && this._isAcceptable(result.summary)) return result;

    // Level 2: aggressive
    this.log.warn("Escalating to aggressive summary (depth=%d)", depth);
    result = await this._runPrompt(AGGRESSIVE_SUMMARY_PROMPT, childrenText, prevCtx, 0.1);
    if (result && this._isAcceptable(result.summary)) return result;

    // Level 3: truncation fallback
    this.log.warn("Using truncation fallback (depth=%d)", depth);
    const truncated = childrenText.slice(0, this.targetTokens * 4);
    return { summary: truncated, key: "summary", tags: [] };
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private async _runPrompt(
    template: string,
    childrenText: string,
    prevCtx: string,
    _temperature: number,
  ): Promise<{ summary: string; key: string; tags: string[] } | null> {
    const prompt = template
      .replace("{{conversation}}", childrenText)
      .replace("{{children_summaries}}", childrenText)
      .replace("{{content}}", childrenText)
      .replace("{{previous_summary}}", prevCtx)
      .replace("{{previous_context}}", prevCtx)
      .replace(/\{\{target_tokens\}\}/g, String(this.targetTokens));

    try {
      const resp = await this._call(prompt);
      const data = parseJson(resp);
      if (data.summary && data.key) {
        return {
          summary: String(data.summary),
          key: String(data.key),
          tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
        };
      }
    } catch (e) {
      this.log.warn("LLM summarisation failed: %s", String(e));
    }
    return null;
  }

  private _readChildren(paths: string[]): string {
    const parts: string[] = [];
    for (const p of paths) {
      if (!fs.existsSync(p)) continue;
      try {
        const { meta, body } = readMd(p);
        const header = meta.key ? `[${meta.key}]` : "";
        parts.push(`${header}\n${body}`);
      } catch {
        this.log.warn("Could not read child: %s", p);
      }
    }
    return parts.join("\n\n---\n\n");
  }

  private _isAcceptable(summary: string): boolean {
    return summary.length / 4 <= this.targetTokens * 1.5;
  }

  private async _call(prompt: string): Promise<string> {
    const apiBase = this.cfg.chatApiBase.replace(/\/$/, "");
    const res = await fetch(`${apiBase}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.cfg.chatApiKey}`,
      },
      body: JSON.stringify({
        model: this.cfg.chatModel,
        messages: [{ role: "user", content: prompt }],
        temperature: 0.2,
      }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!res.ok) throw new Error(`LLM API ${res.status}: ${await res.text()}`);
    const json = (await res.json()) as { choices: { message: { content: string } }[] };
    return json.choices[0]?.message?.content ?? "";
  }
}
