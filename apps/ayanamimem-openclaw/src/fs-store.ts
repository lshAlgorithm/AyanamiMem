/** Filesystem I/O for the hierarchical Markdown memory tree.
 *
 * Port of Python src/memos/memories/textual/hierarchical_markdown/fs.py
 * Uses js-yaml for YAML frontmatter, Node.js fs for all I/O.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import yaml from "js-yaml";
import type { NodeMeta } from "./types.js";

// ── Constants ─────────────────────────────────────────────────────────────────

export const FRESH_DIR = "_fresh";
export const SUMMARY_FILENAME = "_summary.md";
export const ROOT_FILENAME = "_root.md";
export const SHARED_DIR = "_shared";

const EDGE_SECTIONS = new Set(["Children", "Related", "Sequence", "Shared"]);
const FRONTMATTER_RE = /^\s*---\n([\s\S]*?)\n---\n?/;
const SECTION_RE = /^## (.+)$/m;
const LINK_RE = /^-\s+\[([^\]]*)\]\(([^)]+)\)\s*$/;
const SEQ_PREFIX_RE = /^(\d+)/;

// ── Slug / naming ─────────────────────────────────────────────────────────────

/** Convert text to a filesystem-safe ASCII slug. */
export function slugify(text: string, maxLen = 40): string {
  let s = text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")   // strip combining marks
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (s.length > maxLen) s = s.slice(0, maxLen).replace(/-+$/, "");
  return s || "node";
}

/** Next sequence number in a directory by scanning NNN-* entries. */
export function nextSeq(dir: string, forDir = false): number {
  if (!fs.existsSync(dir)) return 1;
  let max = 0;
  for (const entry of fs.readdirSync(dir)) {
    if (entry.startsWith("_")) continue;
    const m = SEQ_PREFIX_RE.exec(entry);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return max + 1;
}

export function leafFilename(seq: number, key: string): string {
  return `${String(seq).padStart(3, "0")}-${slugify(key)}.md`;
}

export function subdirName(seq: number, key: string): string {
  return `${String(seq).padStart(2, "0")}-${slugify(key)}`;
}

// ── Write ─────────────────────────────────────────────────────────────────────

function renderFrontmatter(meta: Record<string, unknown>): string {
  return (yaml.dump(meta, { lineWidth: 120, indent: 2 }) as string).trimEnd();
}

function renderEdges(edges: Record<string, [string, string][]>): string {
  const parts: string[] = [];
  for (const [section, links] of Object.entries(edges)) {
    if (!links.length) continue;
    const lines = [`## ${section}`, ...links.map(([l, p]) => `- [${l}](${p})`)];
    parts.push(lines.join("\n"));
  }
  return parts.join("\n\n");
}

export function writeMd(
  filepath: string,
  body: string,
  meta: Record<string, unknown>,
  edges?: Record<string, [string, string][]>,
): void {
  fs.mkdirSync(path.dirname(filepath), { recursive: true });
  const fm = renderFrontmatter(meta);
  let content = `---\n${fm}\n---\n\n${body}`;
  if (edges) {
    const edgeText = renderEdges(edges);
    if (edgeText) content += `\n\n${edgeText}`;
  }
  if (!content.endsWith("\n")) content += "\n";
  fs.writeFileSync(filepath, content, "utf8");
}

/** Write a leaf .md into dir, returns the filename. */
export function writeLeaf(
  dir: string,
  body: string,
  meta: Record<string, unknown>,
  edges?: Record<string, [string, string][]>,
): string {
  const seq = nextSeq(dir);
  const fname = leafFilename(seq, String(meta.key ?? "leaf"));
  writeMd(path.join(dir, fname), body, meta, edges);
  return fname;
}

/** Write _summary.md into dir with ## Children links. */
export function writeSummary(
  dir: string,
  body: string,
  meta: Record<string, unknown>,
  children: [string, string][],
  extraEdges?: Record<string, [string, string][]>,
): string {
  const edges: Record<string, [string, string][]> = { Children: children, ...(extraEdges ?? {}) };
  const fpath = path.join(dir, SUMMARY_FILENAME);
  writeMd(fpath, body, meta, edges);
  return fpath;
}

// ── Read ──────────────────────────────────────────────────────────────────────

export interface ParsedMd {
  meta: Record<string, unknown>;
  body: string;
  edges: Record<string, [string, string][]>;
}

export function readMd(filepath: string): ParsedMd {
  const text = fs.readFileSync(filepath, "utf8");
  const fmMatch = FRONTMATTER_RE.exec(text);
  let meta: Record<string, unknown> = {};
  let remainder = text;

  if (fmMatch) {
    meta = (yaml.load(fmMatch[1]) as Record<string, unknown>) ?? {};
    remainder = text.slice(fmMatch[0].length);
  }

  // Split body from edge sections
  const sections = remainder.split(SECTION_RE);
  let body = sections[0].trim();
  const edges: Record<string, [string, string][]> = {};

  for (let i = 1; i < sections.length - 1; i += 2) {
    const heading = sections[i].trim();
    const content = sections[i + 1];
    if (EDGE_SECTIONS.has(heading)) {
      const links: [string, string][] = [];
      for (const line of content.trim().split("\n")) {
        const m = LINK_RE.exec(line.trim());
        if (m) links.push([m[1], m[2]]);
      }
      if (links.length) edges[heading] = links;
    } else {
      body += `\n\n## ${heading}\n${content.trim()}`;
    }
  }

  return { meta, body, edges };
}

// ── Directory helpers ─────────────────────────────────────────────────────────

/** Direct children of dir: subdirs with _summary.md + leaf .md files. */
export function listChildren(dir: string): [string, string][] {
  if (!fs.existsSync(dir)) return [];
  type Entry = [number, string, string];
  const children: Entry[] = [];

  for (const entry of fs.readdirSync(dir)) {
    if (entry.startsWith("_")) continue;
    const full = path.join(dir, entry);

    if (fs.statSync(full).isDirectory()) {
      const sum = path.join(full, SUMMARY_FILENAME);
      if (fs.existsSync(sum)) {
        const m = SEQ_PREFIX_RE.exec(entry);
        const seq = m ? parseInt(m[1], 10) : 999;
        const label = entry.includes("-") ? entry.slice(entry.indexOf("-") + 1) : entry;
        children.push([seq, label, `./${entry}/${SUMMARY_FILENAME}`]);
      }
    } else if (entry.endsWith(".md")) {
      const m = SEQ_PREFIX_RE.exec(entry);
      const seq = m ? parseInt(m[1], 10) : 999;
      let label = entry.replace(/\.md$/, "");
      if (label.includes("-")) label = label.slice(label.indexOf("-") + 1);
      children.push([seq, label, `./${entry}`]);
    }
  }

  children.sort((a, b) => a[0] - b[0]);
  return children.map(([, label, rel]) => [label, rel]);
}

/** Walk all .md files under dir (recursive, sorted). */
export function walkTree(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  const results: string[] = [];

  function walk(d: string): void {
    const entries = fs.readdirSync(d).sort();
    for (const entry of entries) {
      if (entry.startsWith("_")) continue;
      const full = path.join(d, entry);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) {
        walk(full);
      } else if (entry.endsWith(".md")) {
        results.push(full);
      }
    }
    // include _summary.md for this directory
    const sum = path.join(d, SUMMARY_FILENAME);
    if (fs.existsSync(sum)) results.push(sum);
  }

  walk(dir);
  return results;
}

/** Count all leaf .md files recursively (not summaries). */
export function countLeaves(dir: string): number {
  if (!fs.existsSync(dir)) return 0;
  let n = 0;
  for (const entry of fs.readdirSync(dir)) {
    if (entry.startsWith("_")) continue;
    const full = path.join(dir, entry);
    if (fs.statSync(full).isDirectory()) n += countLeaves(full);
    else if (entry.endsWith(".md")) n++;
  }
  return n;
}

/** Rough token estimate (1 token ≈ 4 chars). */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.floor(text.length / 4));
}

/** Read .md metadata only (frontmatter) — avoids parsing body/edges. */
export function readMeta(filepath: string): Record<string, unknown> {
  try {
    return readMd(filepath).meta;
  } catch {
    return {};
  }
}
