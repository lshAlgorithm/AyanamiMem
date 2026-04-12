/** Async stale propagation — mark ancestor _summary.md as stale after leaf change.
 *
 * Port of Python src/.../hierarchical_markdown/updater.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { SUMMARY_FILENAME, readMd, writeMd } from "./fs-store.js";
import type { EmbeddingCache } from "./emb-cache.js";
import type { Embedder } from "./embedding.js";
import type { Summarizer } from "./summarizer.js";
import type { Logger } from "./types.js";

export class Updater {
  private memoryDir: string;
  private summarizer: Summarizer;
  private embedder: Embedder;
  private embCache: EmbeddingCache;
  private log: Logger;
  private staleQueue: string[] = [];   // absolute dir paths to re-summarise

  constructor(
    memoryDir: string,
    summarizer: Summarizer,
    embedder: Embedder,
    embCache: EmbeddingCache,
    log: Logger,
  ) {
    this.memoryDir   = memoryDir;
    this.summarizer  = summarizer;
    this.embedder    = embedder;
    this.embCache    = embCache;
    this.log         = log;
  }

  /** Walk upward from changedPath and mark each ancestor _summary.md stale. */
  markStale(changedPath: string): void {
    let current = path.dirname(changedPath);
    const visited = new Set<string>();

    while (current !== this.memoryDir && current !== path.dirname(current)) {
      if (visited.has(current)) break;
      visited.add(current);

      const sumPath = path.join(current, SUMMARY_FILENAME);
      if (!fs.existsSync(sumPath)) {
        current = path.dirname(current);
        continue;
      }

      try {
        const { meta, body, edges } = readMd(sumPath);
        if (!meta.stale) {
          meta.stale = true;
          writeMd(sumPath, body, meta, edges);
          this.log.debug("Marked stale: %s", sumPath);
        }
        if (!this.staleQueue.includes(current)) {
          this.staleQueue.push(current);
        }
      } catch {
        this.log.warn("Failed to mark stale: %s", sumPath);
      }

      current = path.dirname(current);
    }
  }

  /** Process all stale nodes — lowest depth first. */
  async processStale(): Promise<void> {
    if (this.staleQueue.length === 0) return;

    // Sort by depth (number of path components) — deepest first
    const queue = [...this.staleQueue].sort((a, b) => {
      const da = a.split(path.sep).length;
      const db = b.split(path.sep).length;
      return db - da;  // deepest (highest count) first
    });
    this.staleQueue = [];

    for (const dirPath of queue) {
      await this._resummaise(dirPath);
    }
  }

  private async _resummaise(dirPath: string): Promise<void> {
    const sumPath = path.join(dirPath, SUMMARY_FILENAME);
    if (!fs.existsSync(sumPath)) return;

    try {
      const { meta: oldMeta } = readMd(sumPath);
      const depth = Number(oldMeta.depth ?? 1);

      // Collect children
      const childPaths: string[] = [];
      for (const entry of fs.readdirSync(dirPath)) {
        if (entry.startsWith("_")) continue;
        const full = path.join(dirPath, entry);
        if (fs.statSync(full).isDirectory()) {
          const childSum = path.join(full, SUMMARY_FILENAME);
          if (fs.existsSync(childSum)) childPaths.push(childSum);
        } else if (entry.endsWith(".md")) {
          childPaths.push(full);
        }
      }

      if (childPaths.length === 0) return;

      const { summary, key, tags } = await this.summarizer.summarize(childPaths, depth);
      const { meta: curMeta, edges } = readMd(sumPath);
      curMeta.key   = key;
      curMeta.stale = false;
      curMeta.tags  = tags;
      writeMd(sumPath, summary, curMeta, edges);

      // Update embedding cache
      try {
        const [vec] = await this.embedder.embed([key]);
        this.embCache.update({ [dirPath]: vec });
      } catch {
        this.log.warn("Failed to re-embed after re-summarise: %s", dirPath);
      }

      this.log.info("Re-summarised: %s", path.relative(this.memoryDir, dirPath));
    } catch (e) {
      this.log.warn("Re-summarise failed for %s: %s", dirPath, String(e));
    }
  }
}
