/** Compaction engine — group _fresh/ leaves into topic dirs with LLM summaries.
 *
 * Port of Python src/.../hierarchical_markdown/compactor.py
 * setImmediate-based scheduling: non-blocking for the current event loop turn.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import {
  FRESH_DIR, SUMMARY_FILENAME,
  leafFilename, listChildren, nextSeq, readMd, subdirName, writeMd, writeSummary,
} from "./fs-store.js";
import { EmbeddingCache, cosine } from "./emb-cache.js";
import type { BM25Index } from "./bm25.js";
import type { Embedder } from "./embedding.js";
import type { Summarizer } from "./summarizer.js";
import type { CompactResult, Logger } from "./types.js";

// ── Clustering ────────────────────────────────────────────────────────────────

function agglomerativeCluster(
  names: string[],
  embeddings: Record<string, Float32Array>,
  threshold: number,
): string[][] {
  if (names.length === 0) return [];

  // Greedy single-linkage: each item joins the first cluster whose centroid is close enough
  const clusters: string[][] = [];
  const centroids: Float32Array[] = [];

  for (const name of names) {
    const vec = embeddings[name];
    if (!vec) {
      clusters.push([name]);
      centroids.push(new Float32Array(384));
      continue;
    }

    let bestIdx = -1;
    let bestSim = -Infinity;
    for (let i = 0; i < centroids.length; i++) {
      const sim = cosine(vec, centroids[i]);
      if (sim > bestSim) { bestSim = sim; bestIdx = i; }
    }

    if (bestIdx >= 0 && bestSim >= threshold) {
      clusters[bestIdx].push(name);
      // Update centroid: mean of all vectors in cluster
      const members = clusters[bestIdx].map((n) => embeddings[n]).filter(Boolean);
      const dim = 384;
      const sum = new Float32Array(dim);
      for (const m of members) for (let j = 0; j < dim; j++) sum[j] += m[j];
      const norm = Math.sqrt(sum.reduce((s, x) => s + x * x, 0)) || 1;
      centroids[bestIdx] = sum.map((x) => x / norm) as Float32Array;
    } else {
      clusters.push([name]);
      centroids.push(vec);
    }
  }

  return clusters;
}

// ── Compactor ─────────────────────────────────────────────────────────────────

interface CompactorCfg {
  freshTailCount: number;
  compactThreshold: number;
  compactSimilarity: number;
  mergeThreshold: number;
  condensedMinFanout: number;
  condensedTargetTokens?: number;
}

export class Compactor {
  readonly memoryDir: string;
  private summarizer: Summarizer;
  private embedder: Embedder;
  private embCache: EmbeddingCache;
  private bm25: BM25Index | null;
  private cfg: CompactorCfg;
  private log: Logger;
  private _compacting = false;

  constructor(
    memoryDir: string,
    summarizer: Summarizer,
    embedder: Embedder,
    embCache: EmbeddingCache,
    bm25: BM25Index | null,
    cfg: CompactorCfg,
    log: Logger,
  ) {
    this.memoryDir  = memoryDir;
    this.summarizer = summarizer;
    this.embedder   = embedder;
    this.embCache   = embCache;
    this.bm25       = bm25;
    this.cfg        = cfg;
    this.log        = log;
  }

  get freshDir(): string {
    return path.join(this.memoryDir, FRESH_DIR);
  }

  sortedLeaves(): string[] {
    const freshDir = this.freshDir;
    if (!fs.existsSync(freshDir)) return [];
    return fs.readdirSync(freshDir)
      .filter((f) => f.endsWith(".md") && !f.startsWith("_"))
      .sort();
  }

  shouldCompact(): boolean {
    const leaves = this.sortedLeaves();
    return leaves.length - this.cfg.freshTailCount >= this.cfg.compactThreshold;
  }

  async compactIncremental(force = false): Promise<CompactResult> {
    if (this._compacting) {
      this.log.debug("Compaction already running, skipping");
      return { leavesCompacted: 0, dirsCondensed: 0 };
    }
    this._compacting = true;
    let leavesCompacted = 0;
    let dirsCondensed   = 0;
    try {
      if (force || this.shouldCompact()) {
        leavesCompacted = await this.leafPass();
      }
      let depth = 1;
      while (true) {
        const n = await this.condensationPass(depth);
        if (n === 0) break;
        dirsCondensed += n;
        depth++;
        if (this.cfg.condensedMinFanout < 2) break;  // safety
      }
    } finally {
      this._compacting = false;
    }
    return { leavesCompacted, dirsCondensed };
  }

  async leafPass(): Promise<number> {
    const leaves = this.sortedLeaves();
    const toCompact = leaves.slice(0, leaves.length - this.cfg.freshTailCount);
    if (toCompact.length === 0) return 0;

    // Read embeddings from cache
    const freshDir = this.freshDir;
    const embs: Record<string, Float32Array> = {};
    for (const fname of toCompact) {
      const vec = this.embCache.get(path.join(freshDir, fname));
      if (vec) embs[fname] = vec;
    }

    const clusters = agglomerativeCluster(toCompact, embs, this.cfg.compactSimilarity);
    let compacted = 0;

    // Pre-load existing depth-1 dirs for merge decision
    const existingDirs: [string, Float32Array][] = [];
    for (const [name, vec] of Object.entries(this.embCache.childrenOf(this.memoryDir))) {
      const fullPath = path.join(this.memoryDir, name);
      if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isDirectory()) continue;
      const sumPath = path.join(fullPath, SUMMARY_FILENAME);
      if (!fs.existsSync(sumPath)) continue;
      try {
        const meta = readMd(sumPath).meta;
        if (Number(meta.depth ?? 0) === 1) existingDirs.push([fullPath, vec]);
      } catch { /* skip */ }
    }

    for (const cluster of clusters) {
      if (cluster.length === 0) continue;
      const childPaths = cluster.map((f) => path.join(freshDir, f)).filter((p) => fs.existsSync(p));
      if (childPaths.length === 0) continue;

      const { summary, key, tags } = await this.summarizer.summarize(childPaths, 1);
      const now = new Date().toISOString();

      // Merge or create?
      let targetDir: string | null = null;
      if (existingDirs.length > 0) {
        try {
          const [clusterVec] = await this.embedder.embed([key]);
          let bestSim = 0, bestDir = "";
          for (const [d, dv] of existingDirs) {
            const sim = cosine(clusterVec, dv);
            if (sim > bestSim) { bestSim = sim; bestDir = d; }
          }
          if (bestSim >= this.cfg.mergeThreshold) {
            targetDir = bestDir;
            this.log.info("Merging cluster '%s' → %s (sim=%.3f)", key, path.basename(bestDir), bestSim);
          }
        } catch { /* fall through to new dir */ }
      }

      if (!targetDir) {
        const seq = nextSeq(this.memoryDir, true);
        const dirname = subdirName(seq, key);
        targetDir = path.join(this.memoryDir, dirname);
        fs.mkdirSync(targetDir, { recursive: true });
      }

      // Move leaves, rename to cluster key, update frontmatter
      const childrenLinks: [string, string][] = [];
      for (let i = 0; i < cluster.length; i++) {
        const src = path.join(freshDir, cluster[i]);
        if (!fs.existsSync(src)) continue;

        const newFname = leafFilename(i + 1, key);
        const dst = path.join(targetDir, newFname);

        try {
          const { meta, body, edges } = readMd(src);
          meta.key  = key;
          meta.tags = tags;
          writeMd(src, body, meta, edges);
        } catch { /* keep old meta */ }

        fs.renameSync(src, dst);
        this.embCache.renamePrefix(src, dst);
        if (this.bm25) {
          this.bm25.remove([src]);
          try {
            const { body } = readMd(dst);
            this.bm25.update({ [dst]: key + " " + body });
          } catch { /* skip */ }
        }
        childrenLinks.push([newFname.replace(/\.md$/, ""), `./${newFname}`]);
        compacted++;
      }

      // Write or update _summary.md
      const sumPath = path.join(targetDir, SUMMARY_FILENAME);
      if (fs.existsSync(sumPath)) {
        // Merge into existing: add children, mark stale
        try {
          const { meta: oldMeta, body: oldBody, edges: oldEdges } = readMd(sumPath);
          const oldChildren = oldEdges.Children ?? [];
          oldMeta.stale = true;
          oldMeta.descendantCount = Number(oldMeta.descendantCount ?? 0) + childrenLinks.length;
          oldMeta.latestAt = now;
          writeSummary(targetDir, oldBody, oldMeta, [...oldChildren, ...childrenLinks]);
        } catch { /* skip */ }
      } else {
        const allChildren = listChildren(targetDir);
        const meta: Record<string, unknown> = {
          key, depth: 1, nodeKind: "condensed",
          earliestAt: now, latestAt: now,
          tokenCount: Math.floor(summary.length / 4),
          descendantCount: allChildren.length,
          stale: false, tags,
        };
        writeSummary(targetDir, summary, meta, allChildren);

        // Register in cache
        try {
          const [vec] = await this.embedder.embed([key]);
          this.embCache.update({ [targetDir]: vec });
          existingDirs.push([targetDir, vec]);
        } catch { /* skip */ }
      }
    }

    this.log.info("Leaf pass: %d leaves compacted", compacted);
    return compacted;
  }

  async condensationPass(depth: number): Promise<number> {
    // Find all depth-N dirs in memoryDir
    const targetDirs: string[] = [];
    for (const entry of fs.readdirSync(this.memoryDir)) {
      if (entry.startsWith("_")) continue;
      const full = path.join(this.memoryDir, entry);
      if (!fs.existsSync(full) || !fs.statSync(full).isDirectory()) continue;
      const sumPath = path.join(full, SUMMARY_FILENAME);
      if (!fs.existsSync(sumPath)) continue;
      try {
        const meta = readMd(sumPath).meta;
        if (Number(meta.depth ?? 0) === depth) targetDirs.push(full);
      } catch { /* skip */ }
    }

    if (targetDirs.length < this.cfg.condensedMinFanout) return 0;

    // Get embeddings
    const dirEmbs: Record<string, Float32Array> = {};
    for (const d of targetDirs) {
      const name = path.basename(d);
      const vec = this.embCache.get(d);
      if (vec) dirEmbs[name] = vec;
    }

    const dirNames = targetDirs.map((d) => path.basename(d));
    const clusters = agglomerativeCluster(dirNames, dirEmbs, this.cfg.compactSimilarity);
    let condensed = 0;
    const now = new Date().toISOString();

    for (const cluster of clusters) {
      if (cluster.length < 2) continue;
      const clusterSumPaths = cluster
        .map((n) => path.join(this.memoryDir, n, SUMMARY_FILENAME))
        .filter((p) => fs.existsSync(p));
      if (clusterSumPaths.length === 0) continue;

      const { summary, key, tags } = await this.summarizer.summarize(clusterSumPaths, depth + 1);

      const seq = nextSeq(this.memoryDir, true);
      const newDirname = subdirName(seq, key);
      const newDir = path.join(this.memoryDir, newDirname);
      fs.mkdirSync(newDir, { recursive: true });

      const childrenLinks: [string, string][] = [];
      for (const dname of cluster) {
        const src = path.join(this.memoryDir, dname);
        const dst = path.join(newDir, dname);
        if (!fs.existsSync(src)) continue;
        fs.renameSync(src, dst);
        this.embCache.renamePrefix(src, dst);
        if (this.bm25) this.bm25.renamePrefix(src, dst);
        childrenLinks.push([dname, `./${dname}/${SUMMARY_FILENAME}`]);
        condensed++;
      }

      const totalDesc = cluster.reduce((s, dname) => {
        try {
          const sp = path.join(newDir, dname, SUMMARY_FILENAME);
          return s + Number(readMd(sp).meta.descendantCount ?? 0);
        } catch { return s; }
      }, 0);

      const meta: Record<string, unknown> = {
        key, depth: depth + 1, nodeKind: "condensed",
        earliestAt: now, latestAt: now,
        tokenCount: Math.floor(summary.length / 4),
        descendantCount: totalDesc, stale: false, tags,
      };
      writeSummary(newDir, summary, meta, childrenLinks);

      try {
        const [vec] = await this.embedder.embed([key]);
        this.embCache.update({ [newDir]: vec });
      } catch { /* skip */ }
    }

    this.log.info("Condensation depth %d: %d dirs condensed", depth, condensed);
    return condensed;
  }
}
