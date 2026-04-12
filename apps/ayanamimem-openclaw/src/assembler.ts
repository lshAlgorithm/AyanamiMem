/** Top-down tree walk + BM25 hybrid search with RRF, path constraint, and MMR.
 *
 * Port of Python src/.../hierarchical_markdown/assembler.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { FRESH_DIR, SUMMARY_FILENAME, estimateTokens, readMd } from "./fs-store.js";
import { EmbeddingCache, cosine } from "./emb-cache.js";
import type { BM25Index } from "./bm25.js";
import type { Embedder } from "./embedding.js";
import type { SearchHit, RankedCandidate } from "./types.js";

const SEQ_PREFIX_RE = /^(\d+)/;

// ── RRF ───────────────────────────────────────────────────────────────────────

/** Reciprocal Rank Fusion of multiple ranked lists.
 *  score(d) = Σ 1/(k + rank_i(d))   k=60 by default
 */
export function rrfMerge(lists: [string, number][][], k = 60): [string, number][] {
  const scores = new Map<string, number>();
  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const [p] = list[rank];
      scores.set(p, (scores.get(p) ?? 0) + 1 / (k + rank + 1));
    }
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1]);
}

// ── Path constraint ───────────────────────────────────────────────────────────

/** Keep at most one node per root-to-leaf path.
 *  A selected _summary.md covers its entire parent directory.
 *  A selected leaf excludes its parent summary.
 */
export function applyPathConstraint(
  ranked: [string, number][],
  excludedFresh: Set<string>,
): string[] {
  const selected: string[] = [];
  const selectedPaths = new Set<string>();
  const coverageDirs  = new Set<string>();  // dirs whose summary is already selected

  for (const [p] of ranked) {
    if (excludedFresh.has(p)) continue;

    const norm = p.replace(/\/$/, "");
    // Is this path inside a dir already covered by a selected summary?
    let covered = false;
    for (const d of coverageDirs) {
      if (norm.startsWith(d + path.sep) || norm.startsWith(d + "/")) {
        covered = true;
        break;
      }
    }
    if (covered) continue;

    if (path.basename(norm) === SUMMARY_FILENAME) {
      const parentDir = path.dirname(norm);
      // Does any already-selected path live inside this dir?
      let childSelected = false;
      for (const s of selectedPaths) {
        if (s.startsWith(parentDir + path.sep) || s.startsWith(parentDir + "/")) {
          childSelected = true;
          break;
        }
      }
      if (childSelected) continue;
      selected.push(p);
      selectedPaths.add(p);
      coverageDirs.add(parentDir);
    } else {
      selected.push(p);
      selectedPaths.add(p);
    }
  }
  return selected;
}

// ── MMR ───────────────────────────────────────────────────────────────────────

export function mmrRerank(
  candidates: SearchHit[],
  embCache: EmbeddingCache,
  lambda = 0.5,
): SearchHit[] {
  if (candidates.length === 0) return candidates;
  const selected: SearchHit[] = [];
  const remaining = [...candidates];

  while (remaining.length > 0) {
    let best: SearchHit;
    if (selected.length === 0) {
      best = remaining.reduce((a, b) => (a.score > b.score ? a : b));
    } else {
      best = remaining.reduce((bestC, c) => {
        const vc = embCache.get(c.path);
        const score = vc
          ? lambda * c.score -
            (1 - lambda) *
              Math.max(
                ...selected.map((s) => {
                  const vs = embCache.get(s.path);
                  return vs ? cosine(vc, vs) : 0;
                }),
              )
          : lambda * c.score;
        const bVc = embCache.get(bestC.path);
        const bestScore = bVc
          ? lambda * bestC.score -
            (1 - lambda) *
              Math.max(
                ...selected.map((s) => {
                  const vs = embCache.get(s.path);
                  return vs ? cosine(bVc, vs) : 0;
                }),
              )
          : lambda * bestC.score;
        return score > bestScore ? c : bestC;
      });
    }
    selected.push(best);
    remaining.splice(remaining.indexOf(best), 1);
  }
  return selected;
}

// ── Assembler ─────────────────────────────────────────────────────────────────

interface AssemblerCfg {
  enableBm25: boolean;
  enableMmr: boolean;
  rrfK: number;
  mmrLambda: number;
}

export class Assembler {
  private memoryDir: string;
  private embedder: Embedder;
  private embCache: EmbeddingCache;
  private bm25: BM25Index | null;
  private cfg: AssemblerCfg;

  constructor(
    memoryDir: string,
    embedder: Embedder,
    embCache: EmbeddingCache,
    bm25: BM25Index | null,
    cfg: AssemblerCfg,
  ) {
    this.memoryDir = memoryDir;
    this.embedder  = embedder;
    this.embCache  = embCache;
    this.bm25      = bm25;
    this.cfg       = cfg;
  }

  async assemble(query: string, budgetTokens: number, freshTailCount: number): Promise<SearchHit[]> {
    const results: SearchHit[] = [];
    let remaining = budgetTokens;
    const freshPaths = new Set<string>();

    // Phase 1: Fresh tail — always included
    const freshDir = path.join(this.memoryDir, FRESH_DIR);
    const tailNodes = this._freshTail(freshTailCount, freshDir);
    for (const node of tailNodes) {
      const tokens = estimateTokens(node.content);
      if (tokens > remaining) break;
      remaining -= tokens;
      results.push(node);
      freshPaths.add(node.path);
    }

    if (remaining <= 0) return results;

    // Phase 2a: Vector top-down walk
    let queryVec: Float32Array;
    try {
      queryVec = await this.embedder.embedQuery(query);
    } catch {
      return results;  // no embedding available — return fresh tail only
    }

    const excluded = new Set<string>(freshPaths);
    const vectorResults = this._topDownWalk(this.memoryDir, queryVec, remaining * 2, excluded, 0);
    const vectorRanked: [string, number][] = vectorResults
      .sort((a, b) => b.score - a.score)
      .map((r) => [r.path, r.score]);

    // Phase 2b: BM25 leaf scan
    let bm25Ranked: [string, number][] = [];
    if (this.cfg.enableBm25 && this.bm25) {
      bm25Ranked = this.bm25
        .search(query, vectorRanked.length * 3 + 20)
        .filter(([p]) => !freshPaths.has(p));
    }

    // Phase 3: RRF merge
    const merged = bm25Ranked.length > 0
      ? rrfMerge([vectorRanked, bm25Ranked], this.cfg.rrfK)
      : vectorRanked;

    // Phase 4: Path constraint
    const constrained = applyPathConstraint(merged, freshPaths);

    // Build candidates
    const vectorMap = new Map<string, SearchHit>(vectorResults.map((r) => [r.path, r]));
    const rrfMap    = new Map<string, number>(merged);
    const candidates: SearchHit[] = [];

    for (const p of constrained) {
      if (vectorMap.has(p)) {
        const node = { ...vectorMap.get(p)!, score: rrfMap.get(p) ?? vectorMap.get(p)!.score };
        candidates.push(node);
      } else if (fs.existsSync(p)) {
        try {
          const { meta, body } = readMd(p);
          candidates.push({
            path: p,
            content: body,
            key: String(meta.key ?? ""),
            score: rrfMap.get(p) ?? 0,
            depth: Number(meta.depth ?? 0),
            meta: meta as never,
          });
        } catch { /* skip unreadable */ }
      }
    }

    candidates.sort((a, b) => b.score - a.score);

    // Phase 5: Optional MMR
    const final = this.cfg.enableMmr
      ? mmrRerank(candidates, this.embCache, this.cfg.mmrLambda)
      : candidates;

    // Fill budget
    for (const node of final) {
      if (remaining <= 0) break;
      const tokens = estimateTokens(node.content);
      if (tokens > remaining) continue;
      remaining -= tokens;
      results.push(node);
    }

    return results;
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private _freshTail(count: number, freshDir: string): SearchHit[] {
    if (!fs.existsSync(freshDir) || count === 0) return [];
    type E = [number, string];
    const leaves: E[] = [];
    for (const entry of fs.readdirSync(freshDir)) {
      if (entry.startsWith("_") || !entry.endsWith(".md")) continue;
      const m = SEQ_PREFIX_RE.exec(entry);
      leaves.push([m ? parseInt(m[1], 10) : 999, entry]);
    }
    leaves.sort((a, b) => a[0] - b[0]);
    const tail = leaves.slice(-count);
    const hits: SearchHit[] = [];
    for (const [, fname] of tail) {
      const fpath = path.join(freshDir, fname);
      try {
        const { meta, body } = readMd(fpath);
        hits.push({
          path: fpath,
          content: body,
          key: String(meta.key ?? fname),
          score: 1.0,
          depth: 0,
          meta: meta as never,
        });
      } catch { /* skip */ }
    }
    return hits;
  }

  private _topDownWalk(
    dir: string,
    queryVec: Float32Array,
    budget: number,
    excluded: Set<string>,
    depth: number,
  ): SearchHit[] {
    if (path.basename(dir) === FRESH_DIR) return [];

    const children = this.embCache.childrenOf(dir);
    if (Object.keys(children).length === 0) return [];

    const scored: [number, string][] = Object.entries(children)
      .map(([name, vec]) => [cosine(queryVec, vec), name] as [number, string])
      .sort((a, b) => b[0] - a[0]);

    const results: SearchHit[] = [];
    let remaining = budget;

    for (const [sim, name] of scored) {
      if (remaining <= 0) break;
      const childPath = path.join(dir, name);

      if (fs.existsSync(childPath) && fs.statSync(childPath).isDirectory()) {
        const sumPath = path.join(childPath, SUMMARY_FILENAME);
        if (excluded.has(sumPath) || !fs.existsSync(sumPath)) continue;
        try {
          const { meta, body } = readMd(sumPath);
          const sumTokens = estimateTokens(body);
          const subChildren = this.embCache.childrenOf(childPath);

          if (Object.keys(subChildren).length > 0 && remaining > sumTokens * 2) {
            const sub = this._topDownWalk(childPath, queryVec, remaining, excluded, depth + 1);
            if (sub.length > 0) {
              results.push(...sub);
              remaining -= sub.reduce((s, n) => s + estimateTokens(n.content), 0);
              continue;
            }
          }

          if (sumTokens <= remaining) {
            results.push({
              path: sumPath,
              content: body,
              key: String(meta.key ?? ""),
              score: sim,
              depth: depth + 1,
              meta: meta as never,
            });
            remaining -= sumTokens;
            excluded.add(sumPath);
          }
        } catch { /* skip */ }

      } else if (name.endsWith(".md")) {
        if (excluded.has(childPath) || !fs.existsSync(childPath)) continue;
        try {
          const { meta, body } = readMd(childPath);
          const tokens = estimateTokens(body);
          if (tokens <= remaining) {
            results.push({
              path: childPath,
              content: body,
              key: String(meta.key ?? ""),
              score: sim,
              depth,
              meta: meta as never,
            });
            remaining -= tokens;
            excluded.add(childPath);
          }
        } catch { /* skip */ }
      }
    }
    return results;
  }
}
