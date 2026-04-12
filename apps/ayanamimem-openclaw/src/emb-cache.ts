/** In-memory embedding cache: absolute path → Float32Array.
 *
 * Port of Python src/.../hierarchical_markdown/embeddings.py
 * Single-threaded Node.js: no file locks, no atomics needed.
 */

export class EmbeddingCache {
  private cache = new Map<string, Float32Array>();

  update(entries: Record<string, number[] | Float32Array>): void {
    for (const [k, v] of Object.entries(entries)) {
      this.cache.set(k, v instanceof Float32Array ? v : new Float32Array(v));
    }
  }

  remove(paths: string[]): void {
    for (const p of paths) this.cache.delete(p);
  }

  /** Rename a path prefix atomically (used when compactor moves files/dirs). */
  renamePrefix(oldPrefix: string, newPrefix: string): void {
    const old = oldPrefix.replace(/\/$/, "");
    const neu = newPrefix.replace(/\/$/, "");
    for (const k of [...this.cache.keys()]) {
      if (k === old) {
        this.cache.set(neu, this.cache.get(k)!);
        this.cache.delete(k);
      } else if (k.startsWith(old + "/")) {
        this.cache.set(neu + k.slice(old.length), this.cache.get(k)!);
        this.cache.delete(k);
      }
    }
  }

  get(p: string): Float32Array | undefined { return this.cache.get(p); }

  /** Return {childName: vector} for all DIRECT children of dirPath. */
  childrenOf(dirPath: string): Record<string, Float32Array> {
    const prefix = dirPath.replace(/\/$/, "") + "/";
    const result: Record<string, Float32Array> = {};
    for (const [k, v] of this.cache) {
      if (k.startsWith(prefix)) {
        const rest = k.slice(prefix.length);
        if (rest && !rest.includes("/")) result[rest] = v;
      }
    }
    return result;
  }

  clear(): void { this.cache.clear(); }
  get size(): number { return this.cache.size; }
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-9 ? 0 : dot / denom;
}
