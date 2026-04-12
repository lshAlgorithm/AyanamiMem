/** BM25Plus keyword index — pure JavaScript, no dependencies.
 *
 * Port of Python src/.../hierarchical_markdown/bm25_index.py
 * Uses BM25Plus (not BM25Okapi) to avoid zero-IDF issues on small corpora:
 *   idf(t) = log((N+1) / df(t))   — always positive
 *
 * Lazy rebuild: mutations set dirty=true; search triggers _rebuild() as needed.
 */

// ── Tokenizer ─────────────────────────────────────────────────────────────────

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

// ── BM25Plus class ────────────────────────────────────────────────────────────

const K1 = 1.5;
const B  = 0.75;
const DELTA = 1.0;   // BM25+ lower-bound term contribution

export class BM25Index {
  /** path → tokenized words */
  private texts = new Map<string, string[]>();

  // Rebuilt index state
  private dirty = false;
  private paths: string[] = [];
  private corpus: string[][] = [];
  private df = new Map<string, number>();    // term → document frequency
  private avgdl = 0;
  private N = 0;

  // ── Write ───────────────────────────────────────────────────────────────────

  update(entries: Record<string, string>): void {
    for (const [p, text] of Object.entries(entries)) {
      this.texts.set(p, tokenize(text));
    }
    this.dirty = true;
  }

  remove(paths: string[]): void {
    let changed = false;
    for (const p of paths) {
      if (this.texts.delete(p)) changed = true;
    }
    if (changed) this.dirty = true;
  }

  renamePrefix(oldPrefix: string, newPrefix: string): void {
    const old = oldPrefix.replace(/\/$/, "");
    const neu = newPrefix.replace(/\/$/, "");
    for (const k of [...this.texts.keys()]) {
      if (k === old || k.startsWith(old + "/")) {
        const newKey = neu + k.slice(old.length);
        this.texts.set(newKey, this.texts.get(k)!);
        this.texts.delete(k);
      }
    }
    this.dirty = true;
  }

  clear(): void {
    this.texts.clear();
    this.dirty = false;
    this.paths = [];
    this.corpus = [];
    this.df.clear();
    this.avgdl = 0;
    this.N = 0;
  }

  get size(): number { return this.texts.size; }

  // ── Search ──────────────────────────────────────────────────────────────────

  /** Return [(path, score)] sorted descending by BM25Plus score. */
  search(query: string, topK: number): [string, number][] {
    if (this.texts.size === 0) return [];
    if (this.dirty) this._rebuild();

    const qTokens = tokenize(query);
    if (qTokens.length === 0) return [];

    const scores = new Array<number>(this.N).fill(0);
    for (const q of qTokens) {
      const df = this.df.get(q) ?? 0;
      if (df === 0) continue;
      const idf = Math.log((this.N + 1) / df);   // BM25+ IDF — always positive
      for (let i = 0; i < this.N; i++) {
        const dl = this.corpus[i].length;
        const tf = this.corpus[i].filter((t) => t === q).length;
        if (tf === 0) continue;
        const norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / this.avgdl)) + DELTA;
        scores[i] += idf * norm;
      }
    }

    const pairs = scores
      .map((s, i): [string, number] => [this.paths[i], s])
      .filter(([, s]) => s > 0);
    pairs.sort((a, b) => b[1] - a[1]);
    return pairs.slice(0, topK);
  }

  // ── Rebuild ─────────────────────────────────────────────────────────────────

  private _rebuild(): void {
    this.paths  = [...this.texts.keys()];
    this.corpus = this.paths.map((p) => this.texts.get(p)!);
    this.N      = this.paths.length;

    // Document frequency
    this.df.clear();
    for (const doc of this.corpus) {
      for (const term of new Set(doc)) {
        this.df.set(term, (this.df.get(term) ?? 0) + 1);
      }
    }

    // Average document length
    const total = this.corpus.reduce((s, d) => s + d.length, 0);
    this.avgdl  = this.N > 0 ? total / this.N : 1;

    this.dirty = false;
  }
}
