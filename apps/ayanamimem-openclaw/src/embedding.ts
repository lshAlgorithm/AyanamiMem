/** Local Transformers.js embedding — Xenova/all-MiniLM-L6-v2, 384-dim.
 *
 * Same model as the Python sentence-transformers version. Embeddings are
 * bit-for-bit equivalent, so existing .md directories work without rebuild.
 * ~23 MB download on first use (quantized ONNX model).
 */

// @ts-expect-error — Transformers.js has loose types in some versions
import { pipeline } from "@huggingface/transformers";

// ── Singleton pipeline ────────────────────────────────────────────────────────

let pipelinePromise: Promise<unknown> | null = null;

function getPipeline(modelName: string): Promise<unknown> {
  if (!pipelinePromise) {
    pipelinePromise = pipeline("feature-extraction", modelName, {
      dtype: "q8",
      device: "cpu",
    }) as Promise<unknown>;
  }
  return pipelinePromise;
}

// ── Local embedding ───────────────────────────────────────────────────────────

async function embedLocal(texts: string[], modelName: string): Promise<Float32Array[]> {
  const extractor = await getPipeline(modelName) as (
    text: string,
    opts: Record<string, unknown>,
  ) => Promise<{ data: Float32Array }>;

  const results: Float32Array[] = [];
  for (const text of texts) {
    const out = await extractor(text, { pooling: "mean", normalize: true });
    results.push(new Float32Array(out.data).slice(0, 384));
  }
  return results;
}

// ── Remote embedding (OpenAI-compatible) ──────────────────────────────────────

async function embedRemote(
  texts: string[],
  apiBase: string,
  apiKey: string,
  model: string,
): Promise<Float32Array[]> {
  const res = await fetch(`${apiBase}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ input: texts, model }),
  });
  if (!res.ok) throw new Error(`Embedding API ${res.status}: ${await res.text()}`);
  const json = (await res.json()) as { data: { embedding: number[] }[] };
  return json.data.map((d) => new Float32Array(d.embedding));
}

// ── Embedder class ────────────────────────────────────────────────────────────

interface EmbedderCfg {
  model: string;
  apiBase?: string;
  apiKey?: string;
  remoteModel?: string;
}

export class Embedder {
  private cfg: EmbedderCfg;
  private batchSize = 32;

  constructor(cfg: EmbedderCfg) { this.cfg = cfg; }

  async embed(texts: string[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const batchResults = await this.embedBatch(batch);
      results.push(...batchResults);
    }
    return results;
  }

  async embedQuery(text: string): Promise<Float32Array> {
    return (await this.embed([text]))[0];
  }

  private async embedBatch(texts: string[]): Promise<Float32Array[]> {
    try {
      return await embedLocal(texts, this.cfg.model);
    } catch (e) {
      // Only try remote if explicitly configured
      if (this.cfg.apiBase && this.cfg.apiKey && this.cfg.remoteModel) {
        try {
          return await embedRemote(texts, this.cfg.apiBase, this.cfg.apiKey, this.cfg.remoteModel);
        } catch {
          // fall through to re-throw local error
        }
      }
      throw e;
    }
  }
}
