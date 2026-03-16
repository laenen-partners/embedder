# 0001 — Embedder library value-add features

**Status:** proposed
**Date:** 2026-03-16

## Context

The embedder library currently wraps `genkit.Embed()` with a thin options layer and provider auto-detection. This makes adoption easy but doesn't justify a library over calling Genkit directly. We need features that provide real value beyond the Genkit API.

## Problem

Production embedding workloads have recurring needs that Genkit's low-level API doesn't address:

1. **Provider limits** — Google AI caps batch size at 2048 texts per request; other providers have similar limits. Callers must chunk manually.
2. **Transient failures** — API rate limits (429), network blips, and server errors (5xx) are common. Without retry logic every caller writes their own.
3. **Redundant API calls** — Embedding the same text repeatedly wastes money and latency. Caching is universally useful but tedious to wire up.
4. **Inconsistent output** — Some providers don't L2-normalise embeddings; callers doing cosine similarity need normalised vectors.
5. **Throughput** — Large document sets benefit from parallel batch requests, but naive parallelism can overwhelm providers.
6. **Storage cost** — Many use cases don't need full-dimension vectors; truncated embeddings reduce storage and retrieval cost.

## Proposed features

### 1. Auto-batching

Split input slices into provider-appropriate chunks, call the API per chunk, and reassemble results transparently.

```go
// User passes 5000 texts — library auto-chunks into 3 API calls.
vectors, err := emb.Embed(ctx, fiveThousandTexts)
```

**Design decisions:**
- Default batch size of 100 (safe for all known providers).
- `WithBatchSize(n)` option to override.
- Batching is a concern of `Embed()`, not the constructor — no goroutines on init.
- Preserve input order: chunk → call → reassemble by index.
- Errors from any chunk fail the entire call (no partial results). This keeps the API simple. Callers who need partial results can chunk themselves.

### 2. Retry with exponential backoff

Wrap API calls with configurable retry logic for transient errors.

```go
emb := embedder.New(ctx, embedder.WithRetry(3, time.Second))
```

**Design decisions:**
- Retry on: context not cancelled AND error is transient (HTTP 429, 500, 502, 503, 504, connection reset).
- Exponential backoff with jitter: `baseDelay * 2^attempt * (0.5 + rand(0.5))`.
- Default: 3 retries, 1s base delay. `WithRetry(maxRetries, baseDelay)` to customise.
- `WithRetry(0, 0)` disables retry (useful in tests).
- Retry wraps each batch independently — one failed batch retries without re-calling successful batches.
- Respect `Retry-After` header if present (common with 429s).

**Open question:** How to classify "transient" errors from Genkit? Genkit wraps HTTP errors — we may need to unwrap and inspect. Start with string matching on common patterns; refine if Genkit exposes structured error types.

### 3. Caching

Optional in-memory cache to avoid re-embedding identical texts.

```go
emb := embedder.New(ctx, embedder.WithCache(10000)) // cache up to 10k entries
```

**Design decisions:**
- Cache key: `sha256(model + text)` — fast, collision-resistant, model-scoped.
- LRU eviction with configurable max entries.
- Cache is per-Embedder instance (not global) — no shared state between embedders.
- Cache-through: `Embed()` checks cache, calls API for misses, stores results, returns all.
- Thread-safe via `sync.RWMutex`.
- `WithCache(0)` or no option = no caching (default).
- No TTL for v1 — embeddings for a given model+text are deterministic. Add TTL later if needed.
- Cache operates on individual texts within a batch, not entire batches. This maximises hit rate when batches partially overlap.

**Interaction with batching:** Cache lookup happens first. Only cache misses are batched and sent to the API. Results are merged back in order.

### 4. L2 normalisation

Optionally normalise output vectors to unit length for cosine similarity.

```go
emb := embedder.New(ctx, embedder.WithNormalize())
```

**Design decisions:**
- Compute L2 norm; divide each dimension. Skip if norm is zero (degenerate vector).
- Applied after caching (cache stores raw vectors; normalisation is cheap and callers may want both).
- Actually, cache _normalised_ vectors when normalisation is enabled, since the caller always wants the same form. This avoids redundant computation on cache hits.
- No option for other normalisation types (L1, max) — L2 covers 99% of use cases.

### 5. Concurrent batch requests

For large inputs, dispatch batch chunks concurrently with a bounded worker pool.

```go
emb := embedder.New(ctx, embedder.WithConcurrency(4))
```

**Design decisions:**
- Default concurrency: 1 (sequential). `WithConcurrency(n)` enables parallel dispatch.
- Use `errgroup.Group` with `SetLimit(n)` — well-tested, context-aware, propagates first error.
- Concurrency applies to batch chunks only (from feature 1). No concurrency without batching.
- Cancel remaining chunks on first error (via errgroup context).
- Result reassembly uses pre-allocated slice indexed by chunk number — no locking needed.

### 6. Dimensionality truncation

Truncate output vectors to reduce storage cost.

```go
emb := embedder.New(ctx, embedder.WithDimensions(256))
```

**Design decisions:**
- Simply slice `vector[:n]`. Valid for Matryoshka-style models (text-embedding-005, nomic-embed-text-v1.5) which are trained for prefix truncation.
- Applied after normalisation (if enabled) — re-normalise after truncation since truncated vectors are no longer unit length.
- `WithDimensions(0)` or no option = full dimensions (default).
- No validation that the model supports truncation — that's the caller's responsibility.

## Implementation order

Each feature is independent and testable in isolation. Order by value and dependency:

1. **Auto-batching** — foundation that concurrency builds on
2. **Retry** — wraps each batch call; high standalone value
3. **Caching** — interacts with batching (filter misses before batching)
4. **Normalisation** — simple post-processing, no dependencies
5. **Truncation** — simple post-processing, depends on normalisation order
6. **Concurrency** — builds on batching infrastructure

## Embed() pipeline

After all features, the `Embed()` flow becomes:

```
input texts
    │
    ▼
[cache lookup] ──→ cache hits (set aside)
    │
    ▼
  misses
    │
    ▼
[chunk into batches]
    │
    ▼
[dispatch batches] ──→ [retry per batch]
    │  (concurrent)
    ▼
[reassemble results]
    │
    ▼
[merge with cache hits]
    │
    ▼
[normalise]  (if enabled)
    │
    ▼
[truncate]   (if enabled)
    │
    ▼
[store in cache]  (normalised+truncated form)
    │
    ▼
  output
```

**Correction on cache placement:** cache should store the _final_ form (post-normalise, post-truncate). This means cache lookup returns ready-to-use vectors and we don't re-process cache hits. The pipeline becomes:

```
input texts
    │
    ▼
[cache lookup] ──→ hits (final form, set aside)
    │
    ▼
  misses
    │
    ▼
[chunk into batches]
    │
    ▼
[dispatch batches with retry]
    │  (concurrent if configured)
    ▼
[reassemble API results]
    │
    ▼
[normalise misses]  (if enabled)
    │
    ▼
[truncate misses]   (if enabled)
    │
    ▼
[store misses in cache]
    │
    ▼
[merge hits + processed misses by original index]
    │
    ▼
  output
```

## API surface after implementation

```go
emb := embedder.New(ctx,
    embedder.WithModel("googleai/text-embedding-005"),
    embedder.WithBatchSize(100),
    embedder.WithRetry(3, time.Second),
    embedder.WithCache(10000),
    embedder.WithNormalize(),
    embedder.WithDimensions(256),
    embedder.WithConcurrency(4),
)

vectors, err := emb.Embed(ctx, texts)
```

All options are optional. Zero-value behaviour matches current API (single batch, no retry, no cache, raw vectors, full dimensions, sequential).

## Testing strategy

- **Unit tests** with fake Genkit embedder (existing pattern) for all features.
- **Batching:** verify correct chunking, order preservation, error propagation.
- **Retry:** inject failing fake that succeeds on Nth attempt; verify backoff timing with mock clock.
- **Cache:** verify hits/misses, LRU eviction, thread safety under concurrent access.
- **Normalisation:** verify output vectors have unit L2 norm.
- **Truncation:** verify output dimension count; verify re-normalisation after truncation.
- **Concurrency:** verify results match sequential execution; stress test with race detector.
- **Integration:** pipeline test combining all features end-to-end with fake embedder.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Genkit error classification may be fragile | Start with string matching; wrap in helper function for easy update |
| Cache memory pressure with large vectors | LRU with explicit max entries; document memory implications |
| Truncation on non-Matryoshka models produces bad embeddings | Document requirement; no runtime validation (caller's responsibility) |
| Retry masking persistent errors | Cap retries; log each retry attempt |
