# Embedder

Pure Go module for generating text embeddings via Firebase Genkit.

## Quick start

```sh
mise install
cp .env.sample .env
# Set GOOGLE_API_KEY in .env
task test
```

## Usage

```go
emb := embedder.New(ctx)
vectors, err := emb.Embed(ctx, []string{"hello world", "another text"})
// vectors is [][]float32

// With options
emb := embedder.New(ctx, embedder.WithModel("lmstudio/nomic-embed-text-v1.5"))
```

## Architecture

- **Library:** `embedder.go` exposes `New(ctx, opts...) *Embedder` and `Embed(ctx, texts) ([][]float32, error)`
- **No database** — purely stateless; callers store embeddings as needed
- **Genkit** — uses `genkit.Embed()` with configurable embedding model

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDER_MODEL` | `googleai/text-embedding-005` | Genkit embedder model reference |
| `GOOGLE_API_KEY` | | Google AI API key |
| `OPENAI_COMPAT_URL` | | OpenAI-compatible server URL (enables plugin) |
| `OPENAI_COMPAT_PROVIDER` | `openaicompat` | Provider name prefix for model references |
| `OPENAI_COMPAT_MODEL` | | Model name on the compatible server |
| `OPENAI_COMPAT_API_KEY` | | API key for the compatible server |

## Key files

| File | Purpose |
|---|---|
| `embedder.go` | Core embedding logic, `New()` constructor |
| `config.go` | `NewConfig()`, options pattern |
| `plugins/openaicompat/openaicompat.go` | OpenAI-compatible embedding plugin |
