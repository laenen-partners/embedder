# Embedder

Go library for generating text embeddings via Firebase Genkit.

## Quick start

```sh
mise install
task test
```

## Usage

```go
emb := embedder.New(ctx) // reads GOOGLE_API_KEY from env
vectors, err := emb.Embed(ctx, []string{"hello world"})
```

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
| `embedder.go` | `New()`, `Embed()` — core library API |
| `plugins/openaicompat/` | Genkit plugin for OpenAI-compatible embeddings |
