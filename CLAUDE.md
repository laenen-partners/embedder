# Embedder Service

Stateless Connect-RPC service that generates text embeddings via Firebase Genkit.

## Quick start

```sh
mise install
cp .env.sample .env
# Set GOOGLE_API_KEY in .env
task test
task run
```

## Architecture

- **Input:** one or more text strings via `Embed` RPC
- **Output:** corresponding vector embeddings ([]float32 per text)
- **No database** — purely stateless; callers store embeddings as needed
- **Genkit** — uses `genkit.Embed()` with configurable embedding model

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ADDR` | `:3000` | Server listen address |
| `API_KEYS` | | Comma-separated API keys for RPC auth |
| `RATE_LIMIT` | `10` | Requests per second per IP (0 = disabled) |
| `RATE_BURST` | `20` | Burst allowance per IP |
| `CORS_ORIGINS` | | Comma-separated allowed CORS origins |
| `EMBEDDER_MODEL` | `googleai/text-embedding-005` | Genkit embedder model reference |
| `GOOGLE_API_KEY` | | Google AI API key |

## Key files

| File | Purpose |
|---|---|
| `cmd/embedder/main.go` | Entry point, Genkit init, server startup |
| `embedder.go` | Core embedding logic (Genkit wrapper) |
| `handler.go` | Connect-RPC service implementation |
| `server.go` | Handler assembly + middleware stack |
| `config.go` | `ConfigFromEnv()` |
| `auth.go` | API key auth interceptor |
| `middleware.go` | Logging, security headers, rate limiting, CORS |
| `proto/embedder/v1/embedder.proto` | Service definition |
