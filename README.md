# embedder

[![Go Reference](https://pkg.go.dev/badge/github.com/laenen-partners/embedder.svg)](https://pkg.go.dev/github.com/laenen-partners/embedder)

Go library for generating text embeddings via [Firebase Genkit](https://github.com/firebase/genkit). Supports Google AI, Vertex AI, and any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, etc.).

## Installation

```sh
go get github.com/laenen-partners/embedder
```

## Quick start

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/laenen-partners/embedder"
)

func main() {
	ctx := context.Background()

	// Reads GOOGLE_API_KEY from environment automatically.
	emb := embedder.New(ctx)

	vectors, err := emb.Embed(ctx, []string{
		"The quick brown fox jumps over the lazy dog.",
		"A fast auburn canine leaps above a sleepy hound.",
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("generated %d embeddings, first has %d dimensions\n",
		len(vectors), len(vectors[0]))
}
```

## API

### `embedder.New`

```go
func New(ctx context.Context, opts ...Option) *Embedder
```

Creates an `Embedder`. Initialises Genkit internally with the appropriate plugins based on environment variables and options.

### `embedder.NewFromGenkit`

```go
func NewFromGenkit(g *genkit.Genkit, model string) *Embedder
```

Creates an `Embedder` from an existing Genkit instance. Useful when you need full control over plugin initialisation.

### Options

| Option | Default | Description |
|---|---|---|
| `WithModel(model)` | `googleai/text-embedding-005` | Genkit embedder model reference |
| `WithGoogleAPIKey(key)` | `$GOOGLE_API_KEY` | Google AI API key |
| `WithOpenAICompat(url, provider, model, apiKey)` | from env vars | OpenAI-compatible server config |

### `Embedder.Embed`

```go
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error)
```

Generates embeddings for the given texts. Returns one `[]float32` per input text, in the same order. Returns `nil, nil` for empty input.

## Providers

### Google AI

Set `GOOGLE_API_KEY` in your environment, or pass it explicitly:

```go
emb := embedder.New(ctx)
// or
emb := embedder.New(ctx, embedder.WithGoogleAPIKey("your-key"))
```

### Vertex AI

```go
emb := embedder.New(ctx, embedder.WithModel("vertexai/text-embedding-005"))
```

### OpenAI-compatible (LM Studio, Ollama, vLLM, etc.)

Configure via environment variables:

```sh
export OPENAI_COMPAT_URL=http://localhost:1234/v1
export OPENAI_COMPAT_MODEL=nomic-embed-text-v1.5
export OPENAI_COMPAT_PROVIDER=lmstudio  # optional, defaults to "openaicompat"
```

```go
emb := embedder.New(ctx)
```

Or pass options directly:

```go
emb := embedder.New(ctx,
    embedder.WithModel("lmstudio/nomic-embed-text-v1.5"),
    embedder.WithOpenAICompat(
        "http://localhost:1234/v1",
        "lmstudio",
        "nomic-embed-text-v1.5",
        "",
    ),
)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDER_MODEL` | `googleai/text-embedding-005` | Genkit embedder model reference |
| `GOOGLE_API_KEY` | | Google AI API key |
| `OPENAI_COMPAT_URL` | | OpenAI-compatible server URL |
| `OPENAI_COMPAT_PROVIDER` | `openaicompat` | Provider name prefix |
| `OPENAI_COMPAT_MODEL` | | Model name on the compatible server |
| `OPENAI_COMPAT_API_KEY` | | API key for the compatible server |

## Testing

```sh
# Unit tests (no credentials required)
go test -v -count=1 ./...

# Google AI live tests
GOOGLE_API_KEY=your-key go test -v -count=1 -run 'TestLive_GoogleAI' ./...

# OpenAI-compatible live tests (e.g. LM Studio)
OPENAI_COMPAT_URL=http://localhost:1234/v1 \
OPENAI_COMPAT_MODEL=nomic-embed-text-v1.5 \
go test -v -count=1 -run 'TestLive_OpenAICompat' ./...
```

## Requirements

- Go 1.25+
- A Genkit-compatible embedding provider (Google AI, Vertex AI, or OpenAI-compatible endpoint)

## License

See [LICENSE](LICENSE) for details.
