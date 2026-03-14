package embedder

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"

	"github.com/laenen-partners/embedder/plugins/openaicompat"
)

// Embedder wraps a Genkit embedder to produce vector embeddings from text.
type Embedder struct {
	g     *genkit.Genkit
	model string
}

// New creates an Embedder from environment variables and options.
// It initialises Genkit with the appropriate plugins based on the resolved config.
func New(ctx context.Context, opts ...Option) *Embedder {
	cfg := NewConfig(opts...)

	var plugins []api.Plugin
	if cfg.GoogleAPIKey != "" {
		plugins = append(plugins, &googlegenai.GoogleAI{APIKey: cfg.GoogleAPIKey})
	}

	if cfg.OpenAICompatURL != "" {
		p := &openaicompat.Plugin{
			Provider: cfg.OpenAICompatProvider,
			BaseURL:  cfg.OpenAICompatURL,
			APIKey:   cfg.OpenAICompatAPIKey,
		}
		if cfg.OpenAICompatModel != "" {
			p.Embedders = []openaicompat.EmbedderDef{{Name: cfg.OpenAICompatModel}}
		}
		plugins = append(plugins, p)
	}

	g := genkit.Init(ctx, genkit.WithPlugins(plugins...))
	return &Embedder{g: g, model: cfg.Model}
}

// NewFromGenkit creates an Embedder from an existing Genkit instance and model name.
// Useful for testing with custom Genkit configurations.
func NewFromGenkit(g *genkit.Genkit, model string) *Embedder {
	return &Embedder{g: g, model: model}
}

// Embed generates embeddings for the given texts.
// Returns one []float32 per input text, in the same order.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	resp, err := genkit.Embed(ctx, e.g,
		ai.WithEmbedderName(e.model),
		ai.WithTextDocs(texts...),
	)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	if len(resp.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embed: expected %d embeddings, got %d", len(texts), len(resp.Embeddings))
	}

	result := make([][]float32, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		result[i] = emb.Embedding
	}
	return result, nil
}
