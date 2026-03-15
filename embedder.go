// Package embedder provides text embedding via Firebase Genkit.
//
// Usage:
//
//	emb := embedder.New(ctx) // reads GOOGLE_API_KEY from env
//	vectors, err := emb.Embed(ctx, []string{"hello world"})
package embedder

import (
	"context"
	"fmt"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"

	"github.com/laenen-partners/embedder/plugins/openaicompat"
)

const DefaultModel = "googleai/text-embedding-005"

// Embedder wraps a Genkit embedder to produce vector embeddings from text.
type Embedder struct {
	g     *genkit.Genkit
	model string
}

// Option configures an Embedder.
type Option func(*options)

type options struct {
	model                string
	googleAPIKey         string
	openAICompatURL      string
	openAICompatProvider string
	openAICompatModel    string
	openAICompatAPIKey   string
}

// WithModel sets the Genkit embedder model reference (e.g. "googleai/text-embedding-005").
// Defaults to DefaultModel, or EMBEDDER_MODEL env var if set.
func WithModel(model string) Option {
	return func(o *options) { o.model = model }
}

// WithGoogleAPIKey sets the Google AI API key.
// Defaults to GOOGLE_API_KEY env var.
func WithGoogleAPIKey(key string) Option {
	return func(o *options) { o.googleAPIKey = key }
}

// WithOpenAICompat configures an OpenAI-compatible embedding server.
func WithOpenAICompat(url, provider, model, apiKey string) Option {
	return func(o *options) {
		o.openAICompatURL = url
		o.openAICompatProvider = provider
		o.openAICompatModel = model
		o.openAICompatAPIKey = apiKey
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// New creates an Embedder. It initialises Genkit with the appropriate plugins
// based on environment variables and any provided options.
func New(ctx context.Context, opts ...Option) *Embedder {
	o := &options{
		model:                envOr("EMBEDDER_MODEL", DefaultModel),
		googleAPIKey:         os.Getenv("GOOGLE_API_KEY"),
		openAICompatURL:      os.Getenv("OPENAI_COMPAT_URL"),
		openAICompatProvider: envOr("OPENAI_COMPAT_PROVIDER", "openaicompat"),
		openAICompatModel:    os.Getenv("OPENAI_COMPAT_MODEL"),
		openAICompatAPIKey:   os.Getenv("OPENAI_COMPAT_API_KEY"),
	}
	for _, opt := range opts {
		opt(o)
	}

	var plugins []api.Plugin
	if o.googleAPIKey != "" {
		plugins = append(plugins, &googlegenai.GoogleAI{APIKey: o.googleAPIKey})
	}
	if o.openAICompatURL != "" {
		p := &openaicompat.Plugin{
			Provider: o.openAICompatProvider,
			BaseURL:  o.openAICompatURL,
			APIKey:   o.openAICompatAPIKey,
		}
		if o.openAICompatModel != "" {
			p.Embedders = []openaicompat.EmbedderDef{{Name: o.openAICompatModel}}
		}
		plugins = append(plugins, p)
	}

	g := genkit.Init(ctx, genkit.WithPlugins(plugins...))
	return &Embedder{g: g, model: o.model}
}

// NewFromGenkit creates an Embedder from an existing Genkit instance and model name.
// Useful when you need full control over Genkit initialisation.
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
