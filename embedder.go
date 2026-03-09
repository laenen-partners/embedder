package embedder

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// Embedder wraps a Genkit embedder to produce vector embeddings from text.
type Embedder struct {
	g     *genkit.Genkit
	model string
}

// NewEmbedder creates an Embedder that uses the given Genkit instance and model name.
func NewEmbedder(g *genkit.Genkit, model string) *Embedder {
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
