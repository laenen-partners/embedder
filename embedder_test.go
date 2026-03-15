package embedder_test

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/laenen-partners/embedder"
)

const testModel = "test/fake-embedder"

// initTestGenkit creates a Genkit instance with a fake embedder that returns
// deterministic vectors (each dimension = float32(textIndex + 1) * 0.1).
func initTestGenkit(ctx context.Context) *genkit.Genkit {
	g := genkit.Init(ctx)

	genkit.DefineEmbedder(g, testModel, nil,
		func(_ context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			embeddings := make([]*ai.Embedding, len(req.Input))
			for i := range req.Input {
				embeddings[i] = &ai.Embedding{
					Embedding: []float32{float32(i+1) * 0.1, float32(i+1) * 0.2, float32(i+1) * 0.3},
				}
			}
			return &ai.EmbedResponse{Embeddings: embeddings}, nil
		},
	)

	return g
}

func TestEmbed_SingleText(t *testing.T) {
	ctx := context.Background()
	g := initTestGenkit(ctx)
	emb := embedder.NewFromGenkit(g, testModel)

	vectors, err := emb.Embed(ctx, []string{"hello world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(vectors) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(vectors))
	}

	if len(vectors[0]) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(vectors[0]))
	}

	wantFirst := float32(0.1)
	if vectors[0][0] != wantFirst {
		t.Errorf("vectors[0][0] = %f, want %f", vectors[0][0], wantFirst)
	}
}

func TestEmbed_BatchTexts(t *testing.T) {
	ctx := context.Background()
	g := initTestGenkit(ctx)
	emb := embedder.NewFromGenkit(g, testModel)

	texts := []string{"first", "second", "third"}
	vectors, err := emb.Embed(ctx, texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(vectors) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(vectors))
	}

	for i, vec := range vectors {
		if len(vec) != 3 {
			t.Errorf("embedding[%d] dimensions = %d, want 3", i, len(vec))
			continue
		}
		want := float32(i+1) * 0.1
		if vec[0] != want {
			t.Errorf("embedding[%d][0] = %f, want %f", i, vec[0], want)
		}
	}
}

func TestEmbed_EmptyTexts(t *testing.T) {
	ctx := context.Background()
	g := initTestGenkit(ctx)
	emb := embedder.NewFromGenkit(g, testModel)

	vectors, err := emb.Embed(ctx, []string{})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if vectors != nil {
		t.Fatalf("expected nil for empty texts, got %v", vectors)
	}
}

func TestEmbed_NilTexts(t *testing.T) {
	ctx := context.Background()
	g := initTestGenkit(ctx)
	emb := embedder.NewFromGenkit(g, testModel)

	vectors, err := emb.Embed(ctx, nil)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if vectors != nil {
		t.Fatalf("expected nil for nil texts, got %v", vectors)
	}
}

func TestNew_DefaultModel(t *testing.T) {
	ctx := context.Background()
	// New with no credentials still works — just no plugins loaded.
	emb := embedder.New(ctx)
	if emb == nil {
		t.Fatal("expected non-nil embedder")
	}
}
