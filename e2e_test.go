package embedder_test

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/laenen-partners/embedder"
)

const testModel = "test/fake-embedder"

// newTestEmbedder creates an Embedder backed by a fake Genkit embedder that
// returns deterministic vectors: [0.1*(i+1), 0.2*(i+1), 0.3*(i+1)] per text.
func newTestEmbedder(ctx context.Context) *embedder.Embedder {
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

	return embedder.NewFromGenkit(g, testModel)
}

func TestEmbed_SingleText(t *testing.T) {
	ctx := context.Background()
	emb := newTestEmbedder(ctx)

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

	if vectors[0][0] != 0.1 {
		t.Errorf("vectors[0][0] = %f, want 0.1", vectors[0][0])
	}
}

func TestEmbed_BatchTexts(t *testing.T) {
	ctx := context.Background()
	emb := newTestEmbedder(ctx)

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
			t.Errorf("vectors[%d] dimensions = %d, want 3", i, len(vec))
			continue
		}
		want := float32(i+1) * 0.1
		if vec[0] != want {
			t.Errorf("vectors[%d][0] = %f, want %f", i, vec[0], want)
		}
	}
}

func TestEmbed_EmptyTexts(t *testing.T) {
	ctx := context.Background()
	emb := newTestEmbedder(ctx)

	vectors, err := emb.Embed(ctx, []string{})
	if err != nil {
		t.Fatalf("Embed: unexpected error: %v", err)
	}
	if vectors != nil {
		t.Fatalf("expected nil for empty texts, got %v", vectors)
	}
}

func TestEmbed_NilTexts(t *testing.T) {
	ctx := context.Background()
	emb := newTestEmbedder(ctx)

	vectors, err := emb.Embed(ctx, nil)
	if err != nil {
		t.Fatalf("Embed: unexpected error: %v", err)
	}
	if vectors != nil {
		t.Fatalf("expected nil for nil texts, got %v", vectors)
	}
}

func TestNewConfig_Defaults(t *testing.T) {
	t.Setenv("EMBEDDER_MODEL", "")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("OPENAI_COMPAT_URL", "")
	t.Setenv("OPENAI_COMPAT_PROVIDER", "")
	t.Setenv("OPENAI_COMPAT_MODEL", "")
	t.Setenv("OPENAI_COMPAT_API_KEY", "")

	cfg := embedder.NewConfig()

	if cfg.Model != "googleai/text-embedding-005" {
		t.Errorf("default Model = %q, want %q", cfg.Model, "googleai/text-embedding-005")
	}
	if cfg.OpenAICompatProvider != "openaicompat" {
		t.Errorf("default OpenAICompatProvider = %q, want %q", cfg.OpenAICompatProvider, "openaicompat")
	}
}

func TestNewConfig_EnvVars(t *testing.T) {
	t.Setenv("EMBEDDER_MODEL", "vertexai/text-embedding-005")
	t.Setenv("GOOGLE_API_KEY", "test-key")
	t.Setenv("OPENAI_COMPAT_URL", "http://localhost:1234/v1")
	t.Setenv("OPENAI_COMPAT_PROVIDER", "lmstudio")
	t.Setenv("OPENAI_COMPAT_MODEL", "nomic-embed")
	t.Setenv("OPENAI_COMPAT_API_KEY", "compat-key")

	cfg := embedder.NewConfig()

	if cfg.Model != "vertexai/text-embedding-005" {
		t.Errorf("Model = %q, want %q", cfg.Model, "vertexai/text-embedding-005")
	}
	if cfg.GoogleAPIKey != "test-key" {
		t.Errorf("GoogleAPIKey = %q, want %q", cfg.GoogleAPIKey, "test-key")
	}
	if cfg.OpenAICompatURL != "http://localhost:1234/v1" {
		t.Errorf("OpenAICompatURL = %q, want %q", cfg.OpenAICompatURL, "http://localhost:1234/v1")
	}
	if cfg.OpenAICompatProvider != "lmstudio" {
		t.Errorf("OpenAICompatProvider = %q, want %q", cfg.OpenAICompatProvider, "lmstudio")
	}
}

func TestNewConfig_OptionsOverrideEnv(t *testing.T) {
	t.Setenv("EMBEDDER_MODEL", "from-env")
	t.Setenv("GOOGLE_API_KEY", "env-key")

	cfg := embedder.NewConfig(
		embedder.WithModel("from-option"),
		embedder.WithGoogleAPIKey("option-key"),
	)

	if cfg.Model != "from-option" {
		t.Errorf("Model = %q, want %q", cfg.Model, "from-option")
	}
	if cfg.GoogleAPIKey != "option-key" {
		t.Errorf("GoogleAPIKey = %q, want %q", cfg.GoogleAPIKey, "option-key")
	}
}
