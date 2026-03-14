package embedder_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/laenen-partners/embedder"
)

func newGoogleAIEmbedder(t *testing.T) *embedder.Embedder {
	t.Helper()
	if os.Getenv("GOOGLE_API_KEY") == "" {
		t.Skip("GOOGLE_API_KEY not set, skipping Google AI live test")
	}
	return embedder.New(context.Background(),
		embedder.WithModel("googleai/text-embedding-005"),
	)
}

func newOpenAICompatEmbedder(t *testing.T) *embedder.Embedder {
	t.Helper()
	if os.Getenv("OPENAI_COMPAT_URL") == "" {
		t.Skip("OPENAI_COMPAT_URL not set, skipping OpenAI-compatible live test")
	}
	if os.Getenv("OPENAI_COMPAT_MODEL") == "" {
		t.Skip("OPENAI_COMPAT_MODEL not set, skipping OpenAI-compatible live test")
	}

	provider := os.Getenv("OPENAI_COMPAT_PROVIDER")
	if provider == "" {
		provider = "openaicompat"
	}
	model := provider + "/" + os.Getenv("OPENAI_COMPAT_MODEL")
	return embedder.New(context.Background(), embedder.WithModel(model))
}

// assertValidEmbedding checks that an embedding vector has reasonable properties.
func assertValidEmbedding(t *testing.T, name string, values []float32) {
	t.Helper()

	if len(values) < 100 {
		t.Errorf("%s: embedding vector too short: len=%d", name, len(values))
	}

	var normSquared float64
	for _, x := range values {
		normSquared += float64(x) * float64(x)
	}
	norm := math.Sqrt(normSquared)
	if norm < 0.9 || norm > 1.1 {
		t.Errorf("%s: embedding vector not unit length: norm=%f", name, norm)
	}
}

func TestLive_GoogleAI_SingleText(t *testing.T) {
	emb := newGoogleAIEmbedder(t)
	ctx := context.Background()

	vectors, err := emb.Embed(ctx, []string{"The quick brown fox jumps over the lazy dog."})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(vectors))
	}
	assertValidEmbedding(t, "single", vectors[0])
}

func TestLive_GoogleAI_BatchTexts(t *testing.T) {
	emb := newGoogleAIEmbedder(t)
	ctx := context.Background()

	texts := []string{
		"Artificial intelligence is transforming software development.",
		"The weather is sunny today.",
		"Go is a statically typed programming language.",
	}

	vectors, err := emb.Embed(ctx, texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(vectors))
	}

	for i, vec := range vectors {
		assertValidEmbedding(t, texts[i], vec)
	}

	sim01 := cosineSimilarity(vectors[0], vectors[1])
	sim02 := cosineSimilarity(vectors[0], vectors[2])
	t.Logf("similarity(AI, weather)=%f  similarity(AI, Go)=%f", sim01, sim02)

	if sim02 < sim01 {
		t.Logf("note: expected tech topics to be more similar, but AI-Go=%f < AI-weather=%f", sim02, sim01)
	}
}

func TestLive_GoogleAI_DeterministicResults(t *testing.T) {
	emb := newGoogleAIEmbedder(t)
	ctx := context.Background()

	text := "Deterministic embedding test."

	v1, err := emb.Embed(ctx, []string{text})
	if err != nil {
		t.Fatalf("first Embed: %v", err)
	}

	v2, err := emb.Embed(ctx, []string{text})
	if err != nil {
		t.Fatalf("second Embed: %v", err)
	}

	sim := cosineSimilarity(v1[0], v2[0])
	if sim < 0.999 {
		t.Errorf("same text produced different embeddings: similarity=%f", sim)
	}
}

func TestLive_OpenAICompat_SingleText(t *testing.T) {
	emb := newOpenAICompatEmbedder(t)
	ctx := context.Background()

	vectors, err := emb.Embed(ctx, []string{"The quick brown fox jumps over the lazy dog."})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(vectors))
	}
	if len(vectors[0]) < 10 {
		t.Errorf("embedding vector too short: len=%d", len(vectors[0]))
	}
}

func TestLive_OpenAICompat_BatchTexts(t *testing.T) {
	emb := newOpenAICompatEmbedder(t)
	ctx := context.Background()

	texts := []string{"Hello world", "Goodbye world", "Something completely different"}

	vectors, err := emb.Embed(ctx, texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vectors) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(vectors))
	}
	for i, vec := range vectors {
		if len(vec) < 10 {
			t.Errorf("vectors[%d] too short: len=%d", i, len(vec))
		}
	}
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
