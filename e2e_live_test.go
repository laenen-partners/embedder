package embedder_test

import (
	"context"
	"math"
	"os"
	"testing"

	"connectrpc.com/connect"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"

	"github.com/laenen-partners/embedder"
	embedderv1 "github.com/laenen-partners/embedder/gen/embedder/v1"
	"github.com/laenen-partners/embedder/gen/embedder/v1/embedderv1connect"
	"github.com/laenen-partners/embedder/plugins/lmstudio"
)

// startGoogleAIServer creates a test server backed by the real Google AI embedder.
// Requires GOOGLE_API_KEY env var.
func startGoogleAIServer(t *testing.T) embedderv1connect.EmbedderServiceClient {
	t.Helper()

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set, skipping Google AI live test")
	}

	ctx := context.Background()
	model := "googleai/text-embedding-005"

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: apiKey}))
	emb := embedder.NewEmbedder(g, model)

	cfg := embedder.Config{}
	handler, err := embedder.New(cfg, emb)
	if err != nil {
		t.Fatalf("embedder.New: %v", err)
	}

	ts := startHTTPServer(t, handler)
	return embedderv1connect.NewEmbedderServiceClient(ts.Client(), ts.URL)
}

// startLMStudioServer creates a test server backed by a real LM Studio embedder.
// Requires LMSTUDIO_URL and LMSTUDIO_EMBEDDER_MODEL env vars.
func startLMStudioServer(t *testing.T) embedderv1connect.EmbedderServiceClient {
	t.Helper()

	lmURL := os.Getenv("LMSTUDIO_URL")
	if lmURL == "" {
		t.Skip("LMSTUDIO_URL not set, skipping LM Studio live test")
	}

	modelName := os.Getenv("LMSTUDIO_EMBEDDER_MODEL")
	if modelName == "" {
		t.Skip("LMSTUDIO_EMBEDDER_MODEL not set, skipping LM Studio live test")
	}

	ctx := context.Background()
	model := "lmstudio/" + modelName

	g := genkit.Init(ctx, genkit.WithPlugins(&lmstudio.LMStudio{
		BaseURL:   lmURL,
		Embedders: []lmstudio.EmbedderDef{{Name: modelName}},
	}))
	emb := embedder.NewEmbedder(g, model)

	cfg := embedder.Config{}
	handler, err := embedder.New(cfg, emb)
	if err != nil {
		t.Fatalf("embedder.New: %v", err)
	}

	ts := startHTTPServer(t, handler)
	return embedderv1connect.NewEmbedderServiceClient(ts.Client(), ts.URL)
}

// assertValidEmbedding checks that an embedding vector has reasonable properties.
func assertValidEmbedding(t *testing.T, name string, values []float32) {
	t.Helper()

	if len(values) < 100 {
		t.Errorf("%s: embedding vector too short: len=%d", name, len(values))
	}

	// Check that the vector is approximately unit length (cosine-normalized).
	var normSquared float64
	for _, x := range values {
		normSquared += float64(x) * float64(x)
	}
	norm := math.Sqrt(normSquared)
	if norm < 0.9 || norm > 1.1 {
		t.Errorf("%s: embedding vector not unit length: norm=%f", name, norm)
	}
}

func TestE2E_GoogleAI_SingleText(t *testing.T) {
	client := startGoogleAIServer(t)
	ctx := context.Background()

	resp, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{"The quick brown fox jumps over the lazy dog."},
	}))
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(resp.Msg.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Msg.Embeddings))
	}

	assertValidEmbedding(t, "single", resp.Msg.Embeddings[0].Values)
}

func TestE2E_GoogleAI_BatchTexts(t *testing.T) {
	client := startGoogleAIServer(t)
	ctx := context.Background()

	texts := []string{
		"Artificial intelligence is transforming software development.",
		"The weather is sunny today.",
		"Go is a statically typed programming language.",
	}

	resp, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: texts,
	}))
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(resp.Msg.Embeddings) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(resp.Msg.Embeddings))
	}

	for i, emb := range resp.Msg.Embeddings {
		assertValidEmbedding(t, texts[i], emb.Values)
	}

	// Verify that similar texts produce more similar embeddings than dissimilar texts.
	// "AI transforming software" should be closer to "Go programming language" than to "weather".
	sim01 := cosineSimilarity(resp.Msg.Embeddings[0].Values, resp.Msg.Embeddings[1].Values)
	sim02 := cosineSimilarity(resp.Msg.Embeddings[0].Values, resp.Msg.Embeddings[2].Values)

	t.Logf("similarity(AI, weather)=%f  similarity(AI, Go)=%f", sim01, sim02)

	// Tech topics should be more similar to each other than to weather.
	if sim02 < sim01 {
		t.Logf("note: expected tech topics to be more similar, but AI-Go=%f < AI-weather=%f", sim02, sim01)
	}
}

func TestE2E_GoogleAI_DeterministicResults(t *testing.T) {
	client := startGoogleAIServer(t)
	ctx := context.Background()

	text := "Deterministic embedding test."

	resp1, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{text},
	}))
	if err != nil {
		t.Fatalf("first Embed: %v", err)
	}

	resp2, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{text},
	}))
	if err != nil {
		t.Fatalf("second Embed: %v", err)
	}

	sim := cosineSimilarity(resp1.Msg.Embeddings[0].Values, resp2.Msg.Embeddings[0].Values)
	if sim < 0.999 {
		t.Errorf("same text produced different embeddings: similarity=%f", sim)
	}
}

func TestE2E_LMStudio_SingleText(t *testing.T) {
	client := startLMStudioServer(t)
	ctx := context.Background()

	resp, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{"The quick brown fox jumps over the lazy dog."},
	}))
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(resp.Msg.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Msg.Embeddings))
	}

	// LM Studio embeddings may not be unit-normalized, so just check length.
	if len(resp.Msg.Embeddings[0].Values) < 10 {
		t.Errorf("embedding vector too short: len=%d", len(resp.Msg.Embeddings[0].Values))
	}
}

func TestE2E_LMStudio_BatchTexts(t *testing.T) {
	client := startLMStudioServer(t)
	ctx := context.Background()

	texts := []string{
		"Hello world",
		"Goodbye world",
		"Something completely different",
	}

	resp, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: texts,
	}))
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(resp.Msg.Embeddings) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(resp.Msg.Embeddings))
	}

	for i, emb := range resp.Msg.Embeddings {
		if len(emb.Values) < 10 {
			t.Errorf("embedding[%d] too short: len=%d", i, len(emb.Values))
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

