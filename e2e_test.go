package embedder_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"connectrpc.com/connect"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	"github.com/laenen-partners/embedder"
	embedderv1 "github.com/laenen-partners/embedder/gen/embedder/v1"
	"github.com/laenen-partners/embedder/gen/embedder/v1/embedderv1connect"
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

// startServer creates a test server with a fake embedder and returns the
// httptest.Server and a Connect client.
func startServer(t *testing.T, cfgFn func(*embedder.Config)) (*httptest.Server, embedderv1connect.EmbedderServiceClient) {
	t.Helper()

	ctx := context.Background()
	g := initTestGenkit(ctx)
	emb := embedder.NewEmbedder(g, testModel)

	cfg := embedder.Config{}
	if cfgFn != nil {
		cfgFn(&cfg)
	}

	handler, err := embedder.New(cfg, emb)
	if err != nil {
		t.Fatalf("embedder.New: %v", err)
	}

	ts := httptest.NewServer(handler)
	t.Cleanup(ts.Close)

	client := embedderv1connect.NewEmbedderServiceClient(ts.Client(), ts.URL)
	return ts, client
}

// startHTTPServer creates an httptest.Server from an http.Handler and registers cleanup.
func startHTTPServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(handler)
	t.Cleanup(ts.Close)
	return ts
}

// withBearerAuth returns a Connect unary interceptor that sets the Authorization header.
func withBearerAuth(key string) connect.UnaryInterceptorFunc {
	return func(next connect.UnaryFunc) connect.UnaryFunc {
		return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
			req.Header().Set("Authorization", "Bearer "+key)
			return next(ctx, req)
		}
	}
}

func TestE2E_Embed_SingleText(t *testing.T) {
	_, client := startServer(t, nil)
	ctx := context.Background()

	resp, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{"hello world"},
	}))
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(resp.Msg.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Msg.Embeddings))
	}

	values := resp.Msg.Embeddings[0].Values
	if len(values) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(values))
	}

	// First text → index 0 → (0.1, 0.2, 0.3)
	wantFirst := float32(0.1)
	if values[0] != wantFirst {
		t.Errorf("values[0] = %f, want %f", values[0], wantFirst)
	}
}

func TestE2E_Embed_BatchTexts(t *testing.T) {
	_, client := startServer(t, nil)
	ctx := context.Background()

	texts := []string{"first", "second", "third"}
	resp, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: texts,
	}))
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(resp.Msg.Embeddings) != len(texts) {
		t.Fatalf("expected %d embeddings, got %d", len(texts), len(resp.Msg.Embeddings))
	}

	// Verify each embedding has the expected deterministic values.
	for i, emb := range resp.Msg.Embeddings {
		if len(emb.Values) != 3 {
			t.Errorf("embedding[%d] dimensions = %d, want 3", i, len(emb.Values))
			continue
		}
		want := float32(i+1) * 0.1
		if emb.Values[0] != want {
			t.Errorf("embedding[%d].Values[0] = %f, want %f", i, emb.Values[0], want)
		}
	}
}

func TestE2E_Embed_EmptyTexts(t *testing.T) {
	_, client := startServer(t, nil)
	ctx := context.Background()

	_, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{},
	}))
	if err == nil {
		t.Fatal("expected error for empty texts, got nil")
	}
	if code := connect.CodeOf(err); code != connect.CodeInvalidArgument {
		t.Fatalf("expected CodeInvalidArgument, got %v", code)
	}
}

func TestE2E_Embed_NilTexts(t *testing.T) {
	_, client := startServer(t, nil)
	ctx := context.Background()

	_, err := client.Embed(ctx, connect.NewRequest(&embedderv1.EmbedRequest{}))
	if err == nil {
		t.Fatal("expected error for nil texts, got nil")
	}
	if code := connect.CodeOf(err); code != connect.CodeInvalidArgument {
		t.Fatalf("expected CodeInvalidArgument, got %v", code)
	}
}

func TestE2E_HealthEndpoints(t *testing.T) {
	ts, _ := startServer(t, nil)

	for _, endpoint := range []string{"/healthz", "/readyz"} {
		resp, err := ts.Client().Get(ts.URL + endpoint)
		if err != nil {
			t.Fatalf("GET %s: %v", endpoint, err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Errorf("%s status = %d, want 200", endpoint, resp.StatusCode)
		}
		if string(body) != "ok" {
			t.Errorf("%s body = %q, want %q", endpoint, body, "ok")
		}
	}
}

func TestE2E_SecurityHeaders(t *testing.T) {
	ts, _ := startServer(t, nil)

	resp, err := ts.Client().Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	resp.Body.Close()

	checks := map[string]string{
		"X-Content-Type-Options": "nosniff",
		"X-Frame-Options":       "DENY",
		"X-Xss-Protection":      "0",
		"Referrer-Policy":       "strict-origin-when-cross-origin",
	}
	for header, want := range checks {
		got := resp.Header.Get(header)
		if got != want {
			t.Errorf("header %s = %q, want %q", header, got, want)
		}
	}
}

func TestE2E_Auth_RejectsUnauthenticated(t *testing.T) {
	_, client := startServer(t, func(cfg *embedder.Config) {
		cfg.APIKeys = []string{"valid-key-123"}
	})

	_, err := client.Embed(context.Background(), connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{"test"},
	}))
	if err == nil {
		t.Fatal("expected error for unauthenticated request, got nil")
	}
	if code := connect.CodeOf(err); code != connect.CodeUnauthenticated {
		t.Fatalf("expected CodeUnauthenticated, got %v", code)
	}
}

func TestE2E_Auth_RejectsInvalidKey(t *testing.T) {
	ts, _ := startServer(t, func(cfg *embedder.Config) {
		cfg.APIKeys = []string{"valid-key-123"}
	})

	client := embedderv1connect.NewEmbedderServiceClient(
		ts.Client(), ts.URL,
		connect.WithInterceptors(withBearerAuth("wrong-key")),
	)
	_, err := client.Embed(context.Background(), connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{"test"},
	}))
	if err == nil {
		t.Fatal("expected error for invalid key, got nil")
	}
	if code := connect.CodeOf(err); code != connect.CodeUnauthenticated {
		t.Fatalf("expected CodeUnauthenticated, got %v", code)
	}
}

func TestE2E_Auth_AcceptsValidKey(t *testing.T) {
	ts, _ := startServer(t, func(cfg *embedder.Config) {
		cfg.APIKeys = []string{"valid-key-123"}
	})

	client := embedderv1connect.NewEmbedderServiceClient(
		ts.Client(), ts.URL,
		connect.WithInterceptors(withBearerAuth("valid-key-123")),
	)
	resp, err := client.Embed(context.Background(), connect.NewRequest(&embedderv1.EmbedRequest{
		Texts: []string{"authenticated request"},
	}))
	if err != nil {
		t.Fatalf("Embed with valid key: %v", err)
	}
	if len(resp.Msg.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Msg.Embeddings))
	}
}

func TestE2E_RateLimit(t *testing.T) {
	ts, _ := startServer(t, func(cfg *embedder.Config) {
		cfg.RateLimit = 1
		cfg.RateBurst = 1
	})

	// First request should succeed.
	resp, err := ts.Client().Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatalf("first request: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", resp.StatusCode)
	}

	// Rapid follow-up requests should be rate limited.
	got429 := false
	for i := 0; i < 10; i++ {
		resp, err = ts.Client().Get(ts.URL + "/healthz")
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		resp.Body.Close()
		if resp.StatusCode == http.StatusTooManyRequests {
			got429 = true
			break
		}
	}
	if !got429 {
		t.Fatal("expected at least one 429 response from rate limiter")
	}
}

func TestE2E_ConfigFromEnv(t *testing.T) {
	envVars := []string{
		"EMBEDDER_MODEL", "API_KEYS", "RATE_LIMIT", "RATE_BURST", "CORS_ORIGINS",
	}
	for _, k := range envVars {
		t.Setenv(k, "")
	}

	cfg := embedder.ConfigFromEnv()
	if cfg.Model != "googleai/text-embedding-005" {
		t.Errorf("default Model = %q, want %q", cfg.Model, "googleai/text-embedding-005")
	}
	if cfg.RateLimit != 10 {
		t.Errorf("default RateLimit = %f, want 10", cfg.RateLimit)
	}
	if cfg.RateBurst != 20 {
		t.Errorf("default RateBurst = %d, want 20", cfg.RateBurst)
	}

	// Test custom values.
	t.Setenv("EMBEDDER_MODEL", "vertexai/text-embedding-005")
	t.Setenv("API_KEYS", "key1, key2")
	t.Setenv("RATE_LIMIT", "50")
	t.Setenv("RATE_BURST", "100")
	t.Setenv("CORS_ORIGINS", "http://localhost:3000,https://app.example.com")

	cfg = embedder.ConfigFromEnv()
	if cfg.Model != "vertexai/text-embedding-005" {
		t.Errorf("Model = %q, want %q", cfg.Model, "vertexai/text-embedding-005")
	}
	if len(cfg.APIKeys) != 2 || cfg.APIKeys[0] != "key1" || cfg.APIKeys[1] != "key2" {
		t.Errorf("APIKeys = %v, want [key1 key2]", cfg.APIKeys)
	}
	if cfg.RateLimit != 50 {
		t.Errorf("RateLimit = %f, want 50", cfg.RateLimit)
	}
	if cfg.RateBurst != 100 {
		t.Errorf("RateBurst = %d, want 100", cfg.RateBurst)
	}
	if len(cfg.CORSOrigins) != 2 {
		t.Errorf("CORSOrigins = %v, want 2 entries", cfg.CORSOrigins)
	}
}
