package embedder

import "os"

const defaultModel = "googleai/text-embedding-005"

// Config holds settings for the embedder.
type Config struct {
	Model                string // Genkit embedder model reference
	GoogleAPIKey         string // Google AI API key
	OpenAICompatURL      string // OpenAI-compatible server URL
	OpenAICompatProvider string // Provider name prefix (default: "openaicompat")
	OpenAICompatModel    string // Model name on the compatible server
	OpenAICompatAPIKey   string // API key for the compatible server
}

// Option configures the embedder. Options override environment variables.
type Option func(*Config)

// WithModel sets the embedding model reference.
func WithModel(model string) Option {
	return func(c *Config) { c.Model = model }
}

// WithGoogleAPIKey sets the Google AI API key.
func WithGoogleAPIKey(key string) Option {
	return func(c *Config) { c.GoogleAPIKey = key }
}

// WithOpenAICompat configures an OpenAI-compatible embedding server.
func WithOpenAICompat(url, provider, model, apiKey string) Option {
	return func(c *Config) {
		c.OpenAICompatURL = url
		c.OpenAICompatProvider = provider
		c.OpenAICompatModel = model
		c.OpenAICompatAPIKey = apiKey
	}
}

// NewConfig creates a Config populated from environment variables,
// then applies any provided options on top.
func NewConfig(opts ...Option) Config {
	cfg := Config{
		Model:                envOr("EMBEDDER_MODEL", defaultModel),
		GoogleAPIKey:         os.Getenv("GOOGLE_API_KEY"),
		OpenAICompatURL:      os.Getenv("OPENAI_COMPAT_URL"),
		OpenAICompatProvider: envOr("OPENAI_COMPAT_PROVIDER", "openaicompat"),
		OpenAICompatModel:    os.Getenv("OPENAI_COMPAT_MODEL"),
		OpenAICompatAPIKey:   os.Getenv("OPENAI_COMPAT_API_KEY"),
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
