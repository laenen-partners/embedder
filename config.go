package embedder

import (
	"os"
	"strconv"
	"strings"
)

// Config holds settings for the embedder service.
type Config struct {
	Model       string   // Genkit embedder model reference (e.g. "googleai/text-embedding-005")
	APIKeys     []string // API keys for RPC authentication
	RateLimit   float64  // requests per second per IP (0 = disabled)
	RateBurst   int      // burst allowance per IP
	CORSOrigins []string // allowed CORS origins (empty = no CORS)
}

// ConfigFromEnv reads configuration from environment variables.
func ConfigFromEnv() Config {
	model := os.Getenv("EMBEDDER_MODEL")
	if model == "" {
		model = "googleai/text-embedding-005"
	}

	var apiKeys []string
	if keys := os.Getenv("API_KEYS"); keys != "" {
		for _, k := range strings.Split(keys, ",") {
			if trimmed := strings.TrimSpace(k); trimmed != "" {
				apiKeys = append(apiKeys, trimmed)
			}
		}
	}

	rateLimit := 10.0
	if v := os.Getenv("RATE_LIMIT"); v != "" {
		if parsed, err := strconv.ParseFloat(v, 64); err == nil {
			rateLimit = parsed
		}
	}

	rateBurst := 20
	if v := os.Getenv("RATE_BURST"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil {
			rateBurst = parsed
		}
	}

	var corsOrigins []string
	if v := os.Getenv("CORS_ORIGINS"); v != "" {
		for _, o := range strings.Split(v, ",") {
			if trimmed := strings.TrimSpace(o); trimmed != "" {
				corsOrigins = append(corsOrigins, trimmed)
			}
		}
	}

	return Config{
		Model:       model,
		APIKeys:     apiKeys,
		RateLimit:   rateLimit,
		RateBurst:   rateBurst,
		CORSOrigins: corsOrigins,
	}
}
