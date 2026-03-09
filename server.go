package embedder

import (
	"fmt"
	"net/http"

	"connectrpc.com/connect"

	"github.com/laenen-partners/embedder/gen/embedder/v1/embedderv1connect"
)

// New creates an http.Handler that mounts the connect-go EmbedderService.
// The handler includes request logging, security headers, and optional rate limiting and CORS.
func New(cfg Config, embedder *Embedder) (http.Handler, error) {
	if embedder == nil {
		return nil, fmt.Errorf("embedder: embedder is required")
	}

	mux := http.NewServeMux()

	// Health check endpoints (unauthenticated).
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})

	// Mount connect-go RPC handler with auth interceptor.
	var opts []connect.HandlerOption
	if len(cfg.APIKeys) > 0 {
		opts = append(opts, connect.WithInterceptors(NewAuthInterceptor(cfg.APIKeys)))
	}
	path, rpcHandler := embedderv1connect.NewEmbedderServiceHandler(
		NewHandler(embedder), opts...,
	)
	mux.Handle(path, rpcHandler)

	// Apply middleware stack: rate limiting (outermost) -> CORS -> logging -> security headers.
	var handler http.Handler = mux
	handler = SecurityHeaders(handler)
	handler = RequestLogging(handler)
	if len(cfg.CORSOrigins) > 0 {
		handler = CORS(cfg.CORSOrigins)(handler)
	}
	if cfg.RateLimit > 0 {
		handler = RateLimit(cfg.RateLimit, cfg.RateBurst)(handler)
	}

	return handler, nil
}
