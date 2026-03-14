// Package openaicompat provides a Genkit plugin for any OpenAI-compatible
// embedding API (LM Studio, Ollama, LocalAI, vLLM, etc.).
//
// Any server that exposes an OpenAI-compatible /v1/embeddings endpoint can be
// used with this plugin.
//
// # Quick start
//
//	g := genkit.Init(ctx, genkit.WithPlugins(&openaicompat.Plugin{
//	    Provider: "lmstudio",
//	    BaseURL:  "http://localhost:1234/v1",
//	    Embedders: []openaicompat.EmbedderDef{{Name: "nomic-embed-text-v1.5"}},
//	}))
//
//	resp, _ := genkit.Embed(ctx, g,
//	    ai.WithEmbedderName("lmstudio/nomic-embed-text-v1.5"),
//	    ai.WithTextDocs("hello world"),
//	)
package openaicompat

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/plugins/compat_oai"
)

// Plugin is a Genkit plugin that connects to any OpenAI-compatible embedding API.
type Plugin struct {
	// Provider is the name prefix for registered embedders (e.g. "lmstudio", "ollama").
	Provider string

	// BaseURL is the server URL (e.g. "http://localhost:1234/v1").
	BaseURL string

	// APIKey is the API key for authentication. Use a dummy value if not required.
	APIKey string

	// Embedders lists the embedding models to register.
	Embedders []EmbedderDef

	compat *compat_oai.OpenAICompatible
}

// EmbedderDef defines an embedding model available on the server.
type EmbedderDef struct {
	// Name is the model identifier.
	// The Genkit embedder reference will be "<Provider>/<Name>".
	Name string
}

func (p *Plugin) Name() string { return p.Provider }

func (p *Plugin) Init(ctx context.Context) []api.Action {
	apiKey := p.APIKey
	if apiKey == "" {
		apiKey = "no-key"
	}
	p.compat = &compat_oai.OpenAICompatible{
		Provider: p.Provider,
		APIKey:   apiKey,
		BaseURL:  p.BaseURL,
	}
	actions := p.compat.Init(ctx)

	for _, e := range p.Embedders {
		emb := p.compat.DefineEmbedder(p.Provider, e.Name, &ai.EmbedderOptions{
			Label: p.Provider + ": " + e.Name,
		})
		actions = append(actions, emb.(api.Action))
	}

	return actions
}
