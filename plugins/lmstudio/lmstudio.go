// Package lmstudio provides a Genkit plugin for LM Studio's OpenAI-compatible
// embedding API.
//
// LM Studio (https://lmstudio.ai) runs models locally and exposes an
// OpenAI-compatible /v1/embeddings endpoint. This plugin registers embedding
// models loaded in LM Studio with Genkit.
//
// # Quick start
//
//	g := genkit.Init(ctx, genkit.WithPlugins(&lmstudio.LMStudio{
//	    Embedders: []lmstudio.EmbedderDef{{Name: "text-embedding-nomic-embed-text-v1.5"}},
//	}))
//
//	resp, _ := genkit.Embed(ctx, g,
//	    ai.WithEmbedderName("lmstudio/text-embedding-nomic-embed-text-v1.5"),
//	    ai.WithTextDocs("hello world"),
//	)
package lmstudio

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/plugins/compat_oai"
)

const (
	provider   = "lmstudio"
	DefaultURL = "http://localhost:1234/v1"
)

// LMStudio is a Genkit plugin that connects to LM Studio's local API for embeddings.
type LMStudio struct {
	// BaseURL is the LM Studio server URL. Defaults to http://localhost:1234/v1.
	BaseURL string

	// Embedders lists the embedding models to register.
	Embedders []EmbedderDef

	compat *compat_oai.OpenAICompatible
}

// EmbedderDef defines an embedding model available in LM Studio.
type EmbedderDef struct {
	// Name is the model identifier as shown in LM Studio.
	// The Genkit embedder reference will be "lmstudio/<Name>".
	Name string
}

func (l *LMStudio) Name() string { return provider }

func (l *LMStudio) Init(ctx context.Context) []api.Action {
	baseURL := l.BaseURL
	if baseURL == "" {
		baseURL = DefaultURL
	}
	l.compat = &compat_oai.OpenAICompatible{
		Provider: provider,
		APIKey:   "lm-studio",
		BaseURL:  baseURL,
	}
	actions := l.compat.Init(ctx)

	for _, e := range l.Embedders {
		emb := l.compat.DefineEmbedder(provider, e.Name, &ai.EmbedderOptions{
			Label: "LM Studio: " + e.Name,
		})
		actions = append(actions, emb.(api.Action))
	}

	return actions
}
