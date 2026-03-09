package embedder

import (
	"context"
	"fmt"

	"connectrpc.com/connect"

	embedderv1 "github.com/laenen-partners/embedder/gen/embedder/v1"
	"github.com/laenen-partners/embedder/gen/embedder/v1/embedderv1connect"
)

// Handler implements the connect-go EmbedderServiceHandler.
type Handler struct {
	embedderv1connect.UnimplementedEmbedderServiceHandler
	embedder *Embedder
}

// NewHandler creates a connect-go RPC handler backed by the given Embedder.
func NewHandler(embedder *Embedder) *Handler {
	return &Handler{embedder: embedder}
}

func (h *Handler) Embed(ctx context.Context, req *connect.Request[embedderv1.EmbedRequest]) (*connect.Response[embedderv1.EmbedResponse], error) {
	if len(req.Msg.Texts) == 0 {
		return nil, connect.NewError(connect.CodeInvalidArgument, fmt.Errorf("at least one text is required"))
	}

	vectors, err := h.embedder.Embed(ctx, req.Msg.Texts)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	embeddings := make([]*embedderv1.Embedding, len(vectors))
	for i, vec := range vectors {
		embeddings[i] = &embedderv1.Embedding{Values: vec}
	}

	return connect.NewResponse(&embedderv1.EmbedResponse{
		Embeddings: embeddings,
	}), nil
}
