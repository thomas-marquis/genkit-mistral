package mistral

import (
	"context"
	"fmt"
	"math/rand"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/thomas-marquis/genkit-mistral/internal"
	"github.com/thomas-marquis/mistral-client/mistral"
)

const (
	defaultVectorSize = 1024
)

var (
	ErrNoEmbeddings = fmt.Errorf("no embeddings returned by the model")
)

type EmbeddingOptions struct {
	VectorSize int `json:"vectorSize,omitempty"`
}

func newEmbeddingOptionsFromRaw(r map[string]any) *EmbeddingOptions {
	return &EmbeddingOptions{
		VectorSize: internal.GetOr[int](r, "vectorSize", defaultVectorSize),
	}
}

func defineEmbedder(client mistral.Client, modelName string) ai.Embedder {
	return ai.NewEmbedder(
		api.NewName(providerID, modelName),
		&ai.EmbedderOptions{},
		func(ctx context.Context, mr *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			if len(mr.Input) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}

			texts := make([]string, len(mr.Input))
			for i, input := range mr.Input {
				texts[i] = StringFromParts(input.Content)
			}

			req := mistral.NewEmbeddingRequest(modelName, texts)
			embResp, err := client.Embeddings(ctx, req)
			if err != nil {
				return nil, fmt.Errorf("failed to get embedding: %w", err)
			}

			vectors := embResp.Embeddings()
			embeds := make([]*ai.Embedding, len(vectors))
			for i, vector := range vectors {
				embeds[i] = &ai.Embedding{
					Embedding: vector,
				}
			}

			if len(embeds) == 0 {
				return nil, ErrNoEmbeddings
			}

			return &ai.EmbedResponse{
				Embeddings: embeds,
			}, nil
		},
	)
}

func defineFakeEmbedder() ai.Embedder {
	modelName := "fake-embed"
	return ai.NewEmbedder(
		api.NewName(providerID, modelName),
		&ai.EmbedderOptions{
			Label:      strings.ToTitle(modelName),
			Dimensions: defaultVectorSize,
		},
		func(ctx context.Context, mr *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			if len(mr.Input) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}

			cfg, err := getEmbeddingOptionsFromRequest(mr)
			if err != nil {
				return nil, err
			}

			texts := make([]string, len(mr.Input))
			for i, input := range mr.Input {
				texts[i] = StringFromParts(input.Content)
			}

			vecSize := cfg.VectorSize
			if vecSize == 0 {
				vecSize = defaultVectorSize
			}

			embeds := make([]*ai.Embedding, len(texts))
			for i := range texts {
				embeds[i] = &ai.Embedding{
					Embedding: createFakeVector(vecSize),
				}
			}

			return &ai.EmbedResponse{
				Embeddings: embeds,
			}, nil
		},
	)
}

func createFakeVector(size int) []float32 {
	embedding := make([]float32, size)
	for i := range embedding {
		embedding[i] = rand.Float32()
	}
	return embedding
}

func getEmbeddingOptionsFromRequest(mr *ai.EmbedRequest) (*EmbeddingOptions, error) {
	if mr.Options == nil {
		return &EmbeddingOptions{}, nil
	}
	switch m := mr.Options.(type) {
	case *EmbeddingOptions:
		return m, nil
	case map[string]any:
		return newEmbeddingOptionsFromRaw(m), nil
	}
	return nil, fmt.Errorf(
		"invalid embedding request options type: expected EmbeddingOptions, got %T", mr.Options)
}
