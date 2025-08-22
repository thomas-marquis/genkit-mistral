package mistral

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/thomas-marquis/genkit-mistral/internal"
)

const (
	defaultVectorSize = 1024
)

type EmbeddingOptions struct {
	VectorSize int `json:"vectorSize,omitempty"`
}

func newEmbeddingOptionsFromRaw(r map[string]any) *EmbeddingOptions {
	return &EmbeddingOptions{
		VectorSize: internal.GetOr[int](r, "vectorSize", defaultVectorSize),
	}
}

func defineEmbedder(g *genkit.Genkit, client *Client, modelName string, versions []string) {
	genkit.DefineEmbedder(g, providerID, modelName, &ai.EmbedderOptions{},
		func(ctx context.Context, mr *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			if len(mr.Input) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}

			texts := make([]string, len(mr.Input), len(mr.Input))
			for i, input := range mr.Input {
				texts[i] = parseMsgContent(input.Content)
			}

			vectors, err := client.TextEmbedding(ctx, texts, modelName)
			if err != nil {
				return nil, fmt.Errorf("failed to get embedding: %w", err)
			}

			embeds := make([]*ai.Embedding, len(vectors), len(vectors))
			for i, vector := range vectors {
				embeds[i] = &ai.Embedding{
					Embedding: vector,
				}
			}

			return &ai.EmbedResponse{
				Embeddings: embeds,
			}, nil
		},
	)
}

func defineFakeEmbedder(g *genkit.Genkit) {
	genkit.DefineEmbedder(g, providerID, "fake-embed", &ai.EmbedderOptions{},
		func(ctx context.Context, mr *ai.EmbedRequest) (*ai.EmbedResponse, error) {
			if len(mr.Input) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}

			cfg, err := getEmbeddingOptionsFromRequest(mr)
			if err != nil {
				return nil, err
			}

			texts := make([]string, len(mr.Input), len(mr.Input))
			for i, input := range mr.Input {
				texts[i] = parseMsgContent(input.Content)
			}

			vecSize := cfg.VectorSize
			if vecSize == 0 {
				vecSize = defaultVectorSize
			}

			embeds := make([]*ai.Embedding, len(texts), len(texts))
			for i, _ := range texts {
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
