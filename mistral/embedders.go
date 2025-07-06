package mistral

import (
	"context"
	"fmt"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func defineEmbedder(g *genkit.Genkit, client *Client, modelName string, versions []string) {
	genkit.DefineEmbedder(g, providerID, modelName,
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
