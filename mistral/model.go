package mistral

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type ModelConfig struct {
	ai.GenerationCommonConfig
}

func defineModel(g *genkit.Genkit, client *Client, modelName string, versions []string) {
	genkit.DefineModel(g, providerID, modelName,
		&ai.ModelInfo{
			Label: strings.ToTitle(modelName),
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				SystemRole: true,
				Media:      false,
				Tools:      true,
			},
			Versions: versions,
		},
		func(ctx context.Context, mr *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			var cfg ModelConfig
			if mr.Config != nil {
				if typedCfg, ok := mr.Config.(*ModelConfig); ok {
					cfg = *typedCfg
				} else {
					return nil, fmt.Errorf("invalid configuration type: expected ModelConfig, got %T", mr.Config)
				}
			}
			var _ = cfg

			if mr.Docs != nil {

			}

			if len(mr.Messages) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}
			messages := mapMessagesToMistral(mr.Messages)
			response, err := client.ChatCompletion(ctx, messages, modelName)
			if err != nil {
				return nil, fmt.Errorf("failed to get chat completion: %w", err)
			}

			return mapResponse(mr, response.Content), nil
		},
	)
}
