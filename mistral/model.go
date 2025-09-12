package mistral

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/firebase/genkit/go/core/api"
	"github.com/thomas-marquis/genkit-mistral/internal"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"

	"github.com/firebase/genkit/go/ai"
)

const (
	defaultFakeResponseSize = 25
)

func defineSingleModel(c *mistralclient.Client, modelName string, modelInfo *ai.ModelInfo) ai.Model {
	return ai.NewModel(
		api.NewName(providerID, modelName),
		&ai.ModelOptions{
			Label:    modelInfo.Label,
			Stage:    modelInfo.Stage,
			Supports: modelInfo.Supports,
			Versions: modelInfo.Versions,
		},
		func(ctx context.Context, mr *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			cfg, err := getConfigFromRequest(mr)
			if err != nil {
				return nil, err
			}

			if len(mr.Messages) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}
			messages := mapMessagesToMistral(mr.Messages)

			var formatOpt mistralclient.ChatCompletionOption
			if mr.Output.Constrained && mr.Output.Format == "json" {
				formatOpt = mistralclient.WithResponseJsonSchema(mr.Output.Schema)
			} else {
				formatOpt = mistralclient.WithResponseTextFormat()
			}

			opts := []mistralclient.ChatCompletionOption{formatOpt}
			if nbTools := len(mr.Tools); nbTools > 0 {
				if tc := mistralclient.NewToolChoice(string(mr.ToolChoice)); tc != "" {
					opts = append(opts, mistralclient.WithToolChoice(tc))
				}

				tools := make([]mistralclient.ToolDefinition, 0, nbTools)
				for _, tool := range mr.Tools {
					tools = append(tools, mistralclient.ToolDefinition{
						Type: "function",
						Function: mistralclient.ToolFunctionDefinition{
							Name:        tool.Name,
							Description: tool.Description,
							Parameters:  tool.InputSchema,
							Strict:      false,
						},
					})
				}
				opts = append(opts, mistralclient.WithTools(tools))
			}

			response, err := c.ChatCompletion(ctx, messages, modelName, cfg, opts...)
			if err != nil {
				return nil, fmt.Errorf("failed to get chat completion: %w", err)
			}

			return mapResponse(mr, response), nil
		},
	)
}

func defineModel(c *mistralclient.Client, modelName string, modelInfos ai.ModelInfo) []ai.Model {
	var defined []ai.Model

	defined = append(defined, defineSingleModel(c, modelName, &modelInfos))

	if len(modelInfos.Versions) == 0 {
		return defined
	}

	for _, version := range modelInfos.Versions {
		if version == modelName {
			continue
		}
		mi := &ai.ModelInfo{
			Label:    version,
			Stage:    modelInfos.Stage,
			Supports: modelInfos.Supports,
			Versions: modelInfos.Versions,
		}
		defined = append(defined, defineSingleModel(c, version, mi))
	}

	return defined
}

func defineFakeModel() ai.Model {
	modelName := "fake-completion"
	return ai.NewModel(
		api.NewName(providerID, modelName),
		&ai.ModelOptions{
			Label: strings.ToTitle(modelName),
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				SystemRole: true,
				Media:      false,
				Tools:      true,
			},
			Versions: []string{"fake-completion"},
		},
		func(ctx context.Context, mr *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			cfg, err := getConfigFromRequest(mr)
			if err != nil {
				return nil, err
			}

			if len(mr.Messages) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}

			nbWords := calculateFakeWordCount(cfg.Temperature, cfg.MaxOutputTokens)

			fakeResponse, err := internal.FakeText(nbWords)
			if err != nil {
				return nil, fmt.Errorf("failed to generate fake response: %w", err)
			}

			return mapResponseFromText(mr, fakeResponse), nil
		},
	)
}

func getConfigFromRequest(mr *ai.ModelRequest) (*mistralclient.ModelConfig, error) {
	if mr.Config == nil {
		return &mistralclient.ModelConfig{}, nil
	}
	switch m := mr.Config.(type) {
	case *mistralclient.ModelConfig:
		return m, nil
	case mistralclient.ModelConfig:
		return &m, nil
	case map[string]any:
		return mistralclient.NewModelConfigFromRaw(m), nil
	}
	return nil, fmt.Errorf("invalid model request config type: expected *mistral.ModelConfig, got %T", mr.Config)
}

// calculateFakeWordCount determines the number of words to generate for the fake model response.
// The calculation is based on the temperature and maxOutputTokens parameters.
func calculateFakeWordCount(temperature float64, maxOutputTokens int) int {
	n := maxOutputTokens
	if n == 0 {
		n = defaultFakeResponseSize
	}

	maxWords := n / 3
	if temperature == 0 {
		return maxWords
	}

	if temperature < 0 {
		temperature = 0
	}
	if temperature > 1.0 {
		temperature = 1.0
	}

	minWords := n / 6
	if minWords >= maxWords {
		return maxWords
	}

	exponent := math.Pow(10, 2*temperature-1)

	randomFactor := rand.Float64()
	skewedRandomFactor := math.Pow(randomFactor, exponent)

	wordCountRange := float64(maxWords - minWords)
	words := float64(minWords) + wordCountRange*skewedRandomFactor

	return int(math.Round(words))
}
