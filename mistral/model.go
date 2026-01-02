package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/firebase/genkit/go/core/api"
	"github.com/thomas-marquis/genkit-mistral/internal"
	"github.com/thomas-marquis/genkit-mistral/mistral/internal/mapping"
	"github.com/thomas-marquis/mistral-client/mistral"

	"github.com/firebase/genkit/go/ai"
)

const (
	defaultFakeResponseSize = 25
)

var (
	ErrInvalidModelInput = fmt.Errorf("invalid model input")
)

func defineModel(c mistral.Client, modelInfo *ai.ModelInfo) ai.Model {
	return ai.NewModel(
		api.NewName(providerID, modelInfo.Label),
		&ai.ModelOptions{
			Label:    modelInfo.Label,
			Stage:    modelInfo.Stage,
			Supports: modelInfo.Supports,
			Versions: modelInfo.Versions,
		},
		func(ctx context.Context, mr *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			//cfg, err := configFromRequest(mr)
			//if err != nil {
			//	return nil, err
			//}

			req, err := mapping.MapRequestToMistral(modelInfo.Label, mr)
			if err != nil {
				return nil, err
			}

			response, err := c.ChatCompletion(ctx, req)
			if err != nil {
				return nil, fmt.Errorf("failed to get chat completion: %w", err)
			}

			mresp, err := mapping.MapToGenkitResponse(mr, response)
			if err != nil {
				return nil, err
			}

			return mresp, nil
		},
	)
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
			cfg, err := configFromRequest(mr)
			if err != nil {
				return nil, err
			}

			if len(mr.Messages) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}

			nbWords := calculateFakeWordCount(cfg.Temperature, cfg.MaxTokens)

			fakeResponse, err := internal.FakeText(nbWords)
			if err != nil {
				return nil, fmt.Errorf("failed to generate fake response: %w", err)
			}

			return mapResponseFromText(mr, fakeResponse), nil
		},
	)
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

func configFromRequest(req *ai.ModelRequest) (*mistral.CompletionConfig, error) {
	var result mistral.CompletionConfig

	switch config := req.Config.(type) {
	case mistral.CompletionConfig:
		result = config
	case *mistral.CompletionConfig:
		result = *config
	case map[string]any:
		jsonData, err := json.Marshal(config)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(jsonData, &result); err != nil {
			return nil, err
		}
	case nil:
		// Empty but valid config
	default:
		return nil, fmt.Errorf("unexpected config type: %T", req.Config)
	}

	return &result, nil
}
