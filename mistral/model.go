package mistral

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/thomas-marquis/genkit-mistral/internal"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const (
	defaultFakeResponseSize = 25
)

type ModelConfig struct {
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
	Temperature     float64  `json:"temperature,omitempty"`
	TopK            int      `json:"topK,omitempty"`
	TopP            float64  `json:"topP,omitempty"`
	Version         string   `json:"version,omitempty"`
}

func newModelConfigFromRaw(r map[string]any) *ModelConfig {
	return &ModelConfig{
		MaxOutputTokens: internal.GetOrZero[int](r, "maxOutputTokens"),
		StopSequences:   internal.GetSliceOrNil[string](r, "stopSequences"),
		Temperature:     internal.GetOrZero[float64](r, "temperature"),
		TopK:            internal.GetOrZero[int](r, "topK"),
		TopP:            internal.GetOrZero[float64](r, "topP"),
		Version:         internal.GetOrZero[string](r, "version"),
	}
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
			cfg, err := getConfigFromRequest(mr)
			if err != nil {
				return nil, err
			}

			if len(mr.Messages) == 0 {
				return nil, fmt.Errorf("no messages provided in the model request")
			}
			messages := mapMessagesToMistral(mr.Messages)
			response, err := client.ChatCompletion(ctx, messages, modelName, cfg)
			if err != nil {
				return nil, fmt.Errorf("failed to get chat completion: %w", err)
			}

			return mapResponse(mr, response.Content), nil
		},
	)
}

func defineFakeModel(g *genkit.Genkit) {
	modelName := "fake-completion"
	genkit.DefineModel(g, providerID, modelName,
		&ai.ModelInfo{
			Label: strings.ToTitle(modelName),
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				SystemRole: true,
				Media:      false,
				Tools:      true,
			},
			Versions: []string{"fake-completion", "fake-completion-latest"},
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

			return mapResponse(mr, fakeResponse), nil
		},
	)
}

func getConfigFromRequest(mr *ai.ModelRequest) (*ModelConfig, error) {
	if mr.Config == nil {
		return &ModelConfig{}, nil
	}
	switch m := mr.Config.(type) {
	case *ModelConfig:
		return m, nil
	case map[string]any:
		return newModelConfigFromRaw(m), nil
	}
	return nil, fmt.Errorf("invalid model request config type: expected ModelConfig, got %T", mr.Config)
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

	// We map temperature to an exponent to skew the random number distribution.
	// An exponent < 1 skews towards 1 (maxWords), and an exponent > 1 skews towards 0 (minWords).
	// We map temperature [0,1] to exponent [~0.1, 10].
	// temp=0.5 corresponds to exponent=1 (uniform distribution).
	exponent := math.Pow(10, 2*temperature-1)

	// Generate a random factor and skew it.
	randomFactor := rand.Float64()
	skewedRandomFactor := math.Pow(randomFactor, exponent)

	// Map the skewed factor to the word count range.
	// Low temp -> high skewed factor -> closer to maxWords.
	// High temp -> low skewed factor -> closer to minWords.
	wordCountRange := float64(maxWords - minWords)
	words := float64(minWords) + wordCountRange*skewedRandomFactor

	return int(math.Round(words))
}
