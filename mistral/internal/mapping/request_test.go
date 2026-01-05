package mapping_test

import (
	"fmt"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral/internal/mapping"
	"github.com/thomas-marquis/mistral-client/mistral"
)

func TestMapRequestToMistral(t *testing.T) {
	t.Run("should map request with multiple messages", func(t *testing.T) {
		// Given
		mr := &ai.ModelRequest{
			Messages: []*ai.Message{
				{
					Role: ai.RoleSystem,
					Content: []*ai.Part{
						{
							Kind: ai.PartText,
							Text: "you are a useful assistant",
						},
					},
				},
				{
					Role: ai.RoleUser,
					Content: []*ai.Part{
						ai.NewMediaPart("audio/midi", "base64_encoded_audio_data"),
						{
							Kind: ai.PartText,
							Text: "Please transcribe this",
						},
					},
				},
			},
		}

		// When
		res, err := mapping.MapRequestToMistral("mistral-small-latest", mr, nil)

		// Then
		assert.NoError(t, err)
		assert.NotNil(t, res)
		assert.Equal(t, 2, len(res.Messages))
		assert.Equal(t, res.Model, "mistral-small-latest")

		msgs := res.Messages
		assert.Equal(t, mistral.RoleSystem, msgs[0].Role())
		assert.Equal(t, "you are a useful assistant", msgs[0].Content().String())

		assert.Equal(t, mistral.RoleUser, msgs[1].Role())
		chunks := msgs[1].Content().Chunks()
		assert.Equal(t, 2, len(chunks))

		assert.Equal(t, mistral.ContentTypeAudio, chunks[0].Type())
		assert.Equal(t, "base64_encoded_audio_data", chunks[0].(*mistral.AudioChunk).InputAudio)

		assert.Equal(t, mistral.ContentTypeText, chunks[1].Type())
		assert.Equal(t, "Please transcribe this", chunks[1].(*mistral.TextChunk).Text)
	})

	t.Run("should map request with completion config", func(t *testing.T) {
		// Given
		mr := &ai.ModelRequest{
			Messages: []*ai.Message{
				{
					Role: ai.RoleUser,
					Content: []*ai.Part{
						ai.NewTextPart("Say hi"),
					},
				},
			},
		}
		cfg := &mistral.CompletionConfig{
			MaxTokens:   100,
			Temperature: 0.5,
			TopP:        0.7,
			ResponseFormat: &mistral.ResponseFormat{
				Type: mistral.ResponseFormatJsonObject,
			},
			ToolChoice:        mistral.ToolChoiceRequired,
			ParallelToolCalls: false,
			FrequencyPenalty:  0.1,
			PresencePenalty:   0.3,
			N:                 1,
			RandomSeed:        42,
			SafePrompt:        true,
			Stop:              []string{"end", "."},
			Stream:            true, // Not supported yet
		}

		// When
		res, err := mapping.MapRequestToMistral("mistral-small-latest", mr, cfg)

		// Then
		assert.NoError(t, err)
		assert.NotNil(t, res)
		assert.Equal(t, "mistral-small-latest", res.Model)
		assert.Equal(t, cfg.MaxTokens, res.MaxTokens)
		assert.Equal(t, cfg.Temperature, res.Temperature)
		assert.Equal(t, cfg.TopP, res.TopP)
		assert.Equal(t, cfg.ToolChoice, res.ToolChoice)
		assert.Equal(t, cfg.FrequencyPenalty, res.FrequencyPenalty)
		assert.Equal(t, cfg.PresencePenalty, res.PresencePenalty)
		assert.Equal(t, cfg.N, res.N)
		assert.Equal(t, cfg.RandomSeed, res.RandomSeed)
		assert.Equal(t, cfg.SafePrompt, res.SafePrompt)
		assert.False(t, res.Stream)
	})

	for _, tc := range []struct {
		mistralToolChoice mistral.ToolChoiceType
		genkitToolChoice  ai.ToolChoice
	}{
		{mistral.ToolChoiceAuto, ai.ToolChoiceAuto},
		{mistral.ToolChoiceAny, ai.ToolChoiceRequired},
		{mistral.ToolChoiceNone, ai.ToolChoiceNone},
	} {
		t.Run(fmt.Sprintf("should map request with tools with tool choice %s", tc.mistralToolChoice), func(t *testing.T) {
			// Given
			mr := &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							{
								Kind: ai.PartText,
								Text: "Tell me a joke",
							},
						},
					},
				},
				Tools: []*ai.ToolDefinition{
					{
						Description: "create a joke",
						InputSchema: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"category": map[string]any{
									"type":        "string",
									"description": "The category of the joke.",
								},
							},
							"required": []string{"category"},
						},
						Name: "jokeBuilder",
					},
				},
				ToolChoice: tc.genkitToolChoice,
			}

			// When
			res, err := mapping.MapRequestToMistral("mistral-small-latest", mr, &mistral.CompletionConfig{})

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, res.Model, "mistral-small-latest")

			assert.Equal(t, tc.mistralToolChoice, res.ToolChoice)

			tools := res.Tools
			assert.Equal(t, 1, len(tools))

			assert.Equal(t, "jokeBuilder", tools[0].Function.Name)
			assert.Equal(t, "create a joke", tools[0].Function.Description)
			assert.Equal(t, mistral.NewPropertyDefinition(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"category": map[string]any{
						"type":        "string",
						"description": "The category of the joke.",
					},
				},
				"required": []string{"category"},
			}), tools[0].Function.Parameters)
		})
	}

	t.Run("should return an error", func(t *testing.T) {
		t.Run("when message list is empty", func(t *testing.T) {
			// Given
			mr := &ai.ModelRequest{
				Messages: []*ai.Message{},
			}

			// When
			res, err := mapping.MapRequestToMistral("mistral-small-latest", mr, &mistral.CompletionConfig{})

			// Then
			assert.Nil(t, res)
			assert.Error(t, err)
			assert.Equal(t, "message list is empty", err.Error())
		})

		t.Run("when message list is nil", func(t *testing.T) {
			// Given
			mr := &ai.ModelRequest{
				Messages: nil,
			}

			// When
			res, err := mapping.MapRequestToMistral("mistral-small-latest", mr, &mistral.CompletionConfig{})

			// Then
			assert.Nil(t, res)
			assert.Error(t, err)
			assert.Equal(t, "message list is empty", err.Error())
		})

		t.Run("when empty model name is provided", func(t *testing.T) {
			// Given
			mr := &ai.ModelRequest{
				Messages: []*ai.Message{},
			}

			// When
			res, err := mapping.MapRequestToMistral("", mr, &mistral.CompletionConfig{})

			// Then
			assert.Nil(t, res)
			assert.Error(t, err)
			assert.Equal(t, "model name is empty", err.Error())
		})
	})

	t.Run("should map with json object format", func(t *testing.T) {
		// Given
		mr := &ai.ModelRequest{
			Messages: []*ai.Message{
				{
					Role: ai.RoleUser,
					Content: []*ai.Part{
						{
							Kind: ai.PartText,
							Text: "Say hello!",
						},
					},
				},
			},
			Output: &ai.ModelOutputConfig{
				Format:      ai.OutputFormatJSON,
				Constrained: true,
				Schema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"greeting": map[string]any{
							"type": "string",
						},
					},
				},
			},
		}

		// When
		res, err := mapping.MapRequestToMistral("mistral-small-latest", mr, nil)

		// Then
		assert.NoError(t, err)
		assert.NotNil(t, res)
		assert.Equal(t, "mistral-small-latest", res.Model)

		assert.Equal(t, &mistral.JsonSchema{
			Name: "responseJsonSchema",
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"greeting": map[string]any{
						"type": "string",
					},
				},
			},
			Strict: true,
		}, res.ResponseFormat.JsonSchema)
	})
}
