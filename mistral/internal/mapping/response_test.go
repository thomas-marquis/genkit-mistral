package mapping_test

import (
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral/internal/mapping"
	"github.com/thomas-marquis/mistral-client/mistral"
)

func TestMapToGenkitResponse(t *testing.T) {
	t.Run("should map response with simple message", func(t *testing.T) {
		// Given
		resp := &mistral.ChatCompletionResponse{
			Choices: []mistral.ChatCompletionChoice{
				{
					Message:      mistral.NewAssistantMessageFromString("Hello simple human being!"),
					FinishReason: mistral.FinishReasonStop,
				},
			},
			Usage: mistral.UsageInfo{
				PromptTokens:     10,
				CompletionTokens: 100,
				TotalTokens:      110,
			},
			Latency: 3 * time.Second,
		}
		mr := &ai.ModelRequest{}

		// When
		res, err := mapping.MapToGenkitResponse(mr, resp)

		// Then
		assert.NoError(t, err)
		assert.NotNil(t, res)
		assert.Equal(t, mr, res.Request)
		assert.Equal(t, 100, res.Usage.OutputTokens)
		assert.Equal(t, 10, res.Usage.InputTokens)
		assert.Equal(t, 110, res.Usage.TotalTokens)
		assert.Equal(t, float64(3000), res.LatencyMs)
		assert.Equal(t, ai.FinishReason("stop"), res.FinishReason)
		assert.Equal(t, "Hello simple human being!", res.Text())

		content := res.Message.Content
		assert.Equal(t, 1, len(content))
		assert.Equal(t, ai.PartText, content[0].Kind)
		assert.Equal(t, "Hello simple human being!", content[0].Text)
	})

	t.Run("should map response with tool calls", func(t *testing.T) {
		// Given
		resp := &mistral.ChatCompletionResponse{
			Choices: []mistral.ChatCompletionChoice{
				{
					Message: mistral.NewAssistantMessageFromString("",
						mistral.NewToolCall("ref12345", 0, "add", map[string]interface{}{
							"a": 1,
							"b": 2,
						}),
						mistral.NewToolCall("ref67890", 1, "inc", nil)),
					FinishReason: mistral.FinishReasonToolCalls,
				},
			},
			Usage: mistral.UsageInfo{
				PromptTokens:     10,
				CompletionTokens: 100,
				TotalTokens:      110,
			},
			Latency: 3 * time.Second,
		}
		mr := &ai.ModelRequest{}

		// When
		res, err := mapping.MapToGenkitResponse(mr, resp)

		// Then
		assert.NoError(t, err)
		assert.NotNil(t, res)
		assert.Equal(t, mr, res.Request)
		assert.Equal(t, ai.FinishReason("stop"), res.FinishReason)
		assert.Empty(t, res.Text())

		content := res.Message.Content
		assert.Equal(t, 2, len(content))
		assert.Equal(t, ai.PartToolRequest, content[0].Kind)
		assert.Equal(t, "ref12345", content[0].ToolRequest.Ref)
		assert.Equal(t, "add", content[0].ToolRequest.Name)

		assert.Equal(t, ai.PartToolRequest, content[1].Kind)
		assert.Equal(t, "ref67890", content[1].ToolRequest.Ref)
		assert.Equal(t, "inc", content[1].ToolRequest.Name)
	})
}
