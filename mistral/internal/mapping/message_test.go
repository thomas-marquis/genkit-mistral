package mapping_test

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral/internal/mapping"
	"github.com/thomas-marquis/mistral-client/mistral"
)

func TestMapToMistralMessage(t *testing.T) {
	t.Run("should map as a user message", func(t *testing.T) {
		t.Run("with simple text content", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					{
						Kind: ai.PartText,
						Text: "Hello",
					},
					{
						Kind: ai.PartText,
						Text: "World!",
					},
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleUser, res.Role())
			assert.Equal(t, "Hello\nWorld!", res.Content().String())
		})

		t.Run("with text and media contents", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					ai.NewTextPart("Hello world!"),
					ai.NewMediaPart("image/gif", "https://mycdn.net/myimage.gif"),
					ai.NewMediaPart("audio/midi", "base64_encoded_audio_data or audio_file_url or audio_file_uploaded_on_mistral_la_plateforme"),
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleUser, res.Role())

			chunks := res.Content().Chunks()
			assert.Equal(t, 3, len(chunks))

			assert.IsType(t, &mistral.TextContent{}, chunks[0])
			textChunk := chunks[0].(*mistral.TextContent)
			assert.Equal(t, mistral.ContentTypeText, textChunk.ContentType)
			assert.Equal(t, "Hello world!", textChunk.Text)

			assert.IsType(t, &mistral.ImageUrlContent{}, chunks[1])
			imageChunk := chunks[1].(*mistral.ImageUrlContent)
			assert.Equal(t, mistral.ContentTypeImageURL, imageChunk.ContentType)
			assert.Equal(t, "https://mycdn.net/myimage.gif", imageChunk.ImageURL)

			assert.IsType(t, &mistral.AudioContent{}, chunks[2])
			audioChunk := chunks[2].(*mistral.AudioContent)
			assert.Equal(t, mistral.ContentTypeAudio, audioChunk.ContentType)
			assert.Equal(t, "base64_encoded_audio_data or audio_file_url or audio_file_uploaded_on_mistral_la_plateforme", audioChunk.InputAudio)
		})
	})

	t.Run("should map as a system message", func(t *testing.T) {
		t.Run("with simple text content", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleSystem,
				Content: []*ai.Part{
					{
						Kind: ai.PartText,
						Text: "Hello",
					},
					{
						Kind: ai.PartText,
						Text: "World!",
					},
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleSystem, res.Role())
			assert.Equal(t, "Hello\nWorld!", res.Content().String())
		})
	})

	t.Run("should map as an assistant message", func(t *testing.T) {
		t.Run("with simple text content", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					{
						Kind: ai.PartText,
						Text: "Hello",
					},
					{
						Kind: ai.PartText,
						Text: "World!",
					},
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleAssistant, res.Role())
			assert.Equal(t, "Hello\nWorld!", res.Content().String())
		})

		t.Run("with text and media contents", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewTextPart("Hello world!"),
					ai.NewMediaPart("image/gif", "https://mycdn.net/myimage.gif"),
					ai.NewMediaPart("audio/midi", "base64_encoded_audio_data or audio_file_url or audio_file_uploaded_on_mistral_la_plateforme"),
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleAssistant, res.Role())

			chunks := res.Content().Chunks()
			assert.Equal(t, 3, len(chunks))

			assert.IsType(t, &mistral.TextContent{}, chunks[0])
			textChunk := chunks[0].(*mistral.TextContent)
			assert.Equal(t, mistral.ContentTypeText, textChunk.ContentType)
			assert.Equal(t, "Hello world!", textChunk.Text)

			assert.IsType(t, &mistral.ImageUrlContent{}, chunks[1])
			imageChunk := chunks[1].(*mistral.ImageUrlContent)
			assert.Equal(t, mistral.ContentTypeImageURL, imageChunk.ContentType)
			assert.Equal(t, "https://mycdn.net/myimage.gif", imageChunk.ImageURL)

			assert.IsType(t, &mistral.AudioContent{}, chunks[2])
			audioChunk := chunks[2].(*mistral.AudioContent)
			assert.Equal(t, mistral.ContentTypeAudio, audioChunk.ContentType)
			assert.Equal(t, "base64_encoded_audio_data or audio_file_url or audio_file_uploaded_on_mistral_la_plateforme", audioChunk.InputAudio)
		})

		t.Run("with no content and tool calls", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					{
						Kind: ai.PartToolRequest,
						ToolRequest: &ai.ToolRequest{
							Ref:   "ref12345",
							Name:  "add",
							Input: map[string]interface{}{"a": 1, "b": 2},
						},
					},
					{
						Kind: ai.PartToolRequest,
						ToolRequest: &ai.ToolRequest{
							Ref:   "ref6789",
							Name:  "inc",
							Input: map[string]interface{}{"x": 1},
						},
					},
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleAssistant, res.Role())
			assert.Empty(t, res.Content().String())

			msg := res.(*mistral.AssistantMessage)
			tcs := msg.ToolCalls
			assert.Equal(t, 2, len(tcs))

			assert.Equal(t, "ref12345", tcs[0].ID)
			assert.Equal(t, 0, tcs[0].Index)
			assert.Equal(t, "add", tcs[0].Function.Name)
			assert.Equal(t, mistral.JsonMap{"input": map[string]any{"a": 1, "b": 2}}, tcs[0].Function.Arguments)

			assert.Equal(t, "ref6789", tcs[1].ID)
			assert.Equal(t, 1, tcs[1].Index)
			assert.Equal(t, "inc", tcs[1].Function.Name)
			assert.Equal(t, mistral.JsonMap{"input": map[string]any{"x": 1}}, tcs[1].Function.Arguments)
		})
	})

	t.Run("should map as an tool message", func(t *testing.T) {
		t.Run("with tool response content", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleTool,
				Content: []*ai.Part{
					ai.NewToolResponsePart(&ai.ToolResponse{
						Name: "add",
						Ref:  "ref12345",
						Output: map[string]interface{}{
							"result": 12,
						},
					}),
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleTool, res.Role())

			toolMsg := res.(*mistral.ToolMessage)
			assert.Equal(t, "ref12345", toolMsg.ToolCallId)
			assert.Equal(t, "add", toolMsg.Name)
			assert.JSONEq(t, `{
				"result": 12
			}`, toolMsg.Content().String())
		})

		t.Run("with tool response and other text contents", func(t *testing.T) {
			// Given
			genkitMsg := &ai.Message{
				Role: ai.RoleTool,
				Content: []*ai.Part{
					ai.NewToolResponsePart(&ai.ToolResponse{
						Name: "add",
						Ref:  "ref12345",
						Output: map[string]interface{}{
							"result": 12,
						},
					}),
					ai.NewTextPart("Hello world!"),
				},
			}

			// When
			res, err := mapping.MapToMistralMessage(genkitMsg)

			// Then
			assert.NoError(t, err)
			assert.NotNil(t, res)
			assert.Equal(t, mistral.RoleTool, res.Role())

			toolMsg := res.(*mistral.ToolMessage)
			assert.Equal(t, "ref12345", toolMsg.ToolCallId)
			assert.Equal(t, "add", toolMsg.Name)

			chunks := toolMsg.Content().Chunks()
			assert.Equal(t, 2, len(chunks))

			assert.IsType(t, &mistral.TextContent{}, chunks[0])
			assert.JSONEq(t, `{
				"result": 12
			}`, chunks[0].(*mistral.TextContent).Text)

			assert.IsType(t, &mistral.TextContent{}, chunks[1])
			assert.Equal(t, "Hello world!", chunks[1].(*mistral.TextContent).Text)
		})
	})
}
