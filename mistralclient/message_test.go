package mistralclient_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func TestSystemMessage(t *testing.T) {
	t.Run("should be unmarshaled from json with string content", func(t *testing.T) {
		j := `{"role": "system", "content": "hello"}`
		var sm mistralclient.SystemMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &sm))
		assert.Equal(t, mistralclient.ContentString("hello"), sm.Content)
		assert.Equal(t, mistralclient.RoleSystem, sm.Type())
		assert.Equal(t, mistralclient.RoleSystem, sm.Role)
	})

	t.Run("should be unmarshaled from json with array content", func(t *testing.T) {
		j := `{
			"role": "system", 
			"content": [
				{"type": "text", "text": "hello"},
				{"type": "thinking", "thinking": [{"type": "text", "text": "world"}]}
			]
		}`
		var sm mistralclient.SystemMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &sm))
		assert.Equal(t, mistralclient.RoleSystem, sm.Role)
		assert.Len(t, sm.Content.Chunks(), 2)
		assert.Equal(t, "hello", sm.Content.Chunks()[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, "world", sm.Content.Chunks()[1].(*mistralclient.ThinkContent).Thinking[0].(*mistralclient.TextContent).Text)
	})

	t.Run("should be marshaled to json with simple string content", func(t *testing.T) {
		sm := mistralclient.NewSystemMessage(mistralclient.ContentString("hello"))
		j, err := json.Marshal(sm)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"system","content":"hello"}`, string(j))
	})

	t.Run("should be marshaled to json with chunked content", func(t *testing.T) {
		sm := mistralclient.NewSystemMessage(
			mistralclient.ContentChunks{
				mistralclient.NewTextContent("hello"),
				mistralclient.NewThinkContent(mistralclient.NewTextContent("world")),
			},
		)
		j, err := json.Marshal(sm)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"system","content":[{"type":"text","text":"hello"},{"type":"thinking","closed":true,"thinking":[{"type":"text","text":"world"}]}]}`, string(j))
	})
}

func TestUserMessage(t *testing.T) {
	t.Run("should be unmarshaled from json with chunked content", func(t *testing.T) {
		j := `{
			"role": "user", 
			"content": [
				{"type": "text", "text": "hello"},
				{"image_url": "https://example.com/image.png", "type": "image_url"},
				{"document_name": "document.pdf", "document_url": "https://example.com/document.pdf", "type": "document_url"},
				{"reference_ids": [1, 2, 3, 5, 8], "type": "reference"},
				{"file_id": "1234567890", "type": "file"},
				{"type": "thinking", "thinking": [{"type": "text", "text": "world"}]},
				{"input_audio": "https://example.com/audio.mp3", "type": "input_audio"}
			]
		}`
		var um mistralclient.UserMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &um))
		assert.Equal(t, mistralclient.RoleUser, um.Role)
		assert.Len(t, um.Content.Chunks(), 7)

		assert.Equal(t, "hello", um.Content.Chunks()[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, "https://example.com/image.png", um.Content.Chunks()[1].(*mistralclient.ImageUrlContent).ImageURL)
		assert.Equal(t, "document.pdf", um.Content.Chunks()[2].(*mistralclient.DocumentUrlContent).DocumentName)
		assert.Equal(t, "https://example.com/document.pdf", um.Content.Chunks()[2].(*mistralclient.DocumentUrlContent).DocumentURL)
		assert.Equal(t, []int{1, 2, 3, 5, 8}, um.Content.Chunks()[3].(*mistralclient.ReferenceContent).ReferenceIds)
		assert.Equal(t, "1234567890", um.Content.Chunks()[4].(*mistralclient.FileContent).FileId)
		assert.Equal(t, "world", um.Content.Chunks()[5].(*mistralclient.ThinkContent).Thinking[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, "https://example.com/audio.mp3", um.Content.Chunks()[6].(*mistralclient.AudioContent).InputAudio)

	})

	t.Run("should be unmarshalled from json with simple string content", func(t *testing.T) {
		j := `{"role": "user", "content": "hello"}`
		var um mistralclient.UserMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &um))
		assert.Equal(t, mistralclient.ContentString("hello"), um.Content)
		assert.Equal(t, mistralclient.RoleUser, um.Role)
	})

	t.Run("should be unmarshalled from json with null content", func(t *testing.T) {
		j := `{"role": "user", "content": null}`
		var um mistralclient.UserMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &um))
		assert.Equal(t, nil, um.Content)
		assert.Equal(t, mistralclient.RoleUser, um.Role)
	})

	t.Run("should be unmarshalled from json with omitted content", func(t *testing.T) {
		j := `{"role": "user"}`
		var um mistralclient.UserMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &um))
		assert.Equal(t, nil, um.Content)
		assert.Equal(t, mistralclient.RoleUser, um.Role)
	})

	t.Run("should be marshaled to json with simple string content", func(t *testing.T) {
		um := mistralclient.NewUserMessage(mistralclient.ContentString("hello"))
		j, err := json.Marshal(um)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"user","content":"hello"}`, string(j))
	})

	t.Run("should be marshaled to json with chunked content", func(t *testing.T) {
		um := mistralclient.NewUserMessage(
			mistralclient.ContentChunks{
				mistralclient.NewTextContent("hello"),
				mistralclient.NewThinkContent(mistralclient.NewTextContent("world")),
			},
		)
		j, err := json.Marshal(um)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"user","content":[{"type":"text","text":"hello"},{"type":"thinking","closed":true,"thinking":[{"type":"text","text":"world"}]}]}`, string(j))
	})
}

func TestAssistantMessage(t *testing.T) {
	t.Run("should be unmarshaled from json with chunked content", func(t *testing.T) {
		j := `{
			"role": "assistant", 
			"content": [
				{"type": "text", "text": "hello"},
				{"image_url": "https://example.com/image.png", "type": "image_url"},
				{"document_name": "document.pdf", "document_url": "https://example.com/document.pdf", "type": "document_url"},
				{"reference_ids": [1, 2, 3, 5, 8], "type": "reference"},
				{"file_id": "1234567890", "type": "file"},
				{"type": "thinking", "thinking": [{"type": "text", "text": "world"}]},
				{"input_audio": "https://example.com/audio.mp3", "type": "input_audio"}
			]
		}`
		var am mistralclient.AssistantMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &am))
		assert.Equal(t, mistralclient.RoleAssistant, am.Role)
		assert.Equal(t, mistralclient.RoleAssistant, am.Type())
		assert.Equal(t, false, am.Prefix)
		assert.Nil(t, am.ToolCalls)
		assert.Len(t, am.Content.Chunks(), 7)

		assert.Equal(t, "hello", am.Content.Chunks()[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, "https://example.com/image.png", am.Content.Chunks()[1].(*mistralclient.ImageUrlContent).ImageURL)
		assert.Equal(t, "document.pdf", am.Content.Chunks()[2].(*mistralclient.DocumentUrlContent).DocumentName)
		assert.Equal(t, "https://example.com/document.pdf", am.Content.Chunks()[2].(*mistralclient.DocumentUrlContent).DocumentURL)
		assert.Equal(t, []int{1, 2, 3, 5, 8}, am.Content.Chunks()[3].(*mistralclient.ReferenceContent).ReferenceIds)
		assert.Equal(t, "1234567890", am.Content.Chunks()[4].(*mistralclient.FileContent).FileId)
		assert.Equal(t, "world", am.Content.Chunks()[5].(*mistralclient.ThinkContent).Thinking[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, "https://example.com/audio.mp3", am.Content.Chunks()[6].(*mistralclient.AudioContent).InputAudio)
	})

	t.Run("should be unmarshable with tool calls", func(t *testing.T) {
		j := `{
			"role": "assistant", 
			"content": [
				{"type": "text", "text": "hello"}
			],
			"tool_calls": [
				{
					"type": "function",
					"id": "123",
					"index": 0,
					"function": {
						"arguments": {"name": "toto"},
						"name": "testFunction"
					}
				}	
			]
		}`
		var am mistralclient.AssistantMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &am))
		assert.Equal(t, mistralclient.RoleAssistant, am.Role)
		assert.Len(t, am.ToolCalls, 1)
		assert.Equal(t, "123", am.ToolCalls[0].ID)
		assert.Equal(t, "testFunction", am.ToolCalls[0].Function.Name)
	})

	t.Run("should be unmarshalled from json with simple string content", func(t *testing.T) {
		j := `{"role": "assistant", "content": "hello"}`
		var am mistralclient.AssistantMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &am))
		assert.Equal(t, mistralclient.ContentString("hello"), am.Content)
		assert.Equal(t, mistralclient.RoleAssistant, am.Role)
	})

	t.Run("should be unmarshalled from json with null content", func(t *testing.T) {
		j := `{"role": "assistant", "content": null}`
		var am mistralclient.AssistantMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &am))
		assert.Equal(t, nil, am.Content)
		assert.Equal(t, mistralclient.RoleAssistant, am.Role)
	})

	t.Run("should be unmarshalled from json with omitted content", func(t *testing.T) {
		j := `{"role": "assistant"}`
		var am mistralclient.AssistantMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &am))
		assert.Equal(t, nil, am.Content)
		assert.Equal(t, mistralclient.RoleAssistant, am.Role)
	})

	t.Run("should be marshaled to json with simple string content", func(t *testing.T) {
		am := mistralclient.NewAssistantMessage(mistralclient.ContentString("hello"))
		j, err := json.Marshal(am)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"assistant","content":"hello"}`, string(j))
	})

	t.Run("should be marshaled to json with chunked content", func(t *testing.T) {
		am := mistralclient.NewAssistantMessage(
			mistralclient.ContentChunks{
				mistralclient.NewTextContent("hello"),
				mistralclient.NewThinkContent(mistralclient.NewTextContent("world")),
			},
		)
		j, err := json.Marshal(am)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"assistant","content":[{"type":"text","text":"hello"},{"type":"thinking","closed":true,"thinking":[{"type":"text","text":"world"}]}]}`, string(j))
	})

	t.Run("should be marshaled to json with tool calls", func(t *testing.T) {
		am := mistralclient.NewAssistantMessage(
			mistralclient.ContentString("coucou"),
			mistralclient.NewToolCall("123", 0, "testFunction", map[string]interface{}{"name": "toto"}),
		)
		j, err := json.Marshal(am)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"assistant","content":"coucou","tool_calls":[{"id":"123","index":0,"function":{"name":"testFunction","arguments":{"input":{"name":"toto"}}},"type":"function"}]}`, string(j))
	})
}

func TestToolMessage(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{
			"role": "tool",
			"content": [
				{"type": "text", "text": "hello"},
				{"image_url": "https://example.com/image.png", "type": "image_url"},
				{"document_name": "document.pdf", "document_url": "https://example.com/document.pdf", "type": "document_url"},
				{"reference_ids": [1, 2, 3, 5, 8], "type": "reference"},
				{"file_id": "1234567890", "type": "file"},
				{"type": "thinking", "thinking": [{"type": "text", "text": "world"}]},
				{"input_audio": "https://example.com/audio.mp3", "type": "input_audio"}
			],
			"name": "testFunction",
			"tool_call_id": "azerty"
		}`
		var tm mistralclient.ToolMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &tm))
		assert.Equal(t, mistralclient.RoleTool, tm.Role)
		assert.Equal(t, mistralclient.RoleTool, tm.Type())

		assert.Equal(t, "testFunction", tm.Name)
		assert.Equal(t, "azerty", tm.ToolCallId)

		assert.Len(t, tm.Content.Chunks(), 7)
		assert.Equal(t, "hello", tm.Content.Chunks()[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, "https://example.com/image.png", tm.Content.Chunks()[1].(*mistralclient.ImageUrlContent).ImageURL)
		assert.Equal(t, "document.pdf", tm.Content.Chunks()[2].(*mistralclient.DocumentUrlContent).DocumentName)
		assert.Equal(t, "https://example.com/document.pdf", tm.Content.Chunks()[2].(*mistralclient.DocumentUrlContent).DocumentURL)
		assert.Equal(t, []int{1, 2, 3, 5, 8}, tm.Content.Chunks()[3].(*mistralclient.ReferenceContent).ReferenceIds)
		assert.Equal(t, "1234567890", tm.Content.Chunks()[4].(*mistralclient.FileContent).FileId)
		assert.Equal(t, "world", tm.Content.Chunks()[5].(*mistralclient.ThinkContent).Thinking[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, "https://example.com/audio.mp3", tm.Content.Chunks()[6].(*mistralclient.AudioContent).InputAudio)
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		tm := mistralclient.NewToolMessage(
			"testFunction",
			"azerty",
			mistralclient.ContentChunks{
				mistralclient.NewTextContent("hello"),
				mistralclient.NewThinkContent(mistralclient.NewTextContent("world")),
			},
		)
		j, err := json.Marshal(tm)

		assert.NoError(t, err)
		assert.Equal(t, `{"role":"tool","content":[{"type":"text","text":"hello"},{"type":"thinking","closed":true,"thinking":[{"type":"text","text":"world"}]}],"name":"testFunction","tool_call_id":"azerty"}`, string(j))
	})
}
