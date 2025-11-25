package mistralclient

import (
	"encoding/json"
	"fmt"
)

type ChatMessage interface {
	Type() Role
}

type SystemMessage struct {
	Role    Role    `json:"role"`
	Content Content `json:"content"`
}

var _ ChatMessage = (*SystemMessage)(nil)
var _ json.Unmarshaler = (*SystemMessage)(nil)

func NewSystemMessage(content Content) *SystemMessage {
	m := &SystemMessage{
		Role:    RoleSystem,
		Content: content,
	}
	return m
}

func (m *SystemMessage) Type() Role {
	return RoleSystem
}

func (m *SystemMessage) UnmarshalJSON(data []byte) error {
	var res map[string]any
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}

	m.Role = Role(res["role"].(string))

	if stringContent, ok := res["content"].(string); ok {
		m.Content = ContentString(stringContent)
	} else if listContent, ok := res["content"].([]any); ok {
		cts := ContentChunks{}
		for _, chunk := range listContent {
			t := chunk.(map[string]any)

			switch t["type"].(string) {
			case ContentTypeText.String():
				cts = append(cts, NewTextContent(t["text"].(string)))

			case ContentTypeThink.String():
				var tc ThinkContent
				if err := mapToStruct(t, &tc); err != nil {
					return err
				}
				cts = append(cts, &tc)
			}
		}

		m.Content = cts
	} else {
		return fmt.Errorf("invalid content type: %T", res["content"])
	}

	return nil
}

type UserMessage struct {
	Role    Role    `json:"role"`
	Content Content `json:"content"`
}

var _ ChatMessage = (*UserMessage)(nil)
var _ json.Unmarshaler = (*UserMessage)(nil)

func NewUserMessage(content Content) *UserMessage {
	return &UserMessage{
		Role:    RoleUser,
		Content: content,
	}
}

func (m *UserMessage) Type() Role {
	return RoleUser
}

func (m *UserMessage) UnmarshalJSON(data []byte) error {
	var res map[string]any
	var err error
	if err = json.Unmarshal(data, &res); err != nil {
		return err
	}
	m.Content, err = unmarshalMessageContent(res)
	if err != nil {
		return err
	}

	m.Role = Role(res["role"].(string))

	return nil
}

type AssistantMessage struct {
	Role      Role       `json:"role"`
	Content   Content    `json:"content"`
	Prefix    bool       `json:"prefix,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

var _ ChatMessage = (*AssistantMessage)(nil)
var _ json.Unmarshaler = (*AssistantMessage)(nil)

func NewAssistantMessage(content Content, toolCalls ...ToolCall) *AssistantMessage {
	return &AssistantMessage{
		Role:      RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
	}
}

func (m *AssistantMessage) Type() Role {
	return RoleAssistant
}

func (m *AssistantMessage) UnmarshalJSON(data []byte) error {
	var res map[string]any
	var err error
	if err = json.Unmarshal(data, &res); err != nil {
		return err
	}
	m.Content, err = unmarshalMessageContent(res)
	if err != nil {
		return err
	}

	m.Role = Role(res["role"].(string))

	if tcs, ok := res["tool_calls"]; ok {
		m.ToolCalls = make([]ToolCall, len(tcs.([]any)))
		for i, tc := range tcs.([]any) {
			m.ToolCalls[i] = ToolCall{}
			if err := mapToStruct(tc.(map[string]any), &m.ToolCalls[i]); err != nil {
				return err
			}
		}
	}

	return nil
}

type ToolMessage struct {
	Role       Role    `json:"role"`
	Content    Content `json:"content"`
	Name       string  `json:"name"`
	ToolCallId string  `json:"tool_call_id"`
}

var _ ChatMessage = (*ToolMessage)(nil)
var _ json.Unmarshaler = (*ToolMessage)(nil)

func NewToolMessage(name string, toolCallId string, content Content) *ToolMessage {
	return &ToolMessage{
		Role:       RoleTool,
		Content:    content,
		Name:       name,
		ToolCallId: toolCallId,
	}
}

func (m *ToolMessage) Type() Role {
	return RoleTool
}

func (m *ToolMessage) UnmarshalJSON(data []byte) error {
	var res map[string]any
	var err error
	if err = json.Unmarshal(data, &res); err != nil {
		return err
	}
	m.Content, err = unmarshalMessageContent(res)
	if err != nil {
		return err
	}

	m.Role = Role(res["role"].(string))
	m.Name = res["name"].(string)
	m.ToolCallId = res["tool_call_id"].(string)

	return nil
}

func unmarshalMessageContent(raw map[string]any) (Content, error) {
	var ct Content

	rawContent, exists := raw["content"]
	if !exists || rawContent == nil {
		return nil, nil
	}

	if stringContent, ok := rawContent.(string); ok {
		ct = ContentString(stringContent)
	} else if listContent, ok := rawContent.([]any); ok {
		cts := ContentChunks{}
		for _, chunk := range listContent {
			t := chunk.(map[string]any)
			var ptr ContentChunk

			switch t["type"].(string) {
			case ContentTypeText.String():
				ptr = &TextContent{}
			case ContentTypeImageURL.String():
				ptr = &ImageUrlContent{}
			case ContentTypeDocumentURL.String():
				ptr = &DocumentUrlContent{}
			case ContentTypeReference.String():
				ptr = &ReferenceContent{}
			case ContentTypeFile.String():
				ptr = &FileContent{}
			case ContentTypeThink.String():
				ptr = &ThinkContent{}
			case ContentTypeAudio.String():
				ptr = &AudioContent{}
			}
			if err := mapToStruct(t, ptr); err != nil {
				return nil, err
			}
			cts = append(cts, ptr)
		}

		ct = cts
	} else {
		return nil, fmt.Errorf("invalid content type: %T", raw["content"])
	}
	return ct, nil
}
