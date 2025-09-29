package mistralclient

import "fmt"

// Message is a Mistral chat message representation
type Message struct {
	Role         string            `json:"role"`
	Content      string            `json:"content"`
	ToolCalls    []ToolCallRequest `json:"tool_calls,omitempty"`
	ToolCallId   string            `json:"tool_call_id,omitempty"`
	FunctionName string            `json:"name,omitempty"`
}

const (
	RoleHuman     = "user"
	RoleAssistant = "assistant"
	RoleSystem    = "system"
)

func NewHumanMessage(content string) Message {
	return Message{
		Role:    RoleHuman,
		Content: content,
	}
}

func NewAssistantMessage(content string) Message {
	return Message{
		Role:    RoleAssistant,
		Content: content,
	}
}

func NewSystemMessage(content string) Message {
	return Message{
		Role:    RoleSystem,
		Content: content,
	}
}

func (m Message) IsHuman() bool {
	return m.Role == RoleHuman
}

func (m Message) IsAssistant() bool {
	return m.Role == RoleAssistant
}

func (m Message) IsSystem() bool {
	return m.Role == RoleSystem
}

type PropertyDefinition struct {
	AdditionalProperties bool                          `json:"additionalProperties,omitempty"`
	Description          string                        `json:"description"`
	Type                 string                        `json:"type"`
	Properties           map[string]PropertyDefinition `json:"properties,omitempty"`
	Default              any                           `json:"default,omitempty"`
}

func MapFunctionParameters(parameters map[string]any) PropertyDefinition {
	pd := PropertyDefinition{}

	if parameters == nil {
		return pd
	}

	// Map top-level fields
	if v, ok := parameters["description"].(string); ok {
		pd.Description = v
	}
	if v, ok := parameters["type"].(string); ok {
		pd.Type = v
	}
	if v, ok := parameters["additionalProperties"].(bool); ok {
		pd.AdditionalProperties = v
	}
	if v, ok := parameters["default"]; ok {
		pd.Default = v
	}

	// Recursively map properties if present
	if props, ok := parameters["properties"].(map[string]any); ok {
		mapped := make(map[string]PropertyDefinition, len(props))
		for k, raw := range props {
			if m, ok := raw.(map[string]any); ok {
				mapped[k] = MapFunctionParameters(m)
			} else {
				// If not a map, attempt to coerce simple type definitions
				mapped[k] = PropertyDefinition{Type: toString(raw)}
			}
		}
		pd.Properties = mapped
	}

	return pd
}

// toString provides a best-effort string conversion for simple scalar types.
func toString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case fmt.Stringer:
		return t.String()
	case int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64, bool:
		return fmt.Sprintf("%v", v)
	default:
		return ""
	}
}

// ToolFunctionDefinition describes a function for a tool.
type ToolFunctionDefinition struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Strict      bool               `json:"strict,omitempty"`
	Parameters  PropertyDefinition `json:"parameters,omitempty"`
}

// ToolDefinition is a representation of a tool the LLM can use.
type ToolDefinition struct {
	Type     string                 `json:"type"`
	Function ToolFunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
	Name      string  `json:"name"`
	Arguments jsonMap `json:"arguments"`
}

// ToolCallRequest represents a tool call decided by the LLM.
// This object may be used to know which function to call and with which arguments.
type ToolCallRequest struct {
	ID       string             `json:"id"`
	Index    int                `json:"index,omitempty"`
	Function FunctionDefinition `json:"function"`
}

func NewToolCallRequest(id string, index int, funcName string, args any) ToolCallRequest {
	var a jsonMap
	if c, ok := args.(jsonMap); !ok {
		a = jsonMap{"input": args}
	} else {
		a = c
	}

	return ToolCallRequest{
		ID:       id,
		Index:    index,
		Function: FunctionDefinition{Name: funcName, Arguments: a},
	}
}
