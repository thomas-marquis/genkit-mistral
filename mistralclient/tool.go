package mistralclient

import "fmt"

////////////////////////////////////////////////:

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

type FunctionCall struct {
	Name      string  `json:"name"`
	Arguments JsonMap `json:"arguments"`
}

// ToolCall represents a tool call decided by the LLM.
// This object may be used to know which function to call and with which arguments.
type ToolCall struct {
	ID       string       `json:"id"`
	Index    int          `json:"index"`
	Function FunctionCall `json:"function"`
	Type     string       `json:"type"`
}

func NewToolCall(id string, index int, funcName string, args any) ToolCall {
	var a JsonMap
	if c, ok := args.(JsonMap); !ok {
		a = JsonMap{"input": args}
	} else {
		a = c
	}

	return ToolCall{
		ID:       id,
		Index:    index,
		Function: FunctionCall{Name: funcName, Arguments: a},
		Type:     "function",
	}
}

type ToolChoice struct {
	Name string `json:"name"`
}
