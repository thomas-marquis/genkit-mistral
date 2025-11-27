package mistralclient

import (
	"fmt"
	"strings"
)

type ToolChoiceType string

func (tc ToolChoiceType) String() string {
	return string(tc)
}

// NewToolChoiceType creates a new ToolChoice from a string.
func NewToolChoiceType(choice string) ToolChoiceType {
	switch strings.ToLower(choice) {
	case ToolChoiceAuto.String():
		return ToolChoiceAuto
	case ToolChoiceAny.String():
		return ToolChoiceAny
	case ToolChoiceNone.String():
		return ToolChoiceNone
	case "":
		return ""
	default:
		logger.Printf("Invalid tool choice type: %s. Using empty value.", choice)
		return ""
	}
}

const (
	// ToolChoiceAuto is the default mode. Model decides if it uses the tool or not.
	ToolChoiceAuto ToolChoiceType = "auto"

	// ToolChoiceAny forces the model to use a tool.
	ToolChoiceAny ToolChoiceType = "any"

	// ToolChoiceNone prevent model to use a tool.
	ToolChoiceNone ToolChoiceType = "none"
)

// Function describes a function for a tool.
type Function struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Strict      bool               `json:"strict,omitempty"`
	Parameters  PropertyDefinition `json:"parameters,omitempty"`
}

// Tool is a representation of a tool the LLM can use.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

func NewFuncTool(functionName, description string, parameters map[string]any) Tool {
	return Tool{
		Type: "function",
		Function: Function{
			Name:        functionName,
			Description: description,
			Strict:      false,
			Parameters:  NewPropertyDefinition(parameters),
		},
	}
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

type PropertyDefinition struct {
	AdditionalProperties bool                          `json:"additionalProperties,omitempty"`
	Description          string                        `json:"description"`
	Type                 string                        `json:"type"`
	Properties           map[string]PropertyDefinition `json:"properties,omitempty"`
	Default              any                           `json:"default,omitempty"`
}

// NewPropertyDefinition creates a PropertyDefinition from a provided map of parameters.
// It initializes top-level fields such as "description", "type", "additionalProperties", and "default".
// If the "properties" field is present, it recursively maps nested properties into PropertyDefinition.
// Returns an empty PropertyDefinition if the input map is nil.
//
// Example:
//
//	def := NewPropertyDefinition(map[string]any{
//	    "type": "object",
//	    "description": "A user object",
//	    "properties": map[string]any{
//	        "name": map[string]any{
//	            "type": "string",
//	            "description": "The user's name",
//	        },
//	        "age": map[string]any{
//	            "type": "integer",
//	            "default": 18,
//	        },
//	    },
//	})
func NewPropertyDefinition(parameters map[string]any) PropertyDefinition {
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
				mapped[k] = NewPropertyDefinition(m)
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
