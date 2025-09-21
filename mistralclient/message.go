package mistralclient

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

// ToolFunctionDefinition describes a function for a tool.
type ToolFunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Strict      bool           `json:"strict"`
	Parameters  map[string]any `json:"parameters"`
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
