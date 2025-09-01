package mistralclient

// Message is a Mistral chat message representation
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
