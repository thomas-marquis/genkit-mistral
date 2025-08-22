package mistral

import "github.com/firebase/genkit/go/ai"

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

func RoleFromGenkit(role ai.Role) string {
	switch role {
	case ai.RoleUser:
		return RoleHuman
	case ai.RoleModel:
		return RoleAssistant
	case ai.RoleSystem:
		return RoleSystem
	default:
		return string(role) // Fallback to the string representation of the role
	}
}

func RoleFromMistral(role string) ai.Role {
	switch role {
	case RoleHuman:
		return ai.RoleUser
	case RoleAssistant:
		return ai.RoleModel
	case RoleSystem:
		return ai.RoleSystem
	default:
		return ai.RoleUser
	}
}

func newMistralMessageFromGenkit(msg *ai.Message) Message {
	return Message{
		Role:    RoleFromGenkit(msg.Role),
		Content: parseMsgContent(msg.Content),
	}
}

func newGenkitMessageFromMistral(msg Message) *ai.Message {
	return &ai.Message{
		Role:    ai.Role(msg.Role),
		Content: []*ai.Part{ai.NewTextPart(msg.Content)},
	}
}
