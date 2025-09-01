package mistral

import (
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

var (
	logger = log.New(os.Stdout, "mistral-client: ", log.LstdFlags|log.Lshortfile)
)

// StringFromParts returns the content of a multi-parts message as a string.
// The multiple parts are concatenated with a newline character.
func StringFromParts(content []*ai.Part) string {
	var msg string
	for _, part := range content {
		if part.Kind == ai.PartText {
			msg += part.Text + "\n"
		} else {
			logger.Printf("Unexpected message content part: %v\n", part)
		}
	}

	return msg
}

func NewMistralMessageFromGenkit(msg *ai.Message) mistralclient.Message {
	return mistralclient.Message{
		Role:    RoleFromGenkit(msg.Role),
		Content: StringFromParts(msg.Content),
	}
}

func NewGenkitMessageFromMistral(msg mistralclient.Message) *ai.Message {
	return &ai.Message{
		Role:    ai.Role(msg.Role),
		Content: []*ai.Part{ai.NewTextPart(msg.Content)},
	}
}

func MapResponse(mr *ai.ModelRequest, resp string) *ai.ModelResponse {
	aiMessage := &ai.Message{
		Role:    ai.RoleModel,
		Content: []*ai.Part{ai.NewTextPart(resp)},
	}

	return &ai.ModelResponse{
		Request: mr,
		Message: aiMessage,
	}
}

func MapMessagesToGenkit(messages []mistralclient.Message) []*ai.Message {
	m := make([]*ai.Message, len(messages))
	for i, msg := range messages {
		m[i] = NewGenkitMessageFromMistral(msg)
	}
	return nil
}

func MapMessagesToMistral(messages []*ai.Message) []mistralclient.Message {
	m := make([]mistralclient.Message, len(messages))
	for i, msg := range messages {
		m[i] = NewMistralMessageFromGenkit(msg)
	}
	return m
}

func RoleFromGenkit(role ai.Role) string {
	switch role {
	case ai.RoleUser:
		return mistralclient.RoleHuman
	case ai.RoleModel:
		return mistralclient.RoleAssistant
	case ai.RoleSystem:
		return mistralclient.RoleSystem
	default:
		return string(role) // Fallback to the string representation of the role
	}
}

func RoleFromMistral(role string) ai.Role {
	switch role {
	case mistralclient.RoleHuman:
		return ai.RoleUser
	case mistralclient.RoleAssistant:
		return ai.RoleModel
	case mistralclient.RoleSystem:
		return ai.RoleSystem
	default:
		return ai.RoleUser
	}
}
