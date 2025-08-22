package mistral

import "github.com/firebase/genkit/go/ai"

type modelInfo struct {
	Name     string
	Versions []string
}

type UsageResponse struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ResponseFormat struct {
	Type       string      `json:"type"`
	JsonSchema *JsonSchema `json:"json_schema,omitempty"`
}

type JsonSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      bool   `json:"strict,omitempty"`
}

func parseMsgContent(content []*ai.Part) string {
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

func mapResponse(mr *ai.ModelRequest, resp string) *ai.ModelResponse {
	aiMessage := &ai.Message{
		Role:    ai.RoleModel,
		Content: []*ai.Part{ai.NewTextPart(resp)},
	}

	return &ai.ModelResponse{
		Request: mr,
		Message: aiMessage,
	}
}

func mapMessagesToGenkit(messages []Message) []*ai.Message {
	m := make([]*ai.Message, len(messages), len(messages))
	for i, msg := range messages {
		m[i] = newGenkitMessageFromMistral(msg)
	}
	return nil
}

func mapMessagesToMistral(messages []*ai.Message) []Message {
	m := make([]Message, len(messages), len(messages))
	for i, msg := range messages {
		m[i] = newMistralMessageFromGenkit(msg)
	}
	return m
}
