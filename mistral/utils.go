package mistral

import (
	"encoding/json"
	"log"
	"os"
	"strings"

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

// SanitizeToolName formats a function name to be used as a reference in a tool call.
func SanitizeToolName(name string) string {
	runes := []rune(name)

	isAllowed := func(r rune) bool {
		return (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '_' || r == '-'
	}

	var b strings.Builder
	b.Grow(len(runes))

	for i := 0; i < len(runes); {
		r := runes[i]
		switch {
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_':
			b.WriteRune(r)
			i++
		case r == '-':
			// If pattern is "-<disallowed>-" convert to "_-" (skip the middle disallowed runes)
			j := i + 1
			skipped := 0
			for j < len(runes) && !isAllowed(runes[j]) {
				j++
				skipped++
			}
			if j < len(runes) && runes[j] == '-' && skipped > 0 {
				b.WriteRune('_')
				b.WriteRune('-')
				i = j + 1
			} else {
				b.WriteRune('-')
				i++
			}
		default:
			// drop any other disallowed characters
			i++
		}
	}

	result := b.String()
	if len(result) > 256 {
		result = result[:256]
	}
	return result
}

func newMistralMessageFromGenkit(msg *ai.Message) mistralclient.Message {
	content := msg.Content

	m := mistralclient.Message{
		Role: mapRoleFromGenkit(msg.Role),
	}

	for _, part := range content {
		switch part.Kind {
		case ai.PartText:
			m.Content += part.Text + "\n"
		case ai.PartToolRequest:
			m.ToolCalls = append(m.ToolCalls, mistralclient.NewToolCallRequest(
				part.ToolRequest.Ref, 0, part.ToolRequest.Name, part.ToolRequest.Input,
			))
		case ai.PartToolResponse:
			bytes, err := json.Marshal(part.ToolResponse.Output)
			if err != nil {
				logger.Printf("Failed to marshal tool response: %v\n", err)
			} else {
				m.Content += string(bytes) + "\n"
			}
			m.ToolCallId = part.ToolResponse.Ref
			m.FunctionName = part.ToolResponse.Name
		default:
			logger.Printf("Unexpected message content part kind: %v\n", part)
		}
	}

	return m
}

func newGenkitMessageFromMistral(msg mistralclient.Message) *ai.Message {
	return &ai.Message{
		Role:    ai.Role(msg.Role),
		Content: []*ai.Part{ai.NewTextPart(msg.Content)},
	}
}

func mapResponse(mr *ai.ModelRequest, resp mistralclient.ChatCompletionResponse) *ai.ModelResponse {
	var parts []*ai.Part

	response := &ai.ModelResponse{
		Request: mr,
		Usage: &ai.GenerationUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		},
		LatencyMs: float64(resp.Latency.Milliseconds()),
	}

	if len(resp.Choices) == 0 {
		return response
	}

	c := resp.Choices[0]
	if cnt := c.Message.Content; cnt != "" {
		parts = append(parts, ai.NewTextPart(cnt))
	}

	if toolCalls := c.Message.ToolCalls; len(toolCalls) > 0 {
		for _, tc := range toolCalls {
			parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  tc.Function.Name,
				Ref:   tc.ID,
				Input: tc.Function.Arguments,
			}))
		}
	}

	response.Message = &ai.Message{
		Role:    ai.RoleModel,
		Content: parts,
	}
	response.FinishReason = ai.FinishReason(c.FinishReason)

	return response
}

func mapResponseFromText(mr *ai.ModelRequest, resp string) *ai.ModelResponse {
	return &ai.ModelResponse{
		Request: mr,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(resp)},
		},
	}
}

func mapMessagesToGenkit(messages []mistralclient.Message) []*ai.Message {
	m := make([]*ai.Message, len(messages))
	for i, msg := range messages {
		m[i] = newGenkitMessageFromMistral(msg)
	}
	return nil
}

func mapMessagesToMistral(messages []*ai.Message) []mistralclient.Message {
	m := make([]mistralclient.Message, len(messages))
	for i, msg := range messages {
		m[i] = newMistralMessageFromGenkit(msg)
	}
	return m
}

func mapRoleFromGenkit(role ai.Role) string {
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

func mapRoleFromMistral(role string) ai.Role {
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
