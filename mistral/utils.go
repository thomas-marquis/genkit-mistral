package mistral

import (
	"log"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/thomas-marquis/genkit-mistral/mistral/internal/mapping"
)

var (
	ErrInvalidRole = mapping.ErrInvalidRole
	logger         = log.New(os.Stdout, "mistral-client: ", log.LstdFlags|log.Lshortfile)
)

// StringFromParts returns the content of a multi-parts message as a string.
// The multiple parts are concatenated with a newline character.
func StringFromParts(content []*ai.Part) string {
	msg, err := mapping.StringFromParts(content)
	if err != nil {
		logger.Printf("Failed to convert message content to string: %v\n", err)
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

//func newMistralMessageFromGenkit(msg *ai.Message) (mistral.ChatMessage, error) {
//	content := msg.Content
//
//	//role, err := mapRoleFromGenkit(msg.Role)
//	//if err != nil {
//	//	return nil, err
//	//}
//
//	var m mistral.ChatMessage
//	var chunks []mistral.ContentChunk
//
//	for i, part := range content {
//		switch part.Kind {
//		//case ai.PartText:
//		//	chunks = append(chunks, mistral.NewTextContent())
//		//
//		//	m.Content += part.Text
//		//	if i < len(content)-1 {
//		//		m.Content += "\n"
//		//	}
//		case ai.PartToolRequest:
//			funcArgs, err := json.Marshal(part.ToolRequest.Input)
//			if err != nil {
//				logger.Printf("Failed to marshal tool request: %v\n", err)
//			}
//			m.ToolCalls = append(m.ToolCalls, mistral.ToolCall{
//				Id:   part.ToolRequest.Ref,
//				Type: mistralclient.ToolTypeFunction,
//				Function: mistralclient.FunctionCall{
//					Name:      part.ToolRequest.Name,
//					Arguments: string(funcArgs),
//				},
//			})
//		//case ai.PartToolResponse:
//		//	bytes, err := json.Marshal(part.ToolResponse.Output)
//		//	if err != nil {
//		//		logger.Printf("Failed to marshal tool response: %v\n", err)
//		//	} else {
//		//		m.Content += string(bytes) + "\n"
//		//	}
//		//	m.ToolCallId = part.ToolResponse.Ref
//		//	m.FunctionName = part.ToolResponse.Name
//		default:
//			logger.Printf("Unexpected message content part kind: %v\n", part)
//		}
//	}
//
//	switch role {
//	case mistral.RoleUser:
//		m = &mistral.UserMessage{Role: role}
//
//	case mistral.RoleAssistant:
//		m = &mistral.AssistantMessage{Role: role}
//	case mistral.RoleSystem:
//		m = &mistral.SystemMessage{Role: role}
//	case mistral.RoleTool:
//		m = &mistral.ToolMessage{Role: role}
//	}
//
//	//for i, part := range content {
//	//	switch part.Kind {
//	//	case ai.PartText:
//	//		m.Content += part.Text
//	//		if i < len(content)-1 {
//	//			m.Content += "\n"
//	//		}
//	//	case ai.PartToolRequest:
//	//		funcArgs, err := json.Marshal(part.ToolRequest.Input)
//	//		if err != nil {
//	//			logger.Printf("Failed to marshal tool request: %v\n", err)
//	//		}
//	//		m.ToolCalls = append(m.ToolCalls, mistral.ToolCall{
//	//			Id:   part.ToolRequest.Ref,
//	//			Type: mistralclient.ToolTypeFunction,
//	//			Function: mistralclient.FunctionCall{
//	//				Name:      part.ToolRequest.Name,
//	//				Arguments: string(funcArgs),
//	//			},
//	//		})
//	//	case ai.PartToolResponse:
//	//		bytes, err := json.Marshal(part.ToolResponse.Output)
//	//		if err != nil {
//	//			logger.Printf("Failed to marshal tool response: %v\n", err)
//	//		} else {
//	//			m.Content += string(bytes) + "\n"
//	//		}
//	//		m.ToolCallId = part.ToolResponse.Ref
//	//		m.FunctionName = part.ToolResponse.Name
//	//	default:
//	//		logger.Printf("Unexpected message content part kind: %v\n", part)
//	//	}
//	//}
//
//	return m, nil
//}
//
//func mapResponse(mr *ai.ModelRequest, resp *mistral.ChatCompletionResponse) *ai.ModelResponse {
//	var parts []*ai.Part
//
//	response := &ai.ModelResponse{
//		Request: mr,
//		Usage: &ai.GenerationUsage{
//			InputTokens:  resp.Usage.PromptTokens,
//			OutputTokens: resp.Usage.CompletionTokens,
//			TotalTokens:  resp.Usage.TotalTokens,
//		},
//		LatencyMs: float64(resp.Latency.Milliseconds()),
//	}
//
//	if len(resp.Choices) == 0 {
//		return response
//	}
//
//	c := resp.Choices[0]
//	if cnt := c.Message.Content; cnt != "" {
//		parts = append(parts, ai.NewTextPart(cnt))
//	}
//
//	if toolCalls := c.Message.ToolCalls; len(toolCalls) > 0 {
//		for _, tc := range toolCalls {
//			parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
//				Name:  tc.Function.Name,
//				Ref:   tc.ID,
//				Input: tc.Function.Arguments,
//			}))
//		}
//	}
//
//	response.Message = &ai.Message{
//		Role:    ai.RoleModel,
//		Content: parts,
//	}
//	response.FinishReason = ai.FinishReason(c.FinishReason)
//
//	return response
//}

func mapResponseFromText(mr *ai.ModelRequest, resp string) *ai.ModelResponse {
	return &ai.ModelResponse{
		Request: mr,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(resp)},
		},
	}
}

//func mapMessagesToMistral(messages []*ai.Message) ([]mistral.ChatMessage, error) {
//	m := make([]mistral.ChatMessage, len(messages))
//	for i, msg := range messages {
//		mistralMsg, err := newMistralMessageFromGenkit(msg)
//		if err != nil {
//			return nil, err
//		}
//		m[i] = mistralMsg
//	}
//	return m, nil
//}
