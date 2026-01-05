package mapping

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/thomas-marquis/mistral-client/mistral"
)

var (
	ErrInvalidRole = errors.New("invalid role")
	logger         = log.New(os.Stdout, "mistral-client: ", log.LstdFlags|log.Lshortfile)
)

func StringFromParts(content []*ai.Part) (string, error) {
	var msg string
	for i, part := range content {
		if part.Kind == ai.PartText {
			msg += part.Text
			if i < len(content)-1 {
				msg += "\n"
			}
		} else {
			return "", fmt.Errorf("Unexpected message content part: %v\n", part)
		}
	}

	return msg, nil
}

func isContentOnlyText(content []*ai.Part) bool {
	for _, part := range content {
		if part.Kind != ai.PartText {
			return false
		}
	}
	return true
}

func mapMessageContent(parts []*ai.Part) (mistral.ContentChunks, error) {
	content := make(mistral.ContentChunks, 0, len(parts))

	for _, part := range parts {
		switch part.Kind {
		case ai.PartText:
			content = append(content, mistral.NewTextChunk(part.Text))
		case ai.PartMedia:
			if part.IsImage() {
				content = append(content, mistral.NewImageUrlChunk(part.Text))
			} else if part.IsAudio() {
				content = append(content, mistral.NewAudioChunk(part.Text))
			} else {
				logger.Printf("Unsupported media type: %s\n", part.ContentType)
			}
		}
	}

	return content, nil
}

func MapToMistralMessage(msg *ai.Message) ([]mistral.ChatMessage, error) {
	role, err := MapToMistralRole(msg.Role)
	if err != nil {
		return nil, err
	}

	var m []mistral.ChatMessage
	switch role {
	case mistral.RoleUser:
		if isContentOnlyText(msg.Content) {
			strContent, err := StringFromParts(msg.Content)
			if err != nil {
				return nil, err
			}
			m = append(m, mistral.NewUserMessageFromString(strContent))
		} else {
			content, err := mapMessageContent(msg.Content)
			if err != nil {
				return nil, err
			}
			m = append(m, mistral.NewUserMessage(content))
		}

	case mistral.RoleAssistant:
		var assMsg *mistral.AssistantMessage
		if isContentOnlyText(msg.Content) {
			strContent, err := StringFromParts(msg.Content)
			if err != nil {
				return nil, err
			}
			assMsg = mistral.NewAssistantMessageFromString(strContent)
		} else {
			content, err := mapMessageContent(msg.Content)
			if err != nil {
				return nil, err
			}
			assMsg = mistral.NewAssistantMessage(content)
		}
		for i, part := range msg.Content {
			if part.Kind == ai.PartToolRequest {
				assMsg.ToolCalls = append(assMsg.ToolCalls,
					mistral.NewToolCall(part.ToolRequest.Ref, i, part.ToolRequest.Name, part.ToolRequest.Input))
			}
		}
		m = append(m, assMsg)
	case mistral.RoleSystem:
		content, err := StringFromParts(msg.Content)
		if err != nil {
			return nil, err
		}
		m = append(m, mistral.NewSystemMessageFromString(content))

	case mistral.RoleTool:
		for _, part := range msg.Content {
			if part.Kind == ai.PartToolResponse {
				outputBytes, err := json.Marshal(part.ToolResponse.Output)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal tool response output: %w", err)
				}
				m = append(m, mistral.NewToolMessage(
					part.ToolResponse.Name, part.ToolResponse.Ref, mistral.ContentString(outputBytes)))
			}
		}
	}

	return m, nil
}

func MapToMistralRole(role ai.Role) (mistral.Role, error) {
	switch role {
	case ai.RoleUser:
		return mistral.RoleUser, nil
	case ai.RoleModel:
		return mistral.RoleAssistant, nil
	case ai.RoleSystem:
		return mistral.RoleSystem, nil
	case ai.RoleTool:
		return mistral.RoleTool, nil
	default:
		return "", ErrInvalidRole
	}
}
