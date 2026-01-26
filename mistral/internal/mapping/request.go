package mapping

import (
	"errors"

	"github.com/firebase/genkit/go/ai"
	"github.com/thomas-marquis/mistral-client/mistral"
)

var (
	ErrNoModelProvided = errors.New("model name is empty")
	ErrNoMessages      = errors.New("message list is empty")
)

func MapRequestToMistral(model string, mr *ai.ModelRequest, cfg *mistral.CompletionConfig) (*mistral.ChatCompletionRequest, error) {
	if model == "" {
		return nil, ErrNoModelProvided
	}
	if len(mr.Messages) == 0 {
		return nil, ErrNoMessages
	}

	messages := make([]mistral.ChatMessage, 0, len(mr.Messages))
	for _, msg := range mr.Messages {
		m, err := MapToMistralMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, m...)
	}

	req := &mistral.ChatCompletionRequest{
		Messages: messages,
		Model:    model,
	}

	if cfg != nil {
		req.CompletionConfig = *cfg
	}

	req.Stream = false

	if nbTools := len(mr.Tools); nbTools > 0 {
		tools := make([]mistral.Tool, 0, nbTools)
		for _, tool := range mr.Tools {
			tools = append(tools, mistral.NewTool(tool.Name, tool.Description,
				mistral.NewPropertyDefinition(tool.InputSchema)))
		}
		mistral.WithTools(tools)(req)
		req.ToolChoice = mapToMistralToolChoice(mr.ToolChoice)
	}

	if mr.Output != nil && mr.Output.Constrained && mr.Output.Format == "json" {
		mistral.WithResponseJsonSchema(mistral.NewPropertyDefinition(mr.Output.Schema))(req)
	} else {
		mistral.WithResponseTextFormat()(req)
	}

	return req, nil
}

func mapToMistralToolChoice(choice ai.ToolChoice) mistral.ToolChoiceType {
	switch choice {
	case ai.ToolChoiceAuto:
		return mistral.ToolChoiceAuto
	case ai.ToolChoiceRequired:
		return mistral.ToolChoiceAny
	case ai.ToolChoiceNone:
		return mistral.ToolChoiceNone
	default:
		return mistral.ToolChoiceAuto
	}
}
