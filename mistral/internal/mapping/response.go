package mapping

import (
	"github.com/firebase/genkit/go/ai"
	"github.com/thomas-marquis/mistral-client/mistral"
)

func MapToGenkitResponse(mr *ai.ModelRequest, resp *mistral.ChatCompletionResponse) (*ai.ModelResponse, error) {
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
		return response, nil
	}

	choice := resp.Choices[0]
	msg := choice.Message
	if am := resp.AssistantMessage(); am != nil && len(am.ToolCalls) > 0 {
		for _, call := range am.ToolCalls {
			parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
				Input: call.Function.Arguments,
				Name:  call.Function.Name,
				Ref:   call.ID,
			}))
		}
	}
	cnt := msg.Content()
	if cs, ok := cnt.(mistral.ContentString); ok && cs.String() != "" {
		parts = append(parts, ai.NewTextPart(cnt.String()))
	} else {
		for _, chunk := range cnt.Chunks() {
			switch chunk.Type() {
			case mistral.ContentTypeText:
				parts = append(parts, ai.NewTextPart(chunk.(*mistral.TextChunk).Text))
			}
		}
	}

	response.Message = &ai.Message{
		Role:    ai.RoleModel,
		Content: parts,
	}

	response.FinishReason = mapFinishReason(choice.FinishReason)

	return response, nil
}

func mapFinishReason(reason mistral.FinishReason) ai.FinishReason {
	switch reason {
	case mistral.FinishReasonStop:
		return ai.FinishReasonStop
	case mistral.FinishReasonLength:
		return ai.FinishReasonLength
	case mistral.FinishReasonError:
		return ai.FinishReasonInterrupted
	case mistral.FinishReasonToolCalls:
		return ai.FinishReasonStop // I'm not sure about this one...
	case mistral.FinishReasonModelLength:
		return ai.FinishReasonLength
	default:
		return ai.FinishReasonUnknown
	}
}
