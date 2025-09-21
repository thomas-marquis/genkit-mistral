package mistralclient

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type ChatCompletionRequest struct {
	Model             string           `json:"model"`
	Messages          []Message        `json:"messages"`
	Stream            bool             `json:"stream,omitempty"`
	MaxTokens         int              `json:"max_tokens,omitempty"`
	Temperature       float64          `json:"temperature,omitempty"`
	TopP              int              `json:"top_p,omitempty"`
	Stop              []string         `json:"stop,omitempty"`
	ResponseFormat    ResponseFormat   `json:"response_format,omitempty"`
	Tools             []ToolDefinition `json:"tools,omitempty"`
	ToolChoice        string           `json:"tool_choice,omitempty"`
	ParallelToolCalls bool             `json:"parallel_tool_calls,omitempty"`
}

type MessageResponse struct {
	Message
	ToolCalls []ToolCallRequest `json:"tool_calls,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int             `json:"index"`
	Message      MessageResponse `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   UsageResponse          `json:"usage"`
	Latency time.Duration          `json:"latency_ms,omitempty"`
}

type chatCompletionOptions struct {
	ResponseFormat string
	JsonSchema     *JsonSchema
	Tools          []ToolDefinition
	ToolChoice     ToolChoice
}

type ChatCompletionOption func(*chatCompletionOptions)

func WithResponseTextFormat() ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ResponseFormat = "text"
		opts.JsonSchema = nil
	}
}

func WithTools(tools []ToolDefinition) ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.Tools = tools
		opts.ToolChoice = ToolChoiceAuto
	}
}

// ToolChoice defines how the model should use tools or not.
type ToolChoice string

func (tc ToolChoice) String() string {
	return string(tc)
}

// NewToolChoice creates a new ToolChoice from a string.
func NewToolChoice(choice string) ToolChoice {
	switch strings.ToLower(choice) {
	case ToolChoiceAuto.String():
		return ToolChoiceAuto
	case ToolChoiceAny.String():
		return ToolChoiceAny
	case ToolChoiceNone.String():
		return ToolChoiceNone
	case "":
		return ""
	default:
		logger.Printf("Invalid tool choice: %s. Using empty value.", choice)
		return ""
	}
}

const (
	// ToolChoiceAuto is the default mode. Model decides if it uses the tool or not.
	ToolChoiceAuto ToolChoice = "auto"
	// ToolChoiceAny forces the model to use a tool.
	ToolChoiceAny ToolChoice = "any"
	// ToolChoiceNone prevent model to use a tool.
	ToolChoiceNone ToolChoice = "none"
)

func WithToolChoice(toolChoice ToolChoice) ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ToolChoice = toolChoice
	}
}

func WithResponseJsonSchema(schema any) ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ResponseFormat = "json_schema"
		opts.JsonSchema = &JsonSchema{
			Name:   "responseJsonSchema",
			Schema: schema,
			Strict: true,
		}
	}
}

func (c *clientImpl) ChatCompletion(
	ctx context.Context,
	messages []Message,
	model string,
	cfg *ModelConfig,
	opts ...ChatCompletionOption,
) (ChatCompletionResponse, error) {
	c.rateLimiter.Wait()

	opt := &chatCompletionOptions{}
	for _, optFn := range opts {
		optFn(opt)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", c.baseURL)

	reqBody := ChatCompletionRequest{
		Messages:    messages,
		Model:       model,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxOutputTokens,
		TopP:        int(cfg.TopP),
		Stream:      false, // TODO: Implement streaming later
		Stop:        cfg.StopSequences,
		ResponseFormat: ResponseFormat{
			Type:       opt.ResponseFormat,
			JsonSchema: opt.JsonSchema,
		},
	}

	if len(opt.Tools) > 0 {
		reqBody.Tools = opt.Tools
		reqBody.ToolChoice = opt.ToolChoice.String()
	}

	jsonValue, err := json.Marshal(reqBody)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	response, lat, err := sendRequest(ctx, c, http.MethodPost, url, jsonValue)
	if err != nil {
		return ChatCompletionResponse{}, err
	}
	defer response.Body.Close()

	respBody, err := io.ReadAll(response.Body)
	if err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("failed to read response body: %w", err)
	}
	if c.verbose {
		logger.Printf("ChatCompletion called")
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("failed to unmarshal response body: %w", err)
	}
	resp.Latency = lat

	return resp, nil
}
