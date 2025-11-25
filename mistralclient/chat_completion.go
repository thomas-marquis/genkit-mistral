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

	// Model is the ID of the model to use. You can use the ListModels method to see all of your available models, or see https://docs.mistral.ai/getting-started/models overview for model descriptions.
	Model string `json:"model"`

	// Messages is(are) the prompt(s) to generate completions for, encoded as a list of dict with role and content.
	Messages []ChatMessage `json:"messages"`

	// MaxTokens is the maximum number of tokens to generate in the completion. The token count of your prompt plus max_tokens cannot exceed the model's context length.
	MaxTokens int `json:"max_tokens,omitempty"`

	// Temperature to use, we recommend between 0.0 and 0.7.
	//
	// Higher values like 0.7 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
	// We generally recommend altering this or top_p but not both.
	// The default value varies depending on the model you are targeting.
	// Call the /models endpoint to retrieve the appropriate value.
	Temperature float64 `json:"temperature,omitempty"`

	TopP int `json:"top_p,omitempty"`

	// ResponseFormat specifies the format that the model must output.
	//
	// By default it will use \{ "type": "text" \}.
	// Setting to \{ "type": "json_object" \} enables JSON mode, which guarantees the message the model generates is in JSON.
	// When using JSON mode you MUST also instruct the model to produce JSON yourself with a system or a user message.
	// Setting to \{ "type": "json_schema" \} enables JSON schema mode, which guarantees the message the model generates is in JSON and follows the schema you provide.
	ResponseFormat ResponseFormat `json:"response_format,omitempty"`

	Tools []ToolDefinition `json:"tools,omitempty"`

	// ToolChoice controls which (if any) tool is called by the model.
	//
	// "none" means the model will not call any tool and instead generates a message.
	//
	// "auto" means the model can pick between generating a message or calling one or more tools.
	//
	// "any" or required means the model must call one or more tools.
	//
	// Specifying a particular tool via \{"type": "function", "function": \{"name": "my_function"\}\} forces the model to call that tool.
	// You can marshal a ToolChoice object directly into this field.
	ToolChoice string `json:"tool_choice,omitempty"`

	// ParallelToolCalls defines whether to enable parallel function calling during tool use, when enabled the model can call multiple tools in parallel. Default to true when NewChatCompletionRequest is used.
	ParallelToolCalls bool `json:"parallel_tool_calls,omitempty"`

	// FrequencyPenalty penalizes the repetition of words based on their frequency in the generated text. A higher frequency penalty discourages the model from repeating words that have already appeared frequently in the output, promoting diversity and reducing repetition.
	FrequencyPenalty float64 `json:"frequency_penalty,omitempty"`

	// PresencePenalty determines how much the model penalizes the repetition of words or phrases. A higher presence penalty encourages the model to use a wider variety of words and phrases, making the output more diverse and creative.
	PresencePenalty float64 `json:"presence_penalty,omitempty"`

	// N is the number of completions to return for each request, input tokens are only billed once.
	N int `json:"n,omitempty"`

	// PromptMode allows toggling between the reasoning mode and no system prompt. When set to reasoning the system prompt for reasoning models will be used. Default to "".
	PromptMode string `json:"prompt_mode,omitempty"`

	// RandomSeed is the seed to use for random sampling. If set, different calls will generate deterministic results.
	RandomSeed int `json:"random_seed,omitempty"`

	// SafePrompt defines whether to inject a safety prompt before all conversations.
	SafePrompt bool `json:"safe_prompt,omitempty"`

	// Stop generation if this token is detected. Or if one of these tokens is detected when providing an array
	Stop []string `json:"stop,omitempty"`

	// Stream defines whether to stream back partial progress.
	//
	// If set, tokens will be sent as data-only server-side events as they become available, with the stream terminated by a data: [DONE] message.
	// Otherwise, the server will hold the request open until the timeout or until completion, with the response containing the full result as JSON.
	Stream bool `json:"stream,omitempty"`
}

func NewChatCompletionRequest(messages []ChatMessage, model string) ChatCompletionRequest {
	return ChatCompletionRequest{
		Messages:          messages,
		Model:             model,
		ParallelToolCalls: true,
		Stream:            false, // TODO: streaming is not supported yet...
	}
}

type MessageResponse struct {
	Message
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
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
