package mistralclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
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

	// TopP is the nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
	//
	// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
	// We generally recommend altering this or temperature but not both.
	// Default to 1.0
	TopP float64 `json:"top_p,omitempty"`

	// ResponseFormat specifies the format that the model must output.
	//
	// By default, it will use \{ "type": "text" \}.
	// Setting to \{ "type": "json_object" \} enables JSON mode, which guarantees the message the model generates is in JSON.
	// When using JSON mode you MUST also instruct the model to produce JSON yourself with a system or a user message.
	// Setting to \{ "type": "json_schema" \} enables JSON schema mode, which guarantees the message the model generates is in JSON and follows the schema you provide.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	Tools []Tool `json:"tools,omitempty"`

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

type ChatCompletionResponse struct {
	Choices []ChatCompletionChoice `json:"choices"`
	Created time.Time              `json:"created"`
	Id      string                 `json:"id"`
	Model   string                 `json:"model"`
	Object  string                 `json:"object"`
	Usage   UsageInfo              `json:"usage"`

	Latency time.Duration
}

var _ json.Unmarshaler = (*ChatCompletionResponse)(nil)

func (r *ChatCompletionResponse) UnmarshalJSON(data []byte) error {
	type Alias ChatCompletionResponse
	aux := &struct {
		*Alias
		Created int64 `json:"created"`
	}{
		Alias: (*Alias)(r),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	r.Created = time.Unix(aux.Created, 0).UTC()
	return nil
}

type ChatCompletionChoice struct {
	FinishReason FinishReason      `json:"finish_reason"`
	Index        int               `json:"index"`
	Message      *AssistantMessage `json:"message"`
}

type FinishReason string

const (
	FinishReasonLength      FinishReason = "length"
	FinishReasonStop        FinishReason = "stop"
	FinishReasonModelLength FinishReason = "model_length"
	FinishReasonError       FinishReason = "error"
	FinishReasonToolCalls   FinishReason = "tool_calls"
)

type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJsonObject ResponseFormatType = "json_object"
	ResponseFormatJsonSchema ResponseFormatType = "json_schema"
)

type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JsonSchema *JsonSchema        `json:"json_schema,omitempty"`
}

type chatCompletionOptions struct {
	ResponseFormat ResponseFormatType
	JsonSchema     *JsonSchema
	Tools          []Tool
	ToolChoice     ToolChoiceType
}

type ChatCompletionOption func(*chatCompletionOptions)

func WithResponseTextFormat() ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ResponseFormat = ResponseFormatText
		opts.JsonSchema = nil
	}
}

func WithResponseJsonSchema(schema any) ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ResponseFormat = ResponseFormatJsonSchema
		opts.JsonSchema = &JsonSchema{
			Name:   "responseJsonSchema",
			Schema: schema,
			Strict: true,
		}
	}
}

func WithResponseJsonObjectFormat() ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ResponseFormat = ResponseFormatJsonObject
		opts.JsonSchema = nil
	}
}

func WithTools(tools []Tool) ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.Tools = tools
		opts.ToolChoice = ToolChoiceAuto
	}
}

func WithToolChoice(toolChoice ToolChoiceType) ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ToolChoice = toolChoice
	}
}

func (c *clientImpl) ChatCompletion(
	ctx context.Context,
	messages []ChatMessage,
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

	reqBody := NewChatCompletionRequest(messages, model)
	reqBody.MaxTokens = cfg.MaxOutputTokens
	reqBody.Temperature = cfg.Temperature
	reqBody.TopP = cfg.TopP
	reqBody.Stop = cfg.StopSequences

	switch opt.ResponseFormat {
	case ResponseFormatJsonSchema:
		reqBody.ResponseFormat = &ResponseFormat{Type: opt.ResponseFormat, JsonSchema: opt.JsonSchema}
	case ResponseFormatJsonObject, ResponseFormatText:
		reqBody.ResponseFormat = &ResponseFormat{Type: opt.ResponseFormat, JsonSchema: nil}
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

	if c.verbose {
		logger.Printf("ChatCompletion called")
	}

	var resp ChatCompletionResponse
	if err := unmarshallBody(response, &resp); err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("failed to unmarshal response body: %w", err)
	}
	resp.Latency = lat

	return resp, nil
}
