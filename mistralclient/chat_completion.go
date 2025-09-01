package mistralclient

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type ChatCompletionRequest struct {
	Model          string         `json:"model"`
	Messages       []Message      `json:"messages"`
	Stream         bool           `json:"stream,omitempty"`
	MaxTokens      int            `json:"max_tokens,omitempty"`
	Temperature    float64        `json:"temperature,omitempty"`
	TopP           int            `json:"top_p,omitempty"`
	Stop           []string       `json:"stop,omitempty"`
	ResponseFormat ResponseFormat `json:"response_format,omitempty"`
}

type ChatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage UsageResponse `json:"usage"`
}

type chatCompletionOptions struct {
	ResponseFormat string
	JsonSchema     *JsonSchema
}

type ChatCompletionOption func(*chatCompletionOptions)

func WithResponseTextFormat() ChatCompletionOption {
	return func(opts *chatCompletionOptions) {
		opts.ResponseFormat = "text"
		opts.JsonSchema = nil
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

func (r ChatCompletionResponse) Text() string {
	if len(r.Choices) > 0 {
		return r.Choices[0].Message.Content
	}
	return ""
}

func (c *Client) ChatCompletion(
	ctx context.Context,
	messages []Message,
	model string,
	cfg *ModelConfig,
	opts ...ChatCompletionOption,
) (Message, error) {
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

	jsonValue, err := json.Marshal(reqBody)
	if err != nil {
		return Message{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	response, err := sendRequest(ctx, c, http.MethodPost, url, jsonValue)
	if err != nil {
		return Message{}, err
	}
	defer response.Body.Close()

	respBody, err := io.ReadAll(response.Body)
	if err != nil {
		return Message{}, fmt.Errorf("failed to read response body: %w", err)
	}
	if c.verbose {
		logger.Printf("ChatCompletion called")
	}

	var resp ChatCompletionResponse
	err = json.Unmarshal(respBody, &resp)
	if err != nil {
		return Message{}, fmt.Errorf("failed to unmarshal response body: %w", err)
	}

	return NewAssistantMessage(resp.Text()), nil
}
