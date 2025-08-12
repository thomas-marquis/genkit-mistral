package mistral

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	mistralBaseAPIURL = "https://api.mistral.ai"
	defaultTimeout    = 5 * time.Second
)

type Client struct {
	apiKey      string
	baseURL     string
	rateLimiter RateLimiter
	httpClient  *http.Client
	verbose     bool
}

func NewClient(apiKey string, opts ...Option) *Client {
	return newClientWithConfig(apiKey, NewConfig(opts...))
}

func newClientWithConfig(apiKey string, cfg *Config) *Client {
	c := &Client{
		apiKey:      apiKey,
		baseURL:     mistralBaseAPIURL,
		rateLimiter: NewNoneRateLimiter(),
		httpClient: &http.Client{
			Timeout: defaultTimeout,
		},
		verbose: cfg.verbose,
	}

	if cfg.mistralAPIBaseURL != "" {
		c.baseURL = cfg.mistralAPIBaseURL
	}
	if cfg.rateLimiter != nil {
		c.rateLimiter = cfg.rateLimiter
	}
	if cfg.clientTimeout > 0 {
		c.httpClient.Timeout = cfg.clientTimeout
	}
	if cfg.apiKey != "" {
		c.apiKey = cfg.apiKey
	}

	return c
}

func (c *Client) ChatCompletion(ctx context.Context, messages []Message, model string, cfg *ModelConfig) (Message, error) {
	c.rateLimiter.Wait()

	url := fmt.Sprintf("%s/v1/chat/completions", c.baseURL)

	reqBody := ChatCompletionRequest{
		Messages:    messages,
		Model:       model,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxOutputTokens,
		TopP:        int(cfg.TopP),
		Stream:      false, // TODO: Implement streaming later
		Stop:        cfg.StopSequences,
	}

	jsonValue, err := json.Marshal(reqBody)
	if err != nil {
		return Message{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	response, err := sendRequest(ctx, c.httpClient, http.MethodPost, url, bytes.NewBuffer(jsonValue), c.apiKey)
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

func (c *Client) TextEmbedding(ctx context.Context, texts []string, model string) ([]EmbeddingVector, error) {
	c.rateLimiter.Wait()

	url := fmt.Sprintf("%s/v1/embeddings", c.baseURL)

	reqBody := EmbeddingRequest{
		Input: texts,
		Model: model,
	}

	jsonValue, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	response, err := sendRequest(ctx, c.httpClient, http.MethodPost, url, bytes.NewBuffer(jsonValue), c.apiKey)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	respBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if c.verbose {
		logger.Println("TextEmbedding called")
	}

	var resp EmbeddingResponse
	err = json.Unmarshal(respBody, &resp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response body: %w", err)
	}

	vectors := make([]EmbeddingVector, len(resp.Data))
	for i, data := range resp.Data {
		vectors[i] = data.Embedding
	}

	return vectors, nil
}

func sendRequest(ctx context.Context, client *http.Client, method, url string, body io.Reader, apiKey string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json; charset=utf-8")
	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make HTTP request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		errResponseBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP request failed with status %s and body '%s'", resp.Status, string(errResponseBody))
	}

	return resp, nil
}
