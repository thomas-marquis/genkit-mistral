package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	mistralBaseAPIURL = "https://api.mistral.com"
)

type Client struct {
	apiKey       string
	modelName    string
	modelVersion string
	baseURL      string
	rateLimiter  RateLimiter
	httpClient   *http.Client
}

func NewClient(apiKey string, modelName, modelVersion string, opts ...Option) *Client {
	return newClientWithConfig(apiKey, modelName, modelVersion, NewConfig(opts...))
}

func newClientWithConfig(apiKey string, modelName, modelVersion string, cfg *Config) *Client {
	timeout := 5 * time.Second

	c := &Client{
		apiKey:       apiKey,
		modelName:    modelName,
		modelVersion: modelVersion,
		baseURL:      mistralBaseAPIURL,
		rateLimiter:  NewNoneRateLimiter(),
		httpClient: &http.Client{
			Timeout: timeout,
		},
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

func (c *Client) ChatCompletion(messages []Message) (Message, error) {
	c.rateLimiter.Wait()

	url := fmt.Sprintf("%s/v1/chat/completions", c.baseURL)

	reqBody := ChatCompletionRequest{
		Messages: messages,
		Model:    c.modelName + "-" + c.modelVersion,
	}

	jsonValue, err := json.Marshal(reqBody)
	if err != nil {
		return Message{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonValue))
	if err != nil {
		return Message{}, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Accept", "application/json; charset=utf-8")
	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	response, err := c.httpClient.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("failed to make HTTP request: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		errResponseBody, _ := io.ReadAll(response.Body)
		return Message{}, fmt.Errorf("HTTP request failed with status %s and body '%s'", response.Status, string(errResponseBody))
	}

	respBody, err := io.ReadAll(response.Body)
	if err != nil {
		return Message{}, fmt.Errorf("failed to read response body: %w", err)
	}

	var resp ChatCompletionResponse
	err = json.Unmarshal(respBody, &resp)
	if err != nil {
		return Message{}, fmt.Errorf("failed to unmarshal response body: %w", err)
	}

	return NewAssistantMessage(resp.Text()), nil
}
