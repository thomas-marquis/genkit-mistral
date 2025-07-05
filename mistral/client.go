package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type ClientOption func(*Client)

func WithBaseURL(baseURL string) ClientOption {
	return func(c *Client) {
		c.baseURL = strings.TrimSuffix(baseURL, "/")
	}
}

func WithRateLimiter(limiter RateLimiter) ClientOption {
	return func(c *Client) {
		c.rateLimiter = limiter
	}
}

type Client struct {
	apiKey       string
	timeout      time.Duration
	modelName    string
	modelVersion string
	baseURL      string
	rateLimiter  RateLimiter
}

func NewClient(apiKey string, modelName, modelVersion string, opts ...ClientOption) *Client {
	c := &Client{
		apiKey:       apiKey,
		timeout:      5 * time.Second,
		modelName:    modelName,
		modelVersion: modelVersion,
		baseURL:      "https://api.mistral.com",
		rateLimiter:  NewNoneRateLimiter(),
	}

	for _, opt := range opts {
		opt(c)
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

	client := &http.Client{
		Timeout: c.timeout,
	}

	response, err := client.Do(req)
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
