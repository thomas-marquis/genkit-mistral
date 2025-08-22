package mistral

import (
	"context"
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
