package mistralclient

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
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

	retryMaxRetries  int
	retryWaitMin     time.Duration
	retryWaitMax     time.Duration
	retryStatusCodes map[int]struct{}
}

func NewClient(apiKey string, opts ...Option) *Client {
	return NewClientWithConfig(apiKey, NewConfig(opts...))
}

func NewClientWithConfig(apiKey string, cfg *Config) *Client {
	c := &Client{
		apiKey:      apiKey,
		baseURL:     mistralBaseAPIURL,
		rateLimiter: NewNoneRateLimiter(),
		httpClient: &http.Client{
			Timeout:   defaultTimeout,
			Transport: cfg.Transport,
		},
		verbose:          cfg.Verbose,
		retryMaxRetries:  cfg.RetryMaxRetries,
		retryWaitMin:     cfg.RetryWaitMin,
		retryWaitMax:     cfg.RetryWaitMax,
		retryStatusCodes: make(map[int]struct{}),
	}

	if cfg.MistralAPIBaseURL != "" {
		c.baseURL = cfg.MistralAPIBaseURL
	}
	if cfg.RateLimiter != nil {
		c.rateLimiter = cfg.RateLimiter
	}
	if cfg.ClientTimeout > 0 {
		c.httpClient.Timeout = cfg.ClientTimeout
	}
	if cfg.ApiKey != "" {
		c.apiKey = cfg.ApiKey
	}

	// status codes to retry: configured or sensible defaults
	if len(cfg.RetryStatusCodes) > 0 {
		for _, code := range cfg.RetryStatusCodes {
			c.retryStatusCodes[code] = struct{}{}
		}
	} else {
		for _, code := range []int{
			http.StatusTooManyRequests, // 429
			http.StatusInternalServerError,
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout,
		} {
			c.retryStatusCodes[code] = struct{}{}
		}
	}

	return c
}

// isRetryableErr returns true if the error is retryable.
//
// Retriable errors:
//   - [net.Error] with Temporary() == true
//   - [context.DeadlineExceeded]
//   - unexpected EOFs ([io.EOF]) and similar transient I/O issues.
//
// Errors that are not retriable:
//   - [context.Canceled]
//   - any other errors
func isRetryableErr(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return true
		}
		// Temporary is deprecated but still implemented by some errors.
		if te, ok := any(netErr).(interface{ Temporary() bool }); ok && te.Temporary() {
			return true
		}
	}

	return errors.Is(err, io.EOF)
}

func (c *Client) nextBackoff(attempt int) time.Duration {
	if attempt <= 0 {
		return c.retryWaitMin
	}
	wait := c.retryWaitMin * time.Duration(1<<uint(attempt))
	if wait > c.retryWaitMax {
		wait = c.retryWaitMax
	}
	// Full jitter in [0, wait]
	jitter := time.Duration(rand.Int63n(int64(wait)))
	return jitter
}

func sendRequest(ctx context.Context, c *Client, method, url string, body []byte) (*http.Response, error) {
	// attempt = 0 is the first try; we perform up to (1 + retryMaxRetries) attempts total.
	for attempt := 0; attempt <= c.retryMaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}

		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Accept", "application/json; charset=utf-8")
		req.Header.Set("Content-Type", "application/json; charset=utf-8")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			if attempt < c.retryMaxRetries && isRetryableErr(err) {
				wait := c.nextBackoff(attempt)
				if c.verbose {
					logger.Printf("HTTP request error, retrying attempt %d/%d after %v: %v",
						attempt+1, c.retryMaxRetries, wait, err)
				}
				select {
				case <-time.After(wait):
					continue
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}
			return nil, fmt.Errorf("failed to make HTTP request: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			if attempt < c.retryMaxRetries {
				if _, ok := c.retryStatusCodes[resp.StatusCode]; ok {
					// Drain and close the body before retrying
					io.Copy(io.Discard, resp.Body)
					resp.Body.Close()
					wait := c.nextBackoff(attempt)
					if c.verbose {
						logger.Printf("HTTP status %s, retrying attempt %d/%d after %v",
							resp.Status, attempt+1, c.retryMaxRetries, wait)
					}
					select {
					case <-time.After(wait):
						continue
					case <-ctx.Done():
						return nil, ctx.Err()
					}
				}
			}
			errResponseBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			return nil, fmt.Errorf("HTTP request failed with status %s and body '%s'",
				resp.Status, string(errResponseBody))
		}

		return resp, nil
	}

	return nil, fmt.Errorf("exhausted retries without a successful response")
}
