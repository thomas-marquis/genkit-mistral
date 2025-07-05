package mistral

import (
	"strings"
	"time"
)

type Config struct {
	clientTimeout     time.Duration
	mistralAPIBaseURL string
	rateLimiter       RateLimiter
	apiKey            string
}

func NewConfig(opts ...Option) *Config {
	cfg := &Config{}

	for _, opt := range opts {
		opt(cfg)
	}

	return cfg
}

type Option func(*Config)

func WithClientTimeout(timeout time.Duration) Option {
	return func(cfg *Config) {
		cfg.clientTimeout = timeout
	}
}

func WithBaseAPIURL(baseURL string) Option {
	return func(cfg *Config) {
		cfg.mistralAPIBaseURL = strings.TrimSuffix(baseURL, "/")
	}
}

func WithRateLimiter(rateLimiter RateLimiter) Option {
	return func(cfg *Config) {
		cfg.rateLimiter = rateLimiter
	}
}

func WithAPIKey(apiKey string) Option {
	return func(cfg *Config) {
		cfg.apiKey = apiKey
	}
}
