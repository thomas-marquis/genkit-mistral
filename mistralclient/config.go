package mistralclient

import (
	"strings"
	"time"

	"github.com/thomas-marquis/genkit-mistral/internal"
)

type Config struct {
	ClientTimeout     time.Duration
	MistralAPIBaseURL string
	RateLimiter       RateLimiter
	ApiKey            string
	Verbose           bool

	// Retry configuration
	RetryMaxRetries  int
	RetryWaitMin     time.Duration
	RetryWaitMax     time.Duration
	RetryStatusCodes []int
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
		cfg.ClientTimeout = timeout
	}
}

func WithBaseAPIURL(baseURL string) Option {
	return func(cfg *Config) {
		cfg.MistralAPIBaseURL = strings.TrimSuffix(baseURL, "/")
	}
}

func WithRateLimiter(rateLimiter RateLimiter) Option {
	return func(cfg *Config) {
		cfg.RateLimiter = rateLimiter
	}
}

func WithAPIKey(apiKey string) Option {
	return func(cfg *Config) {
		cfg.ApiKey = apiKey
	}
}

func WithVerbose(verbose bool) Option {
	return func(cfg *Config) {
		cfg.Verbose = verbose
	}
}

// WithRetry configures automatic retries for HTTP requests.
// maxRetries is the number of retries after the first attempt.
// waitMin and waitMax control the exponential backoff bounds (set to 0 for default).
func WithRetry(maxRetries int, waitMin, waitMax time.Duration) Option {
	if waitMin == 0 {
		waitMin = 200 * time.Millisecond
	}
	if waitMax == 0 {
		waitMax = 2 * time.Second
	}

	return func(cfg *Config) {
		cfg.RetryMaxRetries = maxRetries
		cfg.RetryWaitMin = waitMin
		cfg.RetryWaitMax = waitMax
	}
}

// WithRetryStatusCodes overrides the list of HTTP status codes that should trigger a retry.
// If not specified, defaults are: 429, 500, 502, 503, 504.
func WithRetryStatusCodes(codes ...int) Option {
	return func(cfg *Config) {
		cfg.RetryStatusCodes = append([]int(nil), codes...)
	}
}

type ModelConfig struct {
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
	Temperature     float64  `json:"temperature,omitempty"`
	TopK            int      `json:"topK,omitempty"`
	TopP            float64  `json:"topP,omitempty"`
	Version         string   `json:"version,omitempty"`
}

func NewModelConfigFromRaw(r map[string]any) *ModelConfig {
	return &ModelConfig{
		MaxOutputTokens: internal.GetOrZero[int](r, "maxOutputTokens"),
		StopSequences:   internal.GetSliceOrNil[string](r, "stopSequences"),
		Temperature:     internal.GetOrZero[float64](r, "temperature"),
		TopK:            internal.GetOrZero[int](r, "topK"),
		TopP:            internal.GetOrZero[float64](r, "topP"),
		Version:         internal.GetOrZero[string](r, "version"),
	}
}
