package mistralclient_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func Test_TextEmbedding_ShouldRetryOn429ThenSucceeds(t *testing.T) {
	var attempts int32

	// Given
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		if r.Method != http.MethodPost || r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		// First attempt gets 429, then succeed.
		if atomic.LoadInt32(&attempts) == 1 {
			http.Error(w, `{"error":"rate limited"}`, http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"emb-xyz",
			"object":"list",
			"model":"mistral-embed",
			"usage":{"prompt_tokens":0,"total_tokens":0},
			"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}]
		}`))
	}))
	defer srv.Close()

	cfg := &mistralclient.Config{
		Verbose:           false,
		MistralAPIBaseURL: srv.URL,
		RetryMaxRetries:   3,
		RetryWaitMin:      1 * time.Millisecond,
		RetryWaitMax:      5 * time.Millisecond,
		// Default retry status codes include 429
	}
	c := mistralclient.NewClientWithConfig("fake-api-key", cfg)
	ctx := context.Background()

	// When
	vecs, err := c.TextEmbedding(ctx, []string{"hello"}, "mistral-embed")

	// Then
	assert.NoError(t, err, "expected no error")
	assert.Equal(t, 1, len(vecs), "expected 1 embedding vector")
	assert.Equal(t, int32(2), atomic.LoadInt32(&attempts), "expected 2 attempts")
}
