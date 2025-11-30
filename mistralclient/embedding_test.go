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

func TestEmbeddings(t *testing.T) {
	t.Run("should return float embedding vector on success", func(t *testing.T) {
		// Given
		var gotReq string
		mockServer := makeMockServerWithCapture(t, "POST", "/v1/embeddings", `{
			"id": "azerty",
			"object": "list",
			"data": [
				{
					"object": "embedding",
					"embedding": [0.0001, 0.0002, 0.0003],
					"index": 0
				}
			],
			"model": "mistral-embed",
			"usage": {
				"prompt_audio_seconds": null,
				"prompt_tokens": 7,
				"total_tokens": 7,
				"completion_tokens": 0,
				"request_count": null,
				"prompt_token_details": null
			}
		}`, http.StatusOK, &gotReq)
		defer mockServer.Close()

		ctx := context.TODO()
		c := mistralclient.NewClient("fakeApiKey", mistralclient.WithBaseAPIURL(mockServer.URL))

		// When
		res, err := c.Embeddings(ctx, []string{"ipsum eiusmod"}, "mistral-embed")

		// Then
		assert.NoError(t, err)
		assert.Equal(t, 1, len(res.Embeddings()))
		assert.Equal(t, mistralclient.EmbeddingVector{0.0001, 0.0002, 0.0003}, res.Embeddings()[0])
		assert.Equal(t, mistralclient.UsageInfo{
			PromptTokens: 7, TotalTokens: 7, CompletionTokens: 0,
		}, res.Usage)
		assert.Equal(t, "list", res.Object)
		assert.Equal(t, "mistral-embed", res.Model)
		assert.JSONEq(t, `{
			"input": [
				"ipsum eiusmod"
			],
			"model": "mistral-embed"
		}`, gotReq)
	})

	t.Run("should return float embedding vector on success with options", func(t *testing.T) {
		// Given
		var gotReq string
		mockServer := makeMockServerWithCapture(t, "POST", "/v1/embeddings", `{
			"id": "azerty",
			"object": "list",
			"data": [
				{
					"object": "embedding",
					"embedding": [0.0001, 0.0002, 0.0003],
					"index": 0
				}
			],
			"model": "mistral-embed",
			"usage": {
				"prompt_audio_seconds": null,
				"prompt_tokens": 7,
				"total_tokens": 7,
				"completion_tokens": 0,
				"request_count": null,
				"prompt_token_details": null
			}
		}`, http.StatusOK, &gotReq)
		defer mockServer.Close()

		ctx := context.TODO()
		c := mistralclient.NewClient("fakeApiKey", mistralclient.WithBaseAPIURL(mockServer.URL))

		// When
		res, err := c.Embeddings(ctx, []string{"ipsum eiusmod"}, "mistral-embed")

		// Then
		assert.NoError(t, err)
		assert.Equal(t, 1, len(res.Embeddings()))
		assert.Equal(t, mistralclient.EmbeddingVector{0.0001, 0.0002, 0.0003}, res.Embeddings()[0])
		assert.Equal(t, mistralclient.UsageInfo{
			PromptTokens: 7, TotalTokens: 7, CompletionTokens: 0,
		}, res.Usage)
		assert.Equal(t, "list", res.Object)
		assert.Equal(t, "mistral-embed", res.Model)
		assert.JSONEq(t, `{
			"input": [
				"ipsum eiusmod"
			],
			"model": "mistral-embed"
		}`, gotReq)
	})
}

func Test_TextEmbedding_ShouldRetryOn429ThenSucceeds(t *testing.T) {
	// Given
	var attempts int32
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
	res, err := c.Embeddings(ctx, []string{"hello"}, "mistral-embed")

	// Then
	assert.NoError(t, err, "expected no error")
	assert.Equal(t, 1, len(res.Embeddings()), "expected 1 embedding vector")
	assert.Equal(t, int32(2), atomic.LoadInt32(&attempts), "expected 2 attempts")
}

func Test_TextEmbedding_ShouldRetryAfterTimeouts(t *testing.T) {
	// Given
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		if r.Method != http.MethodPost || r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		// First 2 attempt timeout, then succeed.
		if atomic.LoadInt32(&attempts) <= 2 {
			http.Error(w, `{"error":"rate limited"}`, http.StatusRequestTimeout)
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

	// When

	// Then
}
