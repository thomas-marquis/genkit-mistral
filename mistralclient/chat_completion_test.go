package mistralclient_test

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

// timeoutNetError implements net.Error with Timeout() = true
type timeoutNetError struct{}

func (timeoutNetError) Error() string   { return "timeout" }
func (timeoutNetError) Timeout() bool   { return true }
func (timeoutNetError) Temporary() bool { return true } // for legacy checks

// flakyRoundTripper fails with a timeout once, then returns a successful response.
type flakyRoundTripper struct {
	failuresLeft int32
	successBody  []byte
}

func (f *flakyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if atomic.AddInt32(&f.failuresLeft, -1) >= 0 {
		return nil, timeoutNetError{}
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     make(http.Header),
		Body:       io.NopCloser(bytes.NewReader(f.successBody)),
		Request:    req,
	}
	resp.Header.Set("Content-Type", "application/json")
	return resp, nil
}

func makeMockServer(t *testing.T, method, path, jsonResponse string, responseCode int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == method && r.URL.Path == path {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(responseCode)
			_, _ = w.Write([]byte(jsonResponse))
		} else {
			http.NotFound(w, r)
		}
	}))
}

func Test_ChatCompletion_ShouldReturnMessageWhenSucceed(t *testing.T) {
	mockServer := makeMockServer(t, "POST", "/v1/chat/completions", `
				{
					"choices": [
						{
							"message": {
								"role": "assistant",
								"content": "Hello, how can I assist you?"
							}
						}
					]
				}`, http.StatusOK)
	defer mockServer.Close()

	fakeApiKey := "fake-api-key"

	t.Run("Successful ChatCompletion", func(t *testing.T) {
		// Given
		ctx := context.TODO()
		c := mistralclient.NewClient(fakeApiKey, mistralclient.WithBaseAPIURL(mockServer.URL))
		inputMsgs := []mistralclient.Message{
			mistralclient.NewSystemMessage("You are a helpful assistant."),
			mistralclient.NewHumanMessage("Hello!"),
		}

		// When
		res, err := c.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistralclient.ModelConfig{})

		// Then
		expected := mistralclient.NewAssistantMessage("Hello, how can I assist you?")
		assert.NoError(t, err)
		assert.Equal(t, expected, res)
	})
}

func Test_ChatCompletion_ShouldRetryOn5xxThenSucceeds(t *testing.T) {
	var attempts int32

	// Given
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		// First two attempts fail with 500, then succeed.
		if atomic.LoadInt32(&attempts) <= 2 {
			http.Error(w, `{"error":"temporary"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"choices": [
				{ "message": { "role": "assistant", "content": "Hello after retries" } }
			]
		}`))
	}))
	defer srv.Close()

	cfg := &mistralclient.Config{
		Verbose:           false,
		MistralAPIBaseURL: srv.URL,
		RetryMaxRetries:   3,
		RetryWaitMin:      1 * time.Millisecond,
		RetryWaitMax:      5 * time.Millisecond,
		RetryStatusCodes:  nil, // use defaults (includes 500)
	}
	c := mistralclient.NewClientWithConfig("fake-api-key", cfg)
	ctx := context.Background()
	inputMsgs := []mistralclient.Message{
		mistralclient.NewHumanMessage("Hi!"),
	}

	// When
	msg, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})

	// Then
	assert.NoError(t, err, "expected no error")
	expected := mistralclient.NewAssistantMessage("Hello after retries")
	assert.Equal(t, expected, msg, "expected message")
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts), "expected 3 attempts")
}

func Test_ChatCompletion_ShouldNotRetryOn400AndFailsImmediately(t *testing.T) {
	var attempts int32

	// Given
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest) // 400 should NOT be retried
	}))
	defer srv.Close()

	cfg := &mistralclient.Config{
		Verbose:           false,
		MistralAPIBaseURL: srv.URL,
		RetryMaxRetries:   5, // even with high max, should not retry on 400
		RetryWaitMin:      1 * time.Millisecond,
		RetryWaitMax:      2 * time.Millisecond,
	}
	c := mistralclient.NewClientWithConfig("fake-api-key", cfg)
	ctx := context.Background()
	inputMsgs := []mistralclient.Message{mistralclient.NewHumanMessage("Hi!")}

	// When
	_, err := c.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistralclient.ModelConfig{})

	// Then
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if got := atomic.LoadInt32(&attempts); got != 1 {
		t.Fatalf("expected exactly 1 attempt on 400, got %d", got)
	}
}

func Test_ChatCompletion_ShouldRetryOnTimeoutErrorThenSucceeds(t *testing.T) {
	// Given
	successJSON := []byte(`{"choices":[{"message":{"role":"assistant","content":"OK after timeout"}}]}`)
	cfg := &mistralclient.Config{
		Verbose:         false,
		RetryMaxRetries: 3,
		RetryWaitMin:    1 * time.Millisecond,
		RetryWaitMax:    5 * time.Millisecond,
		// Base URL irrelevant; transport short-circuits requests
		MistralAPIBaseURL: "http://invalid.local",

		// Set a flaky transport: fails once with timeout, then returns a success payload for ChatCompletion.
		Transport: &flakyRoundTripper{
			failuresLeft: 1,
			successBody:  successJSON,
		},
		ClientTimeout: 2 * time.Second,
	}
	c := mistralclient.NewClientWithConfig("fake-api-key", cfg)

	ctx := context.Background()
	inputMsgs := []mistralclient.Message{mistralclient.NewHumanMessage("Hello")}

	// When
	msg, err := c.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistralclient.ModelConfig{})

	// Then
	assert.NoError(t, err, "expected no error")
	expected := mistralclient.NewAssistantMessage("OK after timeout")
	assert.Equal(t, expected, msg, "expected message content")
}

func Test_ChatCompletion_ShouldFailWhenMaxRetriesReached(t *testing.T) {
	var attempts int32

	// Given: server always returns 503, expect retries to exhaust
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		http.Error(w, `{"error":"unavailable"}`, http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	cfg := &mistralclient.Config{
		Verbose:           false,
		MistralAPIBaseURL: srv.URL,
		RetryMaxRetries:   2, // total 3 attempts
		RetryWaitMin:      1 * time.Millisecond,
		RetryWaitMax:      2 * time.Millisecond,
	}
	c := mistralclient.NewClientWithConfig("fake-api-key", cfg)

	ctx := context.Background()
	inputMsgs := []mistralclient.Message{mistralclient.NewHumanMessage("Hi")}

	// When
	_, err := c.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistralclient.ModelConfig{})

	// Then
	assert.Error(t, err, "expected error")
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts), "expected 3 attempts")
}
