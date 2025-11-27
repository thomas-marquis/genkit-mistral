package mistralclient_test

import (
	"bytes"
	"context"
	"encoding/json"
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

// makeMockServer creates a simple HTTP test server that returns a fixed JSON response.
// It is kept for basic cases.
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

// makeMockServerWithCapture creates an HTTP test server that returns a fixed JSON response
// and also captures the JSON request body. The captured JSON is pretty-printed to make
// assertions easy to read in tests.
func makeMockServerWithCapture(t *testing.T, method, path, jsonResponse string, responseCode int, capturedBody *string) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == method && r.URL.Path == path {
			// Capture and pretty-print the incoming JSON body for assertions
			if r.Body != nil {
				defer r.Body.Close()
				raw, _ := io.ReadAll(r.Body)
				var anyJSON any
				if len(bytes.TrimSpace(raw)) > 0 && json.Unmarshal(raw, &anyJSON) == nil {
					pretty, _ := json.MarshalIndent(anyJSON, "", "  ")
					*capturedBody = string(pretty)
				} else {
					*capturedBody = string(raw)
				}
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(responseCode)
			_, _ = w.Write([]byte(jsonResponse))
			return
		}
		http.NotFound(w, r)
	}))
}

func TestChatCompletion(t *testing.T) {
	t.Run("Should call Mistral /chat/completion endpoint", func(t *testing.T) {
		var gotReq string
		mockServer := makeMockServerWithCapture(t, "POST", "/v1/chat/completions", `
				{
					"id": "1234567",
					"created": 1764230687,
					"model": "mistral-small-latest",
					"usage": {
						"prompt_tokens": 13,
						"total_tokens": 23,
						"completion_tokens": 10
					},
					"object": "chat.completion",
					"choices": [
						{
							"index": 0,
							"finish_reason": "stop",
							"message": {
								"role": "assistant",
								"tool_calls": null,
								"content": "Hello, how can I assist you?"
							}
						}
					]
				}`, http.StatusOK, &gotReq)
		defer mockServer.Close()

		// Given
		ctx := context.TODO()
		c := mistralclient.NewClient("fakeApiKey", mistralclient.WithBaseAPIURL(mockServer.URL))
		inputMsgs := []mistralclient.ChatMessage{
			mistralclient.NewSystemMessageFromString("You are a helpful assistant."),
			mistralclient.NewUserMessageFromString("Hello!"),
		}

		// When
		res, err := c.ChatCompletion(ctx, inputMsgs, "mistral-small-latest", &mistralclient.ModelConfig{})

		// Then
		assert.NoError(t, err)
		assert.Len(t, res.Choices, 1)
		assert.Equal(t, mistralclient.NewAssistantMessageFromString("Hello, how can I assist you?"), res.Choices[0].Message)

		// Check usage
		assert.Equal(t, 13, res.Usage.PromptTokens)
		assert.Equal(t, 23, res.Usage.TotalTokens)
		assert.Equal(t, 10, res.Usage.CompletionTokens)

		assert.Equal(t, "chat.completion", res.Object)
		assert.Equal(t, "mistral-small-latest", res.Model)

		expectedReq := `{
		  "model": "mistral-small-latest",
		  "messages": [
			{
			  "role": "system",
			  "content": "You are a helpful assistant."
			},
			{
			  "role": "user",
			  "content": "Hello!"
			}
		  ],
		  "parallel_tool_calls": true
		}`
		assert.JSONEq(t, expectedReq, gotReq)
	})

	t.Run("should call Mistral with tools, tool choice and multiple messages", func(t *testing.T) {
		var gotReq string
		mockServer := makeMockServerWithCapture(t, "POST", "/v1/chat/completions", `
				{
					"id": "12345",
					"created": 1764282082,
					"model": "mistral-small-latest",
					"usage": {
						"prompt_tokens": 142,
						"total_tokens": 158,
						"completion_tokens": 16
					},
					"object": "chat.completion",
					"choices": [
						{
							"index": 0,
							"finish_reason": "tool_calls",
							"message": {
								"role": "assistant",
								"tool_calls": [
									{
										"id": "abcde",
										"function": {
											"name": "add",
											"arguments": "{\"a\": 2, \"b\": 3}"
										},
										"index": 0
									}
								],
								"content": ""
							}
						}
					]
				}`, http.StatusOK, &gotReq)
		defer mockServer.Close()

		// Given
		ctx := context.TODO()
		c := mistralclient.NewClient("fakeApiKey", mistralclient.WithBaseAPIURL(mockServer.URL))
		inputMsgs := []mistralclient.ChatMessage{
			mistralclient.NewSystemMessageFromString("You are a helpful assistant."),
			mistralclient.NewUserMessageFromString("2 + 3?"),
		}

		// When
		res, err := c.ChatCompletion(ctx,
			inputMsgs,
			"mistral-small-latest",
			&mistralclient.ModelConfig{},
			mistralclient.WithTools([]mistralclient.Tool{
				mistralclient.NewFuncTool("add", "add two numbers", map[string]any{
					"type": "object",
					"properties": map[string]any{
						"a": map[string]any{
							"type": "number",
						},
						"b": map[string]any{
							"type": "number",
						},
					},
				}),
				mistralclient.NewFuncTool("getUserById", "get user by id", map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id": map[string]any{
							"type": "string",
						},
					},
				}),
			}),
			mistralclient.WithToolChoice(mistralclient.ToolChoiceAny),
		)

		// Then
		assert.NoError(t, err)
		assert.Len(t, res.Choices, 1)
		assert.Equal(t, mistralclient.RoleAssistant, res.Choices[0].Message.Type())
		assert.Equal(t, mistralclient.FinishReasonToolCalls, res.Choices[0].FinishReason)
		assert.Len(t, res.Choices[0].Message.ToolCalls, 1)
		assert.Equal(t, "add", res.Choices[0].Message.ToolCalls[0].Function.Name)
		assert.Equal(t, mistralclient.JsonMap{"a": 2., "b": 3.}, res.Choices[0].Message.ToolCalls[0].Function.Arguments)

		expectedReq := `{
		  "model": "mistral-small-latest",
		  "messages": [
			{
			  "role": "system",
			  "content": "You are a helpful assistant."
			},
			{
			  "role": "user",
			  "content": "2 + 3?"
			}
		  ],
		  "tools": [
			{
			  "type": "function",
			  "function": {
				"name": "add",
				"description": "add two numbers",
				"parameters": {
				  "type": "object",
				  "description": "",
				  "properties": {
					"a": {"type": "number", "description": ""},
					"b": {"type": "number", "description": ""}
				  }
				}
			  }
			},
			{
			  "type": "function",
			  "function": {
				"name": "getUserById",
				"description": "get user by id",
				"parameters": {
				  "type": "object",
				  "description": "",
				  "properties": {
					"id": {"type": "string", "description": ""}
				  }
				}
			  }
			}
		  ],
		  "tool_choice": "any",
		  "parallel_tool_calls": true
		}`
		assert.JSONEq(t, expectedReq, gotReq)
	})

	t.Run("Should retry on 5xx then succeed", func(t *testing.T) {
		var attempts int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&attempts, 1)
			if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
				http.NotFound(w, r)
				return
			}
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
		}
		c := mistralclient.NewClientWithConfig("fake-api-key", cfg)
		ctx := context.Background()
		inputMsgs := []mistralclient.ChatMessage{mistralclient.NewUserMessageFromString("Hi!")}

		res, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})
		assert.NoError(t, err)
		assert.Len(t, res.Choices, 1)
		assert.Equal(t, mistralclient.NewAssistantMessageFromString("Hello after retries"), res.Choices[0].Message)
		assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
	})

	t.Run("Should not retry on 400 and fail immediately", func(t *testing.T) {
		var attempts int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&attempts, 1)
			if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
				http.NotFound(w, r)
				return
			}
			http.Error(w, `{
				"object": "error",
				"message": {
					"detail": [
						{
							"type": "extra_forbidden",
							"loc": [
								"body",
								"parallel_tool_callss"
							],
							"msg": "Extra inputs are not permitted",
							"input": true
						}
					]
				},
				"type": "invalid_request_error",
				"param": null,
				"code": null
			}`, http.StatusBadRequest)
		}))
		defer srv.Close()

		cfg := &mistralclient.Config{
			Verbose:           false,
			MistralAPIBaseURL: srv.URL,
			RetryMaxRetries:   5,
			RetryWaitMin:      1 * time.Millisecond,
			RetryWaitMax:      2 * time.Millisecond,
		}
		c := mistralclient.NewClientWithConfig("fake-api-key", cfg)
		ctx := context.Background()
		inputMsgs := []mistralclient.ChatMessage{mistralclient.NewUserMessageFromString("Hi!")}

		_, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})
		assert.Error(t, err)
		assert.Equal(t, mistralclient.ErrorResponse{}, err)
		assert.Equal(t, int32(1), atomic.LoadInt32(&attempts))
	})

	t.Run("Should retry on timeout error then succeed", func(t *testing.T) {
		successJSON := []byte(`{"choices":[{"message":{"role":"assistant","content":"OK after timeout"}}]}`)
		cfg := &mistralclient.Config{
			Verbose:           false,
			RetryMaxRetries:   3,
			RetryWaitMin:      1 * time.Millisecond,
			RetryWaitMax:      5 * time.Millisecond,
			MistralAPIBaseURL: "http://invalid.local",
			Transport: &flakyRoundTripper{
				failuresLeft: 1,
				successBody:  successJSON,
			},
			ClientTimeout: 2 * time.Second,
		}
		c := mistralclient.NewClientWithConfig("fake-api-key", cfg)
		ctx := context.Background()
		inputMsgs := []mistralclient.ChatMessage{mistralclient.NewUserMessageFromString("Hello")}

		res, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})
		assert.NoError(t, err)
		assert.Len(t, res.Choices, 1)
		assert.Equal(t, mistralclient.NewAssistantMessageFromString("OK after timeout"), res.Choices[0].Message)
	})

	t.Run("Should fail when max retries reached", func(t *testing.T) {
		var attempts int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&attempts, 1)
			http.Error(w, `{"error":"unavailable"}`, http.StatusServiceUnavailable)
		}))
		defer srv.Close()

		cfg := &mistralclient.Config{
			Verbose:           false,
			MistralAPIBaseURL: srv.URL,
			RetryMaxRetries:   2,
			RetryWaitMin:      1 * time.Millisecond,
			RetryWaitMax:      2 * time.Millisecond,
		}
		c := mistralclient.NewClientWithConfig("fake-api-key", cfg)
		ctx := context.Background()
		inputMsgs := []mistralclient.ChatMessage{mistralclient.NewUserMessageFromString("Hi")}

		_, err := c.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistralclient.ModelConfig{})
		assert.Error(t, err)
		assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
	})

	t.Run("Response should unmarshal with correct created_at time format", func(t *testing.T) {
		j := `{
            "id": "12345",
            "created": 1764278339,
            "model": "mistral-small-latest",
            "usage": {
                "prompt_tokens": 13,
                "total_tokens": 23,
                "completion_tokens": 10
            },
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "tool_calls": null,
                        "content": "Hello! How can I assist you today?"
                    }
                }
            ]
        }`

		var tc mistralclient.ChatCompletionResponse
		assert.NoError(t, json.Unmarshal([]byte(j), &tc))
		assert.Equal(t, time.Date(2025, time.November, 27, 21, 18, 59, 0, time.UTC), tc.Created)
	})
}

// Deprecated leftover tests removed: all ChatCompletion tests are now under TestChatCompletion.
// Keeping a no-op to avoid accidental reintroduction.
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
	var _ = c
	//ctx := context.Background()
	//inputMsgs := []mistralclient.Message{
	//	mistralclient.NewUserMessage("Hi!"),
	//}
	//
	//// When
	//res, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})
	//
	//// Then
	//assert.NoError(t, err, "expected no error")
	//assert.Len(t, res.Choices, 1, "expected 1 choice")
	//assert.Equal(t, mistralclient.NewAssistantMessage("Hello after retries"), res.Choices[0].Message.Message, "expected message")
	//assert.Equal(t, int32(3), atomic.LoadInt32(&attempts), "expected 3 attempts")
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
	//ctx := context.Background()
	//inputMsgs := []mistralclient.Message{mistralclient.NewUserMessage("Hi!")}
	//
	//// When
	//_, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})
	//
	//// Then
	//if err == nil {
	//	t.Fatalf("expected error, got nil")
	//}
	//if got := atomic.LoadInt32(&attempts); got != 1 {
	//	t.Fatalf("expected exactly 1 attempt on 400, got %d", got)
	//}
	var _ = c
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

	//ctx := context.Background()
	//inputMsgs := []mistralclient.Message{mistralclient.NewUserMessage("Hello")}
	var _ = c
	//
	//// When
	//res, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})
	//
	//// Then
	//assert.NoError(t, err, "expected no error")
	//assert.Len(t, res.Choices, 1, "expected 1 choice")
	//assert.Equal(t, mistralclient.NewAssistantMessage("OK after timeout"), res.Choices[0].Message.Message, "expected message content")
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
	var _ = c

	//ctx := context.Background()
	//inputMsgs := []mistralclient.Message{mistralclient.NewUserMessage("Hi")}
	//
	//// When
	//_, err := c.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistralclient.ModelConfig{})
	//
	//// Then
	//assert.Error(t, err, "expected error")
	//assert.Equal(t, int32(3), atomic.LoadInt32(&attempts), "expected 3 attempts")
}

func TestChatCompletionResponse(t *testing.T) {
	t.Run("should unmarshall with correct created_at time format", func(t *testing.T) {
		j := `{
			"id": "12345",
			"created": 1764278339,
			"model": "mistral-small-latest",
			"usage": {
				"prompt_tokens": 13,
				"total_tokens": 23,
				"completion_tokens": 10
			},
			"object": "chat.completion",
			"choices": [
				{
					"index": 0,
					"finish_reason": "stop",
					"message": {
						"role": "assistant",
						"tool_calls": null,
						"content": "Hello! How can I assist you today?"
					}
				}
			]
		}`

		var tc mistralclient.ChatCompletionResponse

		assert.NoError(t, json.Unmarshal([]byte(j), &tc))
		assert.Equal(t, time.Date(2025, time.November, 27, 21, 18, 59, 0, time.UTC), tc.Created)
	})
}
