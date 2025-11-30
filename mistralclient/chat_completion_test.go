package mistralclient_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

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
		// Given
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

		// When
		_, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})

		// Then
		expectedErr := &mistralclient.ErrorResponse{
			Object: "error",
			Message: mistralclient.ErrorResponseMessage{
				Detail: []mistralclient.ErrorResponseDetail{
					{
						Type:  "extra_forbidden",
						Loc:   []string{"body", "parallel_tool_callss"},
						Msg:   "Extra inputs are not permitted",
						Input: true,
					},
				},
			},
			Type:  "invalid_request_error",
			Param: nil,
			Code:  nil,
		}
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.Equal(t, int32(1), atomic.LoadInt32(&attempts))
	})

	t.Run("Should retry on timeout error then succeed", func(t *testing.T) {
		// Given
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

		// When
		res, err := c.ChatCompletion(ctx, inputMsgs, "mistral-large", &mistralclient.ModelConfig{})

		// Then
		assert.NoError(t, err)
		assert.Len(t, res.Choices, 1)
		assert.Equal(t, mistralclient.NewAssistantMessageFromString("OK after timeout"), res.Choices[0].Message)
	})

	t.Run("Should fail when max retries reached", func(t *testing.T) {
		// Given
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

		// When
		_, err := c.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistralclient.ModelConfig{})

		// Then
		assert.Error(t, err)
		assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
	})
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
