package mistralclient_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

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

type mockCall struct {
	method       string
	path         string
	jsonResponse string
	responseCode int
}

func Test_MistralClient(t *testing.T) {
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
