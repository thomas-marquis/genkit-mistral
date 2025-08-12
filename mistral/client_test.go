package mistral_test

import (
	"context"
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral"
	"net/http"
	"net/http/httptest"
	"testing"
)

func Test_MistralClient(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" && r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`
				{
					"choices": [
						{
							"message": {
								"role": "assistant",
								"content": "Hello, how can I assist you?"
							}
						}
					]
				}`))
		} else {
			http.NotFound(w, r)
		}
	}))
	defer mockServer.Close()

	fakeApiKey := "fake-api-key"

	t.Run("Successful ChatCompletion", func(t *testing.T) {
		// Given
		ctx := context.TODO()
		client := mistral.NewClient(fakeApiKey, mistral.WithBaseAPIURL(mockServer.URL))
		inputMsgs := []mistral.Message{
			mistral.NewSystemMessage("You are a helpful assistant."),
			mistral.NewHumanMessage("Hello!"),
		}

		// When
		res, err := client.ChatCompletion(ctx, inputMsgs, "mistral/mistral-large", &mistral.ModelConfig{})

		// Then
		expected := mistral.NewAssistantMessage("Hello, how can I assist you?")
		assert.NoError(t, err)
		assert.Equal(t, expected, res)
	})
}
