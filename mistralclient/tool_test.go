package mistralclient_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func TestToolCall(t *testing.T) {
	t.Run("should create new tool call with map input", func(t *testing.T) {
		tc := mistralclient.NewToolCall("toolid", 0, "myFunction", map[string]string{"param1": "value1"})

		assert.Equal(t, "toolid", tc.ID)
		assert.Equal(t, "function", tc.Type)
		assert.Equal(t, mistralclient.JsonMap{"input": map[string]string{"param1": "value1"}}, tc.Function.Arguments)
		assert.Equal(t, "myFunction", tc.Function.Name)
	})

	t.Run("should create new tool call with JsonMap input", func(t *testing.T) {
		tc := mistralclient.NewToolCall("toolid", 0, "myFunction", mistralclient.JsonMap{"param1": "value1"})

		assert.Equal(t, "toolid", tc.ID)
		assert.Equal(t, "function", tc.Type)
		assert.Equal(t, mistralclient.JsonMap{"param1": "value1"}, tc.Function.Arguments)
		assert.Equal(t, "myFunction", tc.Function.Name)
	})
}
