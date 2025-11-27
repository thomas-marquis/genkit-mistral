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

func TestMapFunctionParameters(t *testing.T) {
	t.Run("should return zero value when parameters is nil", func(t *testing.T) {
		got := mistralclient.NewPropertyDefinition(nil)
		assert.Equal(t, mistralclient.PropertyDefinition{}, got)
	})

	t.Run("should return zero value when parameters is empty map", func(t *testing.T) {
		got := mistralclient.NewPropertyDefinition(map[string]any{})
		assert.Equal(t, mistralclient.PropertyDefinition{}, got)
	})

	t.Run("should map top level fields", func(t *testing.T) {
		in := map[string]any{
			"description":          "A root object",
			"type":                 "object",
			"additionalProperties": true,
			"default":              map[string]any{"foo": "bar"},
		}

		got := mistralclient.NewPropertyDefinition(in)

		assert.Equal(t, "A root object", got.Description)
		assert.Equal(t, "object", got.Type)
		assert.True(t, got.AdditionalProperties)
		assert.Equal(t, map[string]any{"foo": "bar"}, got.Default)
		assert.Nil(t, got.Properties)
	})

	t.Run("should recursively map nested properties", func(t *testing.T) {
		in := map[string]any{
			"description": "outer",
			"type":        "object",
			"properties": map[string]any{
				"name": map[string]any{
					"type":        "string",
					"description": "the name",
					"default":     "john",
				},
				"age": map[string]any{
					"type":    "integer",
					"default": 42,
				},
			},
		}

		got := mistralclient.NewPropertyDefinition(in)

		if assert.NotNil(t, got.Properties) {
			assert.Len(t, got.Properties, 2)

			name := got.Properties["name"]
			assert.Equal(t, "string", name.Type)
			assert.Equal(t, "the name", name.Description)
			assert.Equal(t, any("john"), name.Default)

			age := got.Properties["age"]
			assert.Equal(t, "integer", age.Type)
			assert.Equal(t, any(42), age.Default)
		}
	})

	t.Run("should coerce simple non-map property definitions to string type", func(t *testing.T) {
		in := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"flag":   true,     // becomes "true"
				"count":  3,        // becomes "3"
				"pi":     3.14,     // becomes "3.14"
				"status": "string", // remains "string"
			},
		}

		got := mistralclient.NewPropertyDefinition(in)

		if assert.NotNil(t, got.Properties) {
			assert.Equal(t, "true", got.Properties["flag"].Type)
			assert.Equal(t, "3", got.Properties["count"].Type)
			// float formatting via Sprintf may be "3.14" or "3.14" with no extra zeros; assert prefix
			assert.Contains(t, got.Properties["pi"].Type, "3.14")
			assert.Equal(t, "string", got.Properties["status"].Type)
		}
	})
}
