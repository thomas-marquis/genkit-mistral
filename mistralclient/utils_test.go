package mistralclient_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func TestJsonMap(t *testing.T) {
	t.Run("should unmarshall simple json map", func(t *testing.T) {
		j := `{ "key": "value" }`
		var m mistralclient.JsonMap

		assert.NoError(t, json.Unmarshal([]byte(j), &m))
		assert.Equal(t, mistralclient.JsonMap{"key": "value"}, m)
	})

	t.Run("should unmarshall json map from quoted json string", func(t *testing.T) {
		j := `"{\"key\": \"value\"}"`
		var m mistralclient.JsonMap

		assert.NoError(t, json.Unmarshal([]byte(j), &m))
		assert.Equal(t, mistralclient.JsonMap{"key": "value"}, m)
	})

	t.Run("should unmarshall empty json map", func(t *testing.T) {
		j := `{}`
		var m mistralclient.JsonMap

		assert.NoError(t, json.Unmarshal([]byte(j), &m))
		assert.Equal(t, mistralclient.JsonMap{}, m)
	})

	t.Run("should unmarshall null json map", func(t *testing.T) {
		j := "null"
		var m mistralclient.JsonMap

		assert.NoError(t, json.Unmarshal([]byte(j), &m))
		assert.Equal(t, mistralclient.JsonMap(nil), m)
	})
}
