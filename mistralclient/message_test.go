package mistralclient_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func TestSystemMessage(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{"type": "system", "content": "hello"}`
		var sm mistralclient.SystemMessage

		assert.NoError(t, json.Unmarshal([]byte(j), &sm))
		assert.Equal(t, "hello", sm.Content)
		assert.Equal(t, mistralclient.RoleSystem, sm.Type())
		assert.Equal(t, mistralclient.RoleSystem, sm.Role)
	})
}

//func TestMessageRoleHelpers(t *testing.T) {
//	m1 := mistralclient.NewUserMessage("hi")
//	if !m1.IsHuman() || m1.IsAssistant() || m1.IsSystem() {
//		t.Fatalf("expected human message flags, got: %+v", m1)
//	}
//
//	m2 := mistralclient.NewAssistantMessage("ok")
//	if !m2.IsAssistant() || m2.IsHuman() || m2.IsSystem() {
//		t.Fatalf("expected assistant message flags, got: %+v", m2)
//	}
//
//	m3 := mistralclient.NewSystemMessage("sys")
//	if !m3.IsSystem() || m3.IsHuman() || m3.IsAssistant() {
//		t.Fatalf("expected system message flags, got: %+v", m3)
//	}
//}
