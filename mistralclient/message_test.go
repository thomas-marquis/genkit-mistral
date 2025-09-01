package mistralclient

import "testing"

func TestMessageRoleHelpers(t *testing.T) {
	m1 := NewHumanMessage("hi")
	if !m1.IsHuman() || m1.IsAssistant() || m1.IsSystem() {
		t.Fatalf("expected human message flags, got: %+v", m1)
	}

	m2 := NewAssistantMessage("ok")
	if !m2.IsAssistant() || m2.IsHuman() || m2.IsSystem() {
		t.Fatalf("expected assistant message flags, got: %+v", m2)
	}

	m3 := NewSystemMessage("sys")
	if !m3.IsSystem() || m3.IsHuman() || m3.IsAssistant() {
		t.Fatalf("expected system message flags, got: %+v", m3)
	}
}
