package mistralclient_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func TestTextContent(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{"text": "hello", "type": "text"}`
		var tc mistralclient.TextContent

		assert.NoError(t, json.Unmarshal([]byte(j), &tc))
		assert.Equal(t, "hello", tc.Text)
		assert.Equal(t, mistralclient.ContentTypeText, tc.Type())
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		tc := mistralclient.NewTextContent("hello")
		j, err := json.Marshal(tc)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"text","text":"hello"}`, string(j))
	})
}

func TestImageUrlContent(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{"image_url": "https://example.com/image.png", "type": "image_url"}`
		var ic mistralclient.ImageUrlContent

		assert.NoError(t, json.Unmarshal([]byte(j), &ic))
		assert.Equal(t, "https://example.com/image.png", ic.ImageURL)
		assert.Equal(t, mistralclient.ContentTypeImageURL, ic.Type())
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		ic := mistralclient.NewImageUrlContent("https://example.com/image.png")
		j, err := json.Marshal(ic)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"image_url","image_url":"https://example.com/image.png"}`, string(j))
	})
}

func TestDocumentUrlContent(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{"document_name": "document.pdf", "document_url": "https://example.com/document.pdf", "type": "document_url"}`
		var duc mistralclient.DocumentUrlContent

		assert.NoError(t, json.Unmarshal([]byte(j), &duc))
		assert.Equal(t, "document.pdf", duc.DocumentName)
		assert.Equal(t, "https://example.com/document.pdf", duc.DocumentURL)
		assert.Equal(t, mistralclient.ContentTypeDocumentURL, duc.Type())
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		duc := mistralclient.NewDocumentUrlContent("document.pdf", "https://example.com/document.pdf")
		j, err := json.Marshal(duc)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"document_url","document_name":"document.pdf","document_url":"https://example.com/document.pdf"}`, string(j))
	})

	t.Run("should be unmarshaled from json with null document name", func(t *testing.T) {
		j := `{"document_name": null, "document_url": "https://example.com/document.pdf", "type": "document_url"}`
		var duc mistralclient.DocumentUrlContent

		assert.NoError(t, json.Unmarshal([]byte(j), &duc))
		assert.Equal(t, "", duc.DocumentName)
		assert.Equal(t, "https://example.com/document.pdf", duc.DocumentURL)
		assert.Equal(t, mistralclient.ContentTypeDocumentURL, duc.Type())
	})

	t.Run("should be marshaled to json with null document name", func(t *testing.T) {
		duc := mistralclient.NewDocumentUrlContent("", "https://example.com/document.pdf")
		j, err := json.Marshal(duc)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"document_url","document_url":"https://example.com/document.pdf"}`, string(j))
	})
}

func TestReferenceContent(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{"reference_ids": [1, 2, 3, 5, 8], "type": "reference"}`
		var rc mistralclient.ReferenceContent

		assert.NoError(t, json.Unmarshal([]byte(j), &rc))
		assert.Equal(t, []int{1, 2, 3, 5, 8}, rc.ReferenceIds)
		assert.Equal(t, mistralclient.ContentTypeReference, rc.Type())
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		rc := mistralclient.NewReferenceContent(1, 2, 3, 5, 8)
		j, err := json.Marshal(rc)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"reference","reference_ids":[1,2,3,5,8]}`, string(j))
	})
}

func TestFileContent(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{"file_id": "1234567890", "type": "file"}`
		var fc mistralclient.FileContent

		assert.NoError(t, json.Unmarshal([]byte(j), &fc))
		assert.Equal(t, "1234567890", fc.FileId)
		assert.Equal(t, mistralclient.ContentTypeFile, fc.Type())
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		fc := mistralclient.NewFileContent("1234567890")
		j, err := json.Marshal(fc)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"file","file_id":"1234567890"}`, string(j))
	})
}

func TestThinkContent(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{
			"type": "thinking",
			"closed": false,
			"thinking": [
				{"type": "text", "text": "hello"},
				{"type": "reference", "reference_ids": [1, 2, 3]}
			]
		}`
		var tc mistralclient.ThinkContent

		assert.NoError(t, json.Unmarshal([]byte(j), &tc))
		assert.Equal(t, mistralclient.ContentTypeThink, tc.Type())
		assert.Equal(t, false, tc.Closed)
		assert.Len(t, tc.Thinking, 2)
		assert.Equal(t, mistralclient.ContentTypeText, tc.Thinking[0].Type())
		assert.Equal(t, "hello", tc.Thinking[0].(*mistralclient.TextContent).Text)
		assert.Equal(t, mistralclient.ContentTypeReference, tc.Thinking[1].Type())
		assert.Equal(t, []int{1, 2, 3}, tc.Thinking[1].(*mistralclient.ReferenceContent).ReferenceIds)
	})

	t.Run("should be unmarshaled from json with closed true by default", func(t *testing.T) {
		j := `{
			"type": "thinking",
			"thinking": []
		}`
		var tc mistralclient.ThinkContent

		assert.NoError(t, json.Unmarshal([]byte(j), &tc))
		assert.Equal(t, mistralclient.ContentTypeThink, tc.Type())
		assert.Equal(t, true, tc.Closed)
		assert.Len(t, tc.Thinking, 0)
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		tc := mistralclient.NewThinkContent(
			mistralclient.NewTextContent("hello"),
			mistralclient.NewReferenceContent(1, 2, 3),
		)
		j, err := json.Marshal(tc)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"thinking","closed":true,"thinking":[{"type":"text","text":"hello"},{"type":"reference","reference_ids":[1,2,3]}]}`, string(j))
	})

	t.Run("should be marshaled to json with closed false", func(t *testing.T) {
		tc := mistralclient.NewThinkContent()
		tc.Closed = false
		j, err := json.Marshal(tc)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"thinking","closed":false,"thinking":[]}`, string(j))
	})

	t.Run("should panic when trying to add an unsupported content type", func(t *testing.T) {
		assert.PanicsWithValue(t, "only text and reference content can be added to a thinking content", func() {
			mistralclient.NewThinkContent(mistralclient.NewImageUrlContent("https://example.com/image.png"))
		})
	})

	t.Run("should panic when trying to add a nil content", func(t *testing.T) {
		assert.PanicsWithValue(t, "nil content cannot be added to a thinking content", func() {
			mistralclient.NewThinkContent(nil)
		})
	})
}

func TestAudioContent(t *testing.T) {
	t.Run("should be unmarshaled from json", func(t *testing.T) {
		j := `{"input_audio": "https://example.com/audio.mp3", "type": "input_audio"}`
		var ac mistralclient.AudioContent

		assert.NoError(t, json.Unmarshal([]byte(j), &ac))
		assert.Equal(t, mistralclient.ContentTypeAudio, ac.Type())
		assert.Equal(t, "https://example.com/audio.mp3", ac.InputAudio)
	})

	t.Run("should be marshaled to json", func(t *testing.T) {
		ac := mistralclient.NewAudioContent("https://example.com/audio.mp3")
		j, err := json.Marshal(ac)

		assert.NoError(t, err)
		assert.Equal(t, `{"type":"input_audio","input_audio":"https://example.com/audio.mp3"}`, string(j))
	})
}
