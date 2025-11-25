package mistralclient

import (
	"encoding/json"
)

type ContentType string

func (c ContentType) String() string {
	return string(c)
}

const (
	ContentTypeText        ContentType = "text"
	ContentTypeImageURL    ContentType = "image_url"
	ContentTypeDocumentURL ContentType = "document_url"
	ContentTypeReference   ContentType = "reference"
	ContentTypeFile        ContentType = "file"
	ContentTypeThink       ContentType = "thinking"
	ContentTypeAudio       ContentType = "input_audio"
)

type Content interface {
	String() string
	Chunks() []ContentChunk
}

type ContentString string

var _ Content = (*ContentString)(nil)

func (s ContentString) String() string {
	return string(s)
}

func (s ContentString) Chunks() []ContentChunk {
	return nil
}

type ContentChunks []ContentChunk

var _ Content = ContentChunks{}

func (c ContentChunks) String() string {
	return ""
}

func (c ContentChunks) Chunks() []ContentChunk {
	return c
}

type ContentChunk interface {
	Type() ContentType
}

type TextContent struct {
	ContentType ContentType `json:"type"`
	Text        string      `json:"text"`
}

var _ ContentChunk = (*TextContent)(nil)

func NewTextContent(text string) *TextContent {
	return &TextContent{
		ContentType: ContentTypeText,
		Text:        text,
	}
}

func (c *TextContent) Type() ContentType {
	return ContentTypeText
}

type ImageUrlContent struct {
	ContentType ContentType `json:"type"`
	ImageURL    string      `json:"image_url"`
}

var _ ContentChunk = (*ImageUrlContent)(nil)

func NewImageUrlContent(imageUrl string) *ImageUrlContent {
	return &ImageUrlContent{
		ContentType: ContentTypeImageURL,
		ImageURL:    imageUrl,
	}
}

func (c *ImageUrlContent) Type() ContentType {
	return ContentTypeImageURL
}

type DocumentUrlContent struct {
	ContentType  ContentType `json:"type"`
	DocumentName string      `json:"document_name,omitempty"`
	DocumentURL  string      `json:"document_url"`
}

var _ ContentChunk = (*DocumentUrlContent)(nil)

func NewDocumentUrlContent(documentName, documentURL string) *DocumentUrlContent {
	return &DocumentUrlContent{
		ContentType:  ContentTypeDocumentURL,
		DocumentName: documentName,
		DocumentURL:  documentURL,
	}
}

func (c *DocumentUrlContent) Type() ContentType {
	return ContentTypeDocumentURL
}

type ReferenceContent struct {
	ContentType  ContentType `json:"type"`
	ReferenceIds []int       `json:"reference_ids"`
}

var _ ContentChunk = (*ReferenceContent)(nil)

func NewReferenceContent(referenceIds ...int) *ReferenceContent {
	return &ReferenceContent{
		ContentType:  ContentTypeReference,
		ReferenceIds: referenceIds,
	}
}

func (c *ReferenceContent) Type() ContentType {
	return ContentTypeReference
}

type FileContent struct {
	ContentType ContentType `json:"type"`
	FileId      string      `json:"file_id"`
}

var _ ContentChunk = (*FileContent)(nil)

func NewFileContent(fileId string) *FileContent {
	return &FileContent{
		ContentType: ContentTypeFile,
		FileId:      fileId,
	}
}

func (c *FileContent) Type() ContentType {
	return ContentTypeFile
}

type ThinkContent struct {
	ContentType ContentType    `json:"type"`
	Closed      bool           `json:"closed"`
	Thinking    []ContentChunk `json:"thinking"`
}

var _ ContentChunk = (*ThinkContent)(nil)
var _ json.Unmarshaler = (*ThinkContent)(nil)

func NewThinkContent(thinking ...ContentChunk) *ThinkContent {
	c := &ThinkContent{
		ContentType: ContentTypeThink,
		Closed:      true,
		Thinking:    make([]ContentChunk, 0),
	}
	for _, t := range thinking {
		if t == nil {
			panic("nil content cannot be added to a thinking content")
		}
		if t.Type() != ContentTypeText && t.Type() != ContentTypeReference {
			panic("only text and reference content can be added to a thinking content")
		}
		c.Thinking = append(c.Thinking, t)
	}
	return c
}

func (c *ThinkContent) Type() ContentType {
	return ContentTypeThink
}

func (c *ThinkContent) UnmarshalJSON(data []byte) error {
	var res map[string]any
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}
	c.ContentType = ContentType(res["type"].(string))
	thinkings := res["thinking"].([]any)
	c.Thinking = make([]ContentChunk, len(thinkings))

	for i, thinking := range thinkings {
		t := thinking.(map[string]any)

		switch t["type"].(string) {
		case ContentTypeText.String():
			c.Thinking[i] = NewTextContent(t["text"].(string))

		case ContentTypeReference.String():
			refIds, exists := t["reference_ids"]
			var refIdsSlice []int
			if exists {
				for _, refId := range refIds.([]any) {
					refIdsSlice = append(refIdsSlice, int(refId.(float64)))
				}
			}
			c.Thinking[i] = NewReferenceContent(refIdsSlice...)
		}
	}

	if cl, ok := res["closed"]; ok {
		c.Closed = cl.(bool)
	} else {
		c.Closed = true
	}

	return nil
}

type AudioContent struct {
	ContentType ContentType `json:"type"`
	InputAudio  string      `json:"input_audio"`
}

var _ ContentChunk = (*AudioContent)(nil)

func NewAudioContent(inputAudio string) *AudioContent {
	return &AudioContent{
		ContentType: ContentTypeAudio,
		InputAudio:  inputAudio,
	}
}

func (c *AudioContent) Type() ContentType {
	return ContentTypeAudio
}
