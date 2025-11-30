package mistralclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type EmbeddingVector []float32

type EmbeddingEncodingFormat string

const (
	EmbeddingEncodingFloat  EmbeddingEncodingFormat = "float"
	EmbeddingEncodingBase64 EmbeddingEncodingFormat = "base64"
)

type EmbeddingOutputDtype string

const (
	EmbeddingOutputDtypeFloat   EmbeddingOutputDtype = "float"
	EmbeddingOutputDtypeInt8    EmbeddingOutputDtype = "int8"
	EmbeddingOutputDtypeUInt8   EmbeddingOutputDtype = "uint8"
	EmbeddingOutputDtypeBinary  EmbeddingOutputDtype = "binary"
	EmbeddingOutputDtypeUBinary EmbeddingOutputDtype = "ubinary"
)

type EmbeddingRequest struct {
	// Model is the ID of the model to be used for embedding.
	Model string `json:"model"`

	// Input i the text content to be embedded.
	Input []string `json:"input"`

	// OutputDimension is the dimension of the output embeddings when feature available.
	// If not provided, a default output dimension will be used.
	OutputDimension int `json:"output_dimension,omitempty"`

	OutputDtype EmbeddingOutputDtype `json:"output_dtype,omitempty"`

	EncodingFormat EmbeddingEncodingFormat `json:"encoding_format,omitempty"`
}

type EmbeddingData struct {
	Object    string          `json:"object"`
	Embedding EmbeddingVector `json:"embedding"`
	Index     int             `json:"index"`
}

type EmbeddingResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Model   string          `json:"model"`
	Usage   UsageInfo       `json:"usage"`
	Data    []EmbeddingData `json:"data"`
	Latency time.Duration   `json:"latency_ms,omitempty"`
}

func (r *EmbeddingResponse) Embeddings() []EmbeddingVector {
	vectors := make([]EmbeddingVector, len(r.Data))
	for i, data := range r.Data {
		vectors[i] = data.Embedding
	}
	return vectors
}

type EmbeddingOptions struct {
	OutputType EmbeddingOutputDtype
	OutputDim  int
	Encoding   EmbeddingEncodingFormat
}

func (c *clientImpl) Embeddings(ctx context.Context, texts []string, model string) (*EmbeddingResponse, error) {
	c.rateLimiter.Wait()

	url := fmt.Sprintf("%s/v1/embeddings", c.baseURL)

	reqBody := EmbeddingRequest{
		Input: texts,
		Model: model,
	}

	jsonValue, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	response, lat, err := sendRequest(ctx, c, http.MethodPost, url, jsonValue)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	if c.verbose {
		logger.Println("Embeddings called")
	}

	var resp EmbeddingResponse
	if err = unmarshallBody(response, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response body: %w", err)
	}

	resp.Latency = lat

	return &resp, nil
}
