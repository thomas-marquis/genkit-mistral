package mistralclient

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type EmbeddingVector []float32

type EmbeddingRequest struct {
	Model           string   `json:"model"`
	Input           []string `json:"input"`
	OutputDimension int      `json:"output_dimension,omitempty"`
	OutputDtype     string   `json:"output_dtype,omitempty"`
}

type EmbeddingResponse struct {
	ID     string        `json:"id"`
	Object string        `json:"object"`
	Model  string        `json:"model"`
	Usage  UsageResponse `json:"usage"`
	Data   []struct {
		Object    string          `json:"object"`
		Embedding EmbeddingVector `json:"embedding"`
		Index     int             `json:"index"`
	} `json:"data"`
	Latency time.Duration `json:"latency_ms,omitempty"`
}

func (r *EmbeddingResponse) Embeddings() []EmbeddingVector {
	vectors := make([]EmbeddingVector, len(r.Data))
	for i, data := range r.Data {
		vectors[i] = data.Embedding
	}
	return vectors
}

func (c *Client) TextEmbedding(ctx context.Context, texts []string, model string) (*EmbeddingResponse, error) {
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

	respBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if c.verbose {
		logger.Println("TextEmbedding called")
	}

	var resp EmbeddingResponse
	err = json.Unmarshal(respBody, &resp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response body: %w", err)
	}

	resp.Latency = lat

	return &resp, nil
}
