package mistral_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
	"github.com/thomas-marquis/genkit-mistral/mocks"
	"go.uber.org/mock/gomock"
)

var (
	ctxType = reflect.TypeOf((*context.Context)(nil)).Elem()
)

func Test_Embedder_ShouldReturnAnSingleVector(t *testing.T) {
	// Given
	ctrl := gomock.NewController(t)
	mockClient := mocks.NewMockClient(ctrl)

	inputText := "Hello, World!"
	expectedVec := []float32{1, 2, 3}

	mockClient.EXPECT().
		TextEmbedding(
			gomock.AssignableToTypeOf(ctxType),
			gomock.Eq([]string{inputText}),
			gomock.Eq("mistral-embed")).
		Return(&mistralclient.EmbeddingResponse{
			Data: []mistralclient.EmbeddingData{
				{
					Embedding: expectedVec,
				},
			},
		}, nil)

	p := mistral.NewPlugin("fake")
	p.Client = mockClient

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p))

	// When
	res, err := genkit.Embed(ctx, g,
		ai.WithDocs(ai.DocumentFromText(inputText, nil)),
		ai.WithEmbedderName("mistral/mistral-embed"))

	// Then
	assert.NoError(t, err)
	assert.Len(t, res.Embeddings, 1)
	assert.Equal(t, res.Embeddings[0].Embedding, expectedVec)
}

func Test_Embedder_ShouldReturnMultipleVectors(t *testing.T) {
	// Given
	ctrl := gomock.NewController(t)
	mockClient := mocks.NewMockClient(ctrl)

	expectedVec1 := []float32{1, 2, 3}
	expectedVec2 := []float32{4, 5, 6}

	mockClient.EXPECT().
		TextEmbedding(
			gomock.AssignableToTypeOf(ctxType),
			gomock.Eq([]string{"Hello, World!", "Hello there!\nMy name is Obi-Wan Kenobi."}),
			gomock.Eq("mistral-embed")).
		Return(&mistralclient.EmbeddingResponse{
			Data: []mistralclient.EmbeddingData{
				{
					Embedding: expectedVec1,
				},
				{
					Embedding: expectedVec2,
				},
			},
		}, nil)

	p := mistral.NewPlugin("fake")
	p.Client = mockClient

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p))

	// When
	res, err := genkit.Embed(ctx, g,
		ai.WithDocs(
			ai.DocumentFromText("Hello, World!", nil),
			&ai.Document{
				Content: []*ai.Part{
					ai.NewTextPart("Hello there!"),
					ai.NewTextPart("My name is Obi-Wan Kenobi."),
				},
			},
		),
		ai.WithEmbedderName("mistral/mistral-embed"))

	// Then
	assert.NoError(t, err)
	assert.Len(t, res.Embeddings, 2)
	assert.Equal(t, res.Embeddings[0].Embedding, expectedVec1)
	assert.Equal(t, res.Embeddings[1].Embedding, expectedVec2)
}

func Test_Embedder_ShouldReturnErrorWhenNoVectorIsReturned(t *testing.T) {
	// Given
	ctrl := gomock.NewController(t)
	mockClient := mocks.NewMockClient(ctrl)

	mockClient.EXPECT().
		TextEmbedding(
			gomock.AssignableToTypeOf(ctxType),
			gomock.Eq([]string{"Hello, World!"}),
			gomock.Eq("mistral-embed")).
		Return(&mistralclient.EmbeddingResponse{
			Data: []mistralclient.EmbeddingData{},
		}, nil)

	p := mistral.NewPlugin("fake")
	p.Client = mockClient

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p))

	// When
	res, err := genkit.Embed(ctx, g,
		ai.WithDocs(
			ai.DocumentFromText("Hello, World!", nil),
		),
		ai.WithEmbedderName("mistral/mistral-embed"))

	// Then
	assert.Nil(t, res)
	assert.Equal(t, mistral.ErrNoEmbeddings, err)
}

func Test_Embedder_ShouldReturnFakeVectorFromFakeModel(t *testing.T) {
	// Given
	ctrl := gomock.NewController(t)
	mockClient := mocks.NewMockClient(ctrl)

	inputText := "Hello, World!"

	mockClient.EXPECT().
		TextEmbedding(gomock.Any(), gomock.Any(), gomock.Any()).
		Times(0)

	p := mistral.NewPlugin("fake")
	p.Client = mockClient

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p))

	// When
	res, err := genkit.Embed(ctx, g,
		ai.WithDocs(ai.DocumentFromText(inputText, nil)),
		ai.WithEmbedderName("mistral/fake-embed"))

	// Then
	assert.NoError(t, err)
	assert.Len(t, res.Embeddings, 1)
	assert.Len(t, res.Embeddings[0].Embedding, 1024)
}
