package mistral_test

import (
	"context"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral"
	"github.com/thomas-marquis/genkit-mistral/mocks"
	mistralclient "github.com/thomas-marquis/mistral-client/mistral"
	"go.uber.org/mock/gomock"
)

var (
	ctxType = reflect.TypeOf((*context.Context)(nil)).Elem()
)

func setupListModelWithEmbedding(c *mocks.MockClient) {
	c.EXPECT().
		ListModels(gomock.Any()).
		Return([]*mistralclient.BaseModelCard{{
			Id: "mistral-embed",
		}}, nil).
		AnyTimes()
}

func TestEmbed(t *testing.T) {
	t.Run("should return a single vector", func(t *testing.T) {
		// Given
		ctrl := gomock.NewController(t)
		mockClient := mocks.NewMockClient(ctrl)

		inputText := "Hello, World!"
		expectedVec := []float32{1, 2, 3}

		setupListModelWithEmbedding(mockClient)

		mockClient.EXPECT().
			Embeddings(
				gomock.AssignableToTypeOf(ctxType),
				gomock.Eq(&mistralclient.EmbeddingRequest{Model: "mistral-embed", Input: []string{inputText}}),
			).
			Return(&mistralclient.EmbeddingResponse{
				Data: []mistralclient.EmbeddingData{
					{
						Embedding: expectedVec,
					},
				},
			}, nil)

		p := mistral.NewPlugin("fake", mistral.WithClient(mockClient))

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
	})

	t.Run("should return multiple vectors", func(t *testing.T) {
		// Given
		ctrl := gomock.NewController(t)
		mockClient := mocks.NewMockClient(ctrl)

		setupListModelWithEmbedding(mockClient)

		expectedVec1 := []float32{1, 2, 3}
		expectedVec2 := []float32{4, 5, 6}

		mockClient.EXPECT().
			Embeddings(
				gomock.AssignableToTypeOf(ctxType),
				gomock.Eq(&mistralclient.EmbeddingRequest{
					Model: "mistral-embed",
					Input: []string{"Hello, World!", "Hello there!\nMy name is Obi-Wan Kenobi."},
				})).
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

		p := mistral.NewPlugin("fake", mistral.WithClient(mockClient))

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
	})

	t.Run("should return error when no vector is returned", func(t *testing.T) {
		// Given
		ctrl := gomock.NewController(t)
		mockClient := mocks.NewMockClient(ctrl)

		setupListModelWithEmbedding(mockClient)

		mockClient.EXPECT().
			Embeddings(
				gomock.AssignableToTypeOf(ctxType),
				gomock.Eq(&mistralclient.EmbeddingRequest{Model: "mistral-embed", Input: []string{"Hello, World!"}}),
			).
			Return(&mistralclient.EmbeddingResponse{
				Data: []mistralclient.EmbeddingData{},
			}, nil)

		p := mistral.NewPlugin("fake", mistral.WithClient(mockClient))

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
	})

	t.Run("should return fake vector from fake model", func(t *testing.T) {
		// Given
		ctrl := gomock.NewController(t)
		mockClient := mocks.NewMockClient(ctrl)

		inputText := "Hello, World!"

		setupListModelWithEmbedding(mockClient)

		mockClient.EXPECT().
			Embeddings(gomock.Any(), gomock.Any()).
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
	})
}
