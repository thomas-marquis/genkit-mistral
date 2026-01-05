package mistral_test

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral"
	"github.com/thomas-marquis/genkit-mistral/mocks"
	mistralclient "github.com/thomas-marquis/mistral-client/mistral"
	"go.uber.org/mock/gomock"
)

func setupListModelWithChatCompletion(c *mocks.MockClient) {
	c.EXPECT().
		ListModels(gomock.Any()).
		Return([]*mistralclient.BaseModelCard{{
			Id: "mistral-small-latest",
			Capabilities: mistralclient.ModelCapabilities{
				CompletionChat: true,
			},
		}}, nil).
		AnyTimes()
}

func TestGenerate(t *testing.T) {
	t.Run("should return generated text", func(t *testing.T) {
		// Given
		ctrl := gomock.NewController(t)
		mockClient := mocks.NewMockClient(ctrl)

		setupListModelWithChatCompletion(mockClient)

		messages := []mistralclient.ChatMessage{
			mistralclient.NewSystemMessageFromString("You are a helpful assistant."),
			mistralclient.NewUserMessageFromString("Hello!"),
		}

		mockClient.EXPECT().
			ChatCompletion(
				gomock.AssignableToTypeOf(ctxType),
				gomock.Eq(&mistralclient.ChatCompletionRequest{
					CompletionConfig: mistralclient.CompletionConfig{
						ResponseFormat: &mistralclient.ResponseFormat{Type: "text"},
					},
					Model:    "mistral-small-latest",
					Messages: messages,
				}),
			).
			Return(&mistralclient.ChatCompletionResponse{
				Choices: []mistralclient.ChatCompletionChoice{
					{Message: mistralclient.NewAssistantMessageFromString("Hello simple human being!")},
				},
			}, nil)

		p := mistral.NewPlugin("fake", mistral.WithClient(mockClient))

		ctx := context.Background()
		g := genkit.Init(ctx, genkit.WithPlugins(p))

		// When
		res, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are a helpful assistant."),
			ai.WithPrompt("Hello!"),
			ai.WithModelName("mistral/mistral-small-latest"))

		// Then
		assert.NoError(t, err)
		assert.Equal(t, "Hello simple human being!", res.Text())
	})

	t.Run("should return generated text with documents", func(t *testing.T) {
		// Given
		ctrl := gomock.NewController(t)
		mockClient := mocks.NewMockClient(ctrl)

		setupListModelWithChatCompletion(mockClient)

		expectedUserMsg := `Hello!


Use the following information to complete your task:

- [0]: How to great as a Human?
- [1]: His name is YodaVery old, he is!

`

		mockClient.EXPECT().
			ChatCompletion(
				gomock.AssignableToTypeOf(ctxType),
				gomock.Cond(func(x *mistralclient.ChatCompletionRequest) bool {
					return assert.Equal(t, "mistral-small-latest", x.Model) &&
						assert.Equal(t, 2, len(x.Messages)) &&
						assert.Equal(t, mistralclient.NewSystemMessageFromString("You are a helpful assistant."), x.Messages[0]) &&
						assert.Equal(t, mistralclient.NewUserMessageFromString(expectedUserMsg), x.Messages[1])
				}),
			).
			Return(&mistralclient.ChatCompletionResponse{
				Choices: []mistralclient.ChatCompletionChoice{
					{Message: mistralclient.NewAssistantMessageFromString("Hello simple human being!")},
				},
			}, nil)

		p := mistral.NewPlugin("fake", mistral.WithClient(mockClient))

		ctx := context.Background()
		g := genkit.Init(ctx, genkit.WithPlugins(p))

		// When
		res, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are a helpful assistant."),
			ai.WithPrompt("Hello!"),
			ai.WithDocs(
				ai.DocumentFromText("How to great as a Human?", nil),
				&ai.Document{
					Content: []*ai.Part{
						ai.NewTextPart("His name is Yoda"),
						ai.NewTextPart("Very old, he is!"),
					},
				},
			),
			ai.WithModelName("mistral/mistral-small-latest"))

		// Then
		assert.NoError(t, err)
		assert.Equal(t, "Hello simple human being!", res.Text())
	})

	t.Run("should return error when no message provided", func(t *testing.T) {
		// Given
		ctrl := gomock.NewController(t)
		mockClient := mocks.NewMockClient(ctrl)

		setupListModelWithChatCompletion(mockClient)

		mockClient.EXPECT().
			ChatCompletion(
				gomock.Any(),
				gomock.Any(),
			).
			Times(0)

		p := mistral.NewPlugin("fake", mistral.WithClient(mockClient))

		ctx := context.Background()
		g := genkit.Init(ctx, genkit.WithPlugins(p))

		// When
		res, err := genkit.Generate(ctx, g,
			ai.WithModelName("mistral/mistral-small-latest"))

		// Then
		assert.Nil(t, res)
		assert.ErrorIs(t, err, mistral.ErrInvalidModelInput)
	})
}
