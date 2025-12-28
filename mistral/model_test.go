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

func Test_Generate_ShouldReturnGeneratedText(t *testing.T) {
	// Given
	ctrl := gomock.NewController(t)
	mockClient := mocks.NewMockClient(ctrl)

	messages := []mistralclient.ChatMessage{
		mistralclient.NewSystemMessageFromString("You are a helpful assistant."),
		mistralclient.NewUserMessageFromString("Hello!"),
	}

	mockClient.EXPECT().
		ChatCompletion(
			gomock.AssignableToTypeOf(ctxType),
			gomock.Eq(&mistralclient.ChatCompletionRequest{
				Model:    "mistral-small-latest",
				Messages: messages,
			}),
		).
		Return(mistralclient.ChatCompletionResponse{
			Choices: []mistralclient.ChatCompletionChoice{
				{Message: mistralclient.NewAssistantMessageFromString("Hello simple human being!")},
			},
		}, nil)

	p := mistral.NewPlugin("fake")
	p.Client = mockClient

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
}

func Test_Generate_ShouldReturnGeneratedTextWithDocuments(t *testing.T) {
	// Given
	ctrl := gomock.NewController(t)
	mockClient := mocks.NewMockClient(ctrl)

	expectedUserMsg := `Hello!


Use the following information to complete your task:

- [0]: How to great as a Human?
- [1]: His name is YodaVery old, he is!

`

	mockClient.EXPECT().
		ChatCompletion(
			gomock.AssignableToTypeOf(ctxType),
			gomock.Cond(func(x *mistralclient.ChatCompletionRequest) bool {
				return x.Model == "mistral-small-latest" &&
					len(x.Messages) == 2 &&
					x.Messages[0] == mistralclient.NewSystemMessageFromString("You are a helpful assistant.") &&
					x.Messages[1] == mistralclient.NewUserMessageFromString(expectedUserMsg)
			}),
		).
		Return(mistralclient.ChatCompletionResponse{
			Choices: []mistralclient.ChatCompletionChoice{
				{Message: mistralclient.NewAssistantMessageFromString("Hello simple human being!")},
			},
		}, nil)

	p := mistral.NewPlugin("fake")
	p.Client = mockClient

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
}

func Test_Generate_ShouldReturnErrorWhenNoMessageProvided(t *testing.T) {
	// Given
	ctrl := gomock.NewController(t)
	mockClient := mocks.NewMockClient(ctrl)

	mockClient.EXPECT().
		ChatCompletion(
			gomock.Any(),
			gomock.Any(),
		).
		Times(0)

	p := mistral.NewPlugin("fake")
	p.Client = mockClient

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(p))

	// When
	res, err := genkit.Generate(ctx, g,
		ai.WithModelName("mistral/mistral-small-latest"))

	// Then
	assert.Nil(t, res)
	assert.ErrorIs(t, err, mistral.ErrInvalidModelInput)
}
