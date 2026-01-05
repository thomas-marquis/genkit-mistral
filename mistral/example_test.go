package mistral_test

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/thomas-marquis/genkit-mistral/mistral"
)

func ExampleNewPlugin_initPluginAndGenerateText() {
	mistralApiKey := "your_api_key"
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(
			mistral.NewPlugin(mistralApiKey,
				mistral.WithAPICallsDisabled(), // required for this example, don't use this option in production
			),
		),
		genkit.WithDefaultModel("mistral/fake-completion"),
	)

	res, err := genkit.Generate(ctx, g,
		ai.WithSystem("you are a helpful assistant"),
		ai.WithPrompt("Tell me a joke"),
	)

	if err == nil {
		fmt.Printf("Generation succeeded, AI respond with role %s\n", res.Message.Role)
	} else {
		fmt.Printf("Generation failed with reason %s\n", err)
	}
	// Output:
	// Generation succeeded, AI respond with role model
}

func ExampleNewPlugin_underConstraintTextGenerationWithAMockedModel() {
	// This example may be useful if you want to mock a model for unit testing purpose.
	// Please notice that you can do this with all model providers.

	// We start by initializing genkit, as usual
	mistralApiKey := "your_api_key"
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(
			mistral.NewPlugin(mistralApiKey,
				mistral.WithAPICallsDisabled(), // required for this example, don't use this option in production
			),
		),
		genkit.WithDefaultModel("mistral/fake-completion"),
	)

	// Then, we need to create a mock model under a custom provider namespace (here: myapp)
	genkit.DefineModel(g, "myapp/mock-completion",
		&ai.ModelOptions{
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll, // in our example, we want the "model" supports constrained generation
				Multiturn:   true,
			},
		},
		func(ctx context.Context, request *ai.ModelRequest, s core.StreamCallback[*ai.ModelResponseChunk]) (*ai.ModelResponse, error) {
			// Here is the function that will be called when the model is called.
			return &ai.ModelResponse{
				Message: ai.NewModelMessage(ai.NewJSONPart(`{"joke_content": "le mec à un phare, il s'appelle On, ....", "lol_level": 10000000000}`)),
			}, nil
		})

	type expectedOutput struct {
		JokeContent string `json:"joke_content"`
		LolLevel    int    `json:"lol_level"`
	}

	res, err := genkit.Generate(ctx, g,
		ai.WithSystem("you are a helpful assistant"),
		ai.WithPrompt("Tell me a joke"),
		ai.WithOutputType(expectedOutput{}),
		ai.WithModelName("myapp/mock-completion"), // we can override the default model here
	)

	if err == nil {
		var joke expectedOutput
		if err := res.Output(&joke); err != nil {
			fmt.Printf("Failed to parse output: %s\n", err)
		} else {
			fmt.Printf("Is this \"%s\" really level %d???!!\n", joke.JokeContent, joke.LolLevel)
		}
	} else {
		fmt.Printf("Generation failed with reason %s\n", err)
	}
	// Output:
	// Is this "le mec à un phare, il s'appelle On, ...." really level 10000000000???!!
}

func ExampleNewPlugin_initPluginAndComputeAnEmbedding() {
	mistralApiKey := "your_api_key"
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(
			mistral.NewPlugin(mistralApiKey,
				mistral.WithAPICallsDisabled(), // required for this example, don't use this option in production
			),
		),
	)

	docToEmbed := ai.DocumentFromText("Is scribe a good situation?", nil)
	res, err := genkit.Embed(ctx, g,
		ai.WithDocs(docToEmbed),
		ai.WithEmbedderName("mistral/fake-embed"),
	)

	if err == nil {
		fmt.Printf("Embedding succeeded with vector length %d\n", len(res.Embeddings[0].Embedding))
	} else {
		fmt.Printf("Embedding failed with reason %s\n", err)
	}
	// Output:
	// Embedding succeeded with vector length 1024
}
