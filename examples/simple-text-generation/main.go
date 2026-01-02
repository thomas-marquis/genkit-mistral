package main

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/thomas-marquis/genkit-mistral/mistral"
)

func main() {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	if apiKey == "" {
		panic("Please set MISTRAL_API_KEY environment variable")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(mistral.NewPlugin(apiKey)))

	res, err := genkit.Generate(ctx, g,
		ai.WithModelName("mistral/mistral-small-latest"),
		ai.WithSystem("you are a helpful assistant"),
		ai.WithPrompt("Tell me a joke"))

	if err != nil {
		panic(err)
	}
	println(res.Text())
}
