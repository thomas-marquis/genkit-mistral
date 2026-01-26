# genkit-mistral ✨

Enables Mistral AI support for Firebase Genkit (Go SDK). Build agents and apps with Genkit while calling Mistral models and embeddings with a clean, typed Go API.

[![Go Reference](https://pkg.go.dev/badge/github.com/thomas-marquis/genkit-mistral.svg)](https://pkg.go.dev/github.com/thomas-marquis/genkit-mistral)
[![CI](https://github.com/thomas-marquis/genkit-mistral/actions/workflows/ci.yml/badge.svg)](https://github.com/thomas-marquis/genkit-mistral/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENCE)

<p align="center">
  <img src="resources/assets/logo-tr.png" width="350" alt="genkit-mistral logo">
</p>

## Quick start 🚀

Install (Genkit + this plugin):

```bash
go get github.com/firebase/genkit/go
go get github.com/thomas-marquis/genkit-mistral
```

## Usage

See the full API and more examples in the Go reference: https://pkg.go.dev/github.com/thomas-marquis/genkit-mistral

Check the [Genkit's documentation](https://genkit.dev/go/docs/models/) for more insights.

Genkit-mistral is just a Genkit plugin. So, it doesn't change the way you use Genkit (and you can change the plugin without modifying your code).
The plugin-specific part is visible only in the `NewPlugin` function, during initialization.
Optionally, some options can be passed to this function:
- `WithClient`, if you want to use a custom HTTP client (that implements the `Client` interface from `mistral-client`).
- `WithAPICallsDisabled`, for testing purposes. Only fake models provided by `mistral-client` are available. No need to provide a valid API key.
- `WithClientOptions`, if you want to customize the HTTP client. Available options are documented [here](https://pkg.go.dev/github.com/thomas-marquis/mistral-client@v0.3.0/mistral#Option).

Some usage examples can be found [here](https://github.com/thomas-marquis/genkit-examples) and in the current repo's `/examples` folder.

### Basit text generation

```go
package main

import (
    "context"
    "fmt"
    "os"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/ai"
    "github.com/thomas-marquis/genkit-mistral/mistral"
)

func main() {
	mistralApiKey := os.Getenv("MISTRAL_API_KEY")
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(
			mistral.NewPlugin(mistralApiKey),
		),
		genkit.WithDefaultModel("mistral/mistral-small-latest"),
	)

	res, err := genkit.Generate(ctx, g,
		ai.WithSystem("you are a helpful assistant"),
		ai.WithPrompt("Tell me a joke"),
	)
	if err != nil {
		panic(err)
	}
    fmt.Println(res.Text())
}
```

### Text embedding

```go
package main

import (
    "context"
    "fmt"
    "os"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/ai"
    "github.com/thomas-marquis/genkit-mistral/mistral"
)

func main() {
	mistralApiKey := os.Getenv("MISTRAL_API_KEY")
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(
			mistral.NewPlugin(mistralApiKey),
		),
	)

	docToEmbed := ai.DocumentFromText("Is scribe a good situation?", nil)
	res, err := genkit.Embed(ctx, g,
		ai.WithDocs(docToEmbed),
		ai.WithEmbedderName("mistral/mistral-embed"),
	)
	if err != nil {
		panic(err)
	}
    fmt.Println(res.Embeddings[0].Embedding)
}
```

### Output format constrained

```go
package main

import (
    "context"
    "fmt"
    "os"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/ai"
    "github.com/thomas-marquis/genkit-mistral/mistral"
)

func main() {
	mistralApiKey := os.Getenv("MISTRAL_API_KEY")
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(
			mistral.NewPlugin(mistralApiKey),
		),
		genkit.WithDefaultModel("mistral/mistral-small-latest"),
	)

	type expectedOutput struct {
		JokeContent string `json:"joke_content"`
		LolLevel    int    `json:"lol_level"`
	}

	res, err := genkit.Generate(ctx, g,
		ai.WithSystem("you are a helpful assistant"),
		ai.WithPrompt("Tell me a joke"),
		ai.WithOutputType(expectedOutput{}),
	)

	if err != nil {
		panic(err)
    }
	
    var joke expectedOutput
    if err := res.Output(&joke); err != nil {
        fmt.Printf("Failed to parse output: %s\n", err)
    } 
    fmt.Printf("Is this \"%s\" really level %d???!!\n", joke.JokeContent, joke.LolLevel)
}
```

### Use fake models (for testing or local development)

These two fake models are available:
- `mistral/fake-completiont`: return some Lorem Ipsum text
- `mistral/fake-embed`: return random embedding vectors

Why use fake models?
- For integration tests, when what you want to test does not depend on the actual result of the model
- For local development, when you just want to know if your application starts or runs correctly

## Models and embeddings 🧠

You can find all tes mistral models with this command:

```bash
curl --request GET \
  --url https://api.mistral.ai/v1/models \
  --header 'Authorization: Bearer <your API token>'
```

This library is built on top of [mistral-client](https://github.com/thomas-marquis/mistral-client).
A method `ListModels` is available from this one to list all available models.

## Useful resources 📚

| Resource       | For what?                            | Link(s)                                                                                                      |
|----------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Mistral AI     | French AI provider                   | [Website](https://mistral.ai/) <br/> [API documentation](https://docs.mistral.ai/api/)                       |
| Genkit         | Agentic framework (Go, Node.js, Python) | [Docs](https://firebase.google.com/docs/genkit) <br/> [Go SDK](https://genkit.dev/go/docs/get-started-go/)   |
| La Plateforme  | Mistral's cloud platform for developers | [Website](https://mistral.ai/products/la-plateforme) <br/> [Pricing](https://mistral.ai/pricing#api-pricing) |
| mistral-client | HTTP client for Mistral AI written in Go. | [Repo](https://github.com/thomas-marquis/mistral-client) <br/> [Documentation](https://thomas-marquis.github.io/mistral-client/)                                                                             |

## Contributing 🤝

Contributions are welcome! Please open an issue or a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.