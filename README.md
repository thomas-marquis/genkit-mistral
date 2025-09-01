# genkit-mistral ✨

Enables Mistral AI support for Firebase Genkit (Go SDK). Build agents and apps with Genkit while calling Mistral models and embeddings with a clean, typed Go API.

[![Go Reference](https://pkg.go.dev/badge/github.com/thomas-marquis/genkit-mistral.svg)](https://pkg.go.dev/github.com/thomas-marquis/genkit-mistral) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENCE)

## Quick start 🚀

Install (Genkit + this plugin):

```bash
go get github.com/firebase/genkit/go
go get github.com/thomas-marquis/genkit-mistral
```

## Usage

See the full API and more examples in the Go reference: https://pkg.go.dev/github.com/thomas-marquis/genkit-mistral


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


## Models and embeddings 🧠



## Useful resources 📚

| Resource      | For what?                                      | Link(s)                                                                                                             |
|---------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Mistral AI    | French AI provider                              | [Website](https://mistral.ai/) <br/> [API documentation](https://docs.mistral.ai/api/)                              |
| Genkit        | Agentic framework (Go, Node.js, Python)         | [Docs](https://firebase.google.com/docs/genkit) <br/> [Go SDK](https://genkit.dev/go/docs/get-started-go/)          |
| La Plateforme | Mistral's cloud platform for developers         | [Website](https://mistral.ai/products/la-plateforme) <br/> [Pricing](https://mistral.ai/pricing#api-pricing)        |

## Contributing 🤝

Contributions are welcome! Please open an issue or a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.