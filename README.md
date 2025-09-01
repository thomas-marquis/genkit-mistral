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

Check the [Genkit's documentation](https://genkit.dev/go/docs/models/) for more insights.

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

Supported chat/completions models (family name with versions/aliases):
- mistral-medium (mistral-medium-2505, mistral-medium-2508, mistral-medium-latest)
- mistral-large (mistral-large-2407, mistral-large-2411, mistral-large-latest)
- mistral-small (mistral-small-2312, mistral-small-2409, mistral-small-2501, mistral-small-2503, mistral-small-2506, mistral-small-latest, open-mixtral-8x7b)
- mistral-tiny (mistral-tiny-2312, mistral-tiny-2407, mistral-tiny-latest, open-mistral-7b, open-mistral-nemo, open-mistral-nemo-2407)
- open-mixtral-8x22b (open-mixtral-8x22b, open-mixtral-8x22b-2404)
- pixtral-large (pixtral-large-2411, pixtral-large-latest, mistral-large-pixtral-2411)
- pixtral-12b (pixtral-12b-2409, pixtral-12b, pixtral-12b-latest)
- codestral (codestral-2411-rc5, codestral-2412, codestral-2501, codestral-2508, codestral-latest)
- devstral-small (devstral-small-2505, devstral-small-2507, devstral-small-latest)
- devstral-medium (devstral-medium-2507, devstral-medium-latest)
- magistral-medium (magistral-medium-2506, magistral-medium-2507, magistral-medium-latest)
- magistral-small (magistral-small-2506, magistral-small-2507, magistral-small-latest)
- mistral-saba (mistral-saba-2502, mistral-saba-latest)
- voxtral-mini (voxtral-mini-2507, voxtral-mini-latest)
- voxtral-small (voxtral-small-2507, voxtral-small-latest)
- mistral-moderation (mistral-moderation-2411, mistral-moderation-latest)
- mistral-ocr (mistral-ocr-2503, mistral-ocr-2505, mistral-ocr-latest)
- voxtral-mini-transcribe (voxtral-mini-transcribe-2507)


Supported embedders:
- mistral-embed
- codestral-embed

You can find all tes mistral models with this command:

```bash
curl --request GET \
  --url https://api.mistral.ai/v1/models \
  --header 'Authorization: Bearer <your API token>'
```


## Useful resources 📚

| Resource      | For what?                                      | Link(s)                                                                                                             |
|---------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Mistral AI    | French AI provider                              | [Website](https://mistral.ai/) <br/> [API documentation](https://docs.mistral.ai/api/)                              |
| Genkit        | Agentic framework (Go, Node.js, Python)         | [Docs](https://firebase.google.com/docs/genkit) <br/> [Go SDK](https://genkit.dev/go/docs/get-started-go/)          |
| La Plateforme | Mistral's cloud platform for developers         | [Website](https://mistral.ai/products/la-plateforme) <br/> [Pricing](https://mistral.ai/pricing#api-pricing)        |

## Contributing 🤝

Contributions are welcome! Please open an issue or a PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.