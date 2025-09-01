# genkit-mistral ✨

Enables Mistral AI support for Firebase Genkit (Go SDK). Build agents and apps with Genkit while calling Mistral models and embeddings with a clean, typed Go API.

[![Go Reference](https://pkg.go.dev/badge/github.com/thomas-marquis/genkit-mistral.svg)](https://pkg.go.dev/github.com/thomas-marquis/genkit-mistral) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENCE)

## Quick start 🚀

Install (Genkit + this plugin):

```bash
go get github.com/firebase/genkit/go
go get github.com/thomas-marquis/genkit-mistral
```

Set your Mistral API key (for example in your shell):

```bash
export MISTRAL_API_KEY=your_api_key_here
```

Minimal usage with the low-level client (straightforward and great for scripts):

```go
package main

import (
    "context"
    "fmt"
    "os"
    "github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func main() {
    apiKey := getenv("MISTRAL_API_KEY", "")
    client := mistralclient.NewClient(apiKey,
        mistralclient.WithVerbose(true),
        mistralclient.WithRetry(3, 0, 0), // sensible defaults
    )

    ctx := context.Background()
    msg := []mistralclient.Message{
        mistralclient.NewSystemMessage("You are a helpful assistant."),
        mistralclient.NewHumanMessage("Say hello in one short sentence."),
    }

    resp, err := client.ChatCompletion(ctx, msg, "mistral-small-latest", &mistralclient.ModelConfig{
        Temperature: 0.2,
    }, mistralclient.WithResponseTextFormat())
    if err != nil { panic(err) }

    fmt.Println(resp.Content)
}

func getenv(k, def string) string { if v := os.Getenv(k); v != "" { return v }; return def }
```

Using with Genkit (high-level) in your app: you create and init the plugin, then use models/embedders by name. Full Genkit samples live in the package docs, but the essence is:

```go
// inside your setup/bootstrap code
p := mistral.NewPlugin(os.Getenv("MISTRAL_API_KEY"))
if err := p.Init(ctx, genkitInstance); err != nil { panic(err) }

// Later, use a model by name (e.g., "mistral-small-latest") via Genkit's APIs.
```

See the full API and more examples in the Go reference: https://pkg.go.dev/github.com/thomas-marquis/genkit-mistral

## Models and embeddings 🧠

- Chat/completions: e.g. `mistral-small`, `mistral-large`, `codestral`, etc. (with `-latest` or versioned variants)
- Embeddings: `mistral-embed`

The client supports:
- Configurable retries with exponential backoff and jitter
- Pluggable rate limiting (token bucket)
- JSON schema-constrained outputs (via `WithResponseJsonSchema`)

## Useful resources 📚

| Resource      | For what?                                      | Link(s)                                                                                                             |
|---------------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Mistral AI    | French AI provider                              | [Website](https://mistral.ai/) <br/> [API documentation](https://docs.mistral.ai/api/)                              |
| La Plateforme | Mistral's cloud platform for developers         | [Website](https://mistral.ai/products/la-plateforme) <br/> [Pricing](https://mistral.ai/pricing#api-pricing)        |
| Genkit        | Agentic framework (Go, Node.js, Python)         | [Docs](https://firebase.google.com/docs/genkit) <br/> [Go SDK](https://genkit.dev/go/docs/get-started-go/)          |

## Contributing 🤝

### Code structure

- mistral/ — Genkit plugin wiring (models, embedders, mapping between Genkit and Mistral)
- mistralclient/ — Low-level HTTP client for Mistral (chat, embeddings, retries, rate limiting)
- internal/ — Small helpers and test fixtures
- scripts/ — Local CI helpers

### Running tests

- Package tests:
  - `go test ./internal ./mistralclient`
- Note: running `go test ./...` from the repo root may fail due to a known import cycle between `mistral` and `mistralclient` tests. Test packages individually as above.

### Updating CI locally

Install local tooling (GitHub Actions runner via `act`):

```bash
./scripts/install_act.sh
```

Run the workflow linter:

```bash
actionlint
```

Run a workflow locally:

```bash
./scripts/run_ci_local.sh <workflow_name>
```

— Happy hacking! 🛠️