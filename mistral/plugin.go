package mistral

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const providerID = "mistral"

var (
	allModelsAndVersions = []modelInfo{
		{
			Name:     "mistral-medium",
			Versions: []string{"mistral-medium-latest", "mistral-medium-2505"},
		},
		{
			Name:     "ministral-3b",
			Versions: []string{"ministral-3b-2410", "ministral-3b-latest"},
		},
		{
			Name:     "ministral-8b",
			Versions: []string{"ministral-8b-2410", "ministral-8b-latest"},
		},
		{
			Name:     "mistral-tiny",
			Versions: []string{"open-mistral-7b", "mistral-tiny-2312", "mistral-tiny-latest", "open-mistral-nemo", "open-mistral-nemo-2407", "mistral-tiny-2407"},
		},
		{
			Name:     "mistral-small",
			Versions: []string{"open-mixtral-8x7b", "mistral-small-2312", "mistral-small-latest", "mistral-small-2409", "mistral-small-2501", "mistral-small-2503", "mistral-small-2506"},
		},
		{
			Name:     "open-mixtral-8x22b",
			Versions: []string{"open-mixtral-8x22b-2404", "open-mixtral-8x22b-latest"},
		},
		{
			Name:     "mistral-large",
			Versions: []string{"mistral-large-2407", "mistral-large-2411", "mistral-large-latest"},
		},
		{
			Name:     "pixtral-large",
			Versions: []string{"pixtral-large-2411", "pixtral-large-latest", "mistral-large-pixtral-2411"},
		},
		{
			Name:     "codestral",
			Versions: []string{"codestral-2501", "codestral-latest", "codestral-2412", "codestral-2411-rc5"},
		},
		{
			Name:     "devstral-small",
			Versions: []string{"devstral-small-2505", "devstral-small-latest"},
		},
		{
			Name:     "pixtral-12b",
			Versions: []string{"pixtral-12b-2409", "pixtral-12b-latest"},
		},
		{
			Name:     "mistral-saba",
			Versions: []string{"mistral-saba-2502", "mistral-saba-latest"},
		},
		{
			Name:     "magistral-medium",
			Versions: []string{"magistral-medium-2506", "magistral-medium-latest"},
		},
		{
			Name:     "magistral-small",
			Versions: []string{"magistral-small-2506", "magistral-small-latest"},
		},
	}

	allEmbeddingsAndVersions = []modelInfo{
		{
			Name:     "mistral-embed",
			Versions: []string{"mistral-embed"},
		},
	}
)

type Plugin struct {
	APIKey string
	Client *Client
	config *Config
}

func NewPlugin(apiKey string, opts ...Option) *Plugin {
	return &Plugin{
		APIKey: apiKey,
		config: NewConfig(opts...),
	}
}

func (p *Plugin) Name() string {
	return providerID
}

func (p *Plugin) Init(ctx context.Context, g *genkit.Genkit) error {
	c := newClientWithConfig(p.APIKey, p.config)
	p.Client = c
	for _, model := range allModelsAndVersions {
		defineModel(g, c, model.Name, model.Versions)
	}
	for _, model := range allEmbeddingsAndVersions {
		defineEmbedder(g, c, model.Name, model.Versions)
	}
	return nil
}

var _ genkit.Plugin = &Plugin{}

func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, providerID, name)
}

func ModelRef(name string, config *ModelConfig) ai.ModelRef {
	return ai.NewModelRef(name, config)
}
