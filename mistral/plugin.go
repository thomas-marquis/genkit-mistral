package mistral

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const providerID = "mistral"

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
	c := newClientWithConfig(p.APIKey, "mistral-large", "latest", p.config)
	p.Client = c
	defineModel(g, c)
	return nil
}

var _ genkit.Plugin = &Plugin{}

func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, providerID, name)
}

func ModelRef(name string, config *ModelConfig) ai.ModelRef {
	return ai.NewModelRef(name, config)
}
