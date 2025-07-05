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
}

func NewPlugin(apiKey string) *Plugin {
	return &Plugin{
		APIKey: apiKey,
	}
}

func (p *Plugin) Name() string {
	return providerID
}

func (p *Plugin) Init(ctx context.Context, g *genkit.Genkit) error {
	c := NewClient(p.APIKey, "mistral-large", "latest")
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
