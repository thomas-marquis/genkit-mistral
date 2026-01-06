package mistral

import (
	"context"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/thomas-marquis/mistral-client/mistral"
)

const providerID = "mistral"

type Plugin struct {
	sync.Mutex

	APIKey string
	Client mistral.Client

	apiCallsDisabled bool
}

type Option func(plugin *Plugin)

// WithClient sets the client to use for the plugin.
// For exotic use case, you can define your own mistral.Client implementation with this option.
// Don't use it with WithClientOptions.
func WithClient(client mistral.Client) Option {
	return func(p *Plugin) {
		p.Client = client
	}
}

// WithAPICallsDisabled disables all API calls to Mistral.
// With this option, you don't need to provide a real API key
// therefore, you can now only use the fake models.
//
// For test or dev purposes only
func WithAPICallsDisabled() Option {
	return func(p *Plugin) {
		p.apiCallsDisabled = true
	}
}

// WithClientOptions sets the options to use for the client (timeout, custom transport...).
// Don't use it with WithClient.
func WithClientOptions(opts ...mistral.Option) Option {
	return func(p *Plugin) {
		p.Client = mistral.New(p.APIKey, opts...)
	}
}

func NewPlugin(apiKey string, opts ...Option) *Plugin {
	p := &Plugin{
		APIKey: apiKey,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

func (p *Plugin) Name() string {
	return providerID
}

func (p *Plugin) Init(ctx context.Context) []api.Action {
	if p.Client == nil {
		p.Client = mistral.New(p.APIKey)
	}

	var err error
	var mistralModels []*mistral.BaseModelCard
	if !p.apiCallsDisabled {
		mistralModels, err = p.Client.ListModels(ctx)
		if err != nil {
			panic(err)
		}
	}

	p.Lock()
	defer p.Unlock()

	var actions []api.Action
	modelSet := make(map[string]struct{})

	for _, card := range mistralModels {
		if _, ok := modelSet[card.Id]; !ok {
			if !card.IsEmbedding() {
				model := defineModel(p.Client, mapCardToModelInfo(card))
				actions = append(actions, model.(api.Action))
			} else {
				actions = append(actions, defineEmbedder(p.Client, card.Id).(api.Action))
			}
			modelSet[card.Id] = struct{}{}
		}
	}
	actions = append(actions, defineFakeModel().(api.Action))
	actions = append(actions, defineFakeEmbedder().(api.Action))

	return actions
}

var _ api.Plugin = &Plugin{}

func mapCardToModelInfo(card *mistral.BaseModelCard) *ai.ModelInfo {
	stage := ai.ModelStageStable
	if !card.Deprecation.IsZero() && card.Deprecation.After(time.Now()) {
		stage = ai.ModelStageDeprecated
	}

	return &ai.ModelInfo{
		Label: card.Id,
		Stage: stage,
		Supports: &ai.ModelSupports{
			Constrained: ai.ConstrainedSupportAll,
			Context:     false,
			Media:       card.Capabilities.Vision || card.Capabilities.Audio,
			Multiturn:   card.Capabilities.CompletionChat,
			SystemRole:  card.Capabilities.CompletionChat,
			ToolChoice:  card.Capabilities.FunctionCalling,
			Tools:       card.Capabilities.FunctionCalling,
		},
		Versions: card.Aliases,
	}
}
