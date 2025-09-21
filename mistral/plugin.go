package mistral

import (
	"context"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

const providerID = "mistral"

var (
	llmModels = map[string]ai.ModelInfo{
		"mistral-medium-2505": {
			Label: "mistral-medium-2505",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-medium-2505"},
		},
		"mistral-large-latest": {
			Label: "mistral-large-latest",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-large-latest"},
		},
		"mistral-medium-2508": {
			Label: "mistral-medium-2508",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-medium-2508", "mistral-medium-latest"},
		},
		"ministral-3b-2410": {
			Label: "ministral-3b-2410",
			Stage: ai.ModelStageLegacy,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   false,
				SystemRole:  false,
				ToolChoice:  false,
				Tools:       false,
			},
			Versions: []string{"ministral-3b-2410", "ministral-3b-latest"},
		},
		"ministral-8b-2410": {
			Label: "ministral-8b-2410",
			Stage: ai.ModelStageLegacy,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"ministral-8b-2410", "ministral-8b-latest"},
		},
		"open-mistral-7b": {
			Label: "open-mistral-7b",
			Stage: ai.ModelStageLegacy,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"open-mistral-7b", "mistral-tiny", "mistral-tiny-2312"},
		},
		"open-mistral-nemo": {
			Label: "open-mistral-nemo",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"open-mistral-nemo", "open-mistral-nemo-2407", "mistral-tiny-2407", "mistral-tiny-latest"},
		},
		"open-mixtral-8x7b": {
			Label: "open-mixtral-8x7b",
			Stage: ai.ModelStageLegacy,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"open-mixtral-8x7b", "mistral-small", "mistral-small-2312"},
		},
		"open-mixtral-8x22b": {
			Label: "open-mixtral-8x22b",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"open-mixtral-8x22b", "open-mixtral-8x22b-2404"},
		},
		"mistral-small-2409": {
			Label: "mistral-small-2409",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-small-2409"},
		},
		"mistral-large-2407": {
			Label: "mistral-large-2407",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-large-2407"},
		},
		"mistral-large-2411": {
			Label: "mistral-large-2411",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-large-2411"},
		},
		"pixtral-large-2411": {
			Label: "pixtral-large-2411",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"pixtral-large-2411", "pixtral-large-latest", "mistral-large-pixtral-2411"},
		},
		"codestral-2501": {
			Label: "codestral-2501",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"codestral-2501", "codestral-2412", "codestral-2411-rc5"},
		},
		"codestral-2508": {
			Label: "codestral-2508",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"codestral-2508", "codestral-latest"},
		},
		"devstral-small-2505": {
			Label: "devstral-small-2505",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"devstral-small-2505"},
		},
		"devstral-small-2507": {
			Label: "devstral-small-2507",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"devstral-small-2507", "devstral-small-latest"},
		},
		"devstral-medium-2507": {
			Label: "devstral-medium-2507",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"devstral-medium-2507", "devstral-medium-latest"},
		},
		"pixtral-12b-2409": {
			Label: "pixtral-12b-2409",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"pixtral-12b-2409", "pixtral-12b", "pixtral-12b-latest"},
		},
		"mistral-small-2501": {
			Label: "mistral-small-2501",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-small-2501"},
		},
		"mistral-small-2503": {
			Label: "mistral-small-2503",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-small-2503"},
		},
		"mistral-small-2506": {
			Label: "mistral-small-2506",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-small-2506", "mistral-small-latest"},
		},
		"mistral-saba-2502": {
			Label: "mistral-saba-2502",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-saba-2502", "mistral-saba-latest"},
		},
		"magistral-medium-2506": {
			Label: "magistral-medium-2506",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"magistral-medium-2506"},
		},
		"magistral-medium-2507": {
			Label: "magistral-medium-2507",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"magistral-medium-2507", "magistral-medium-latest"},
		},
		"magistral-small-2506": {
			Label: "magistral-small-2506",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"magistral-small-2506"},
		},
		"magistral-small-2507": {
			Label: "magistral-small-2507",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"magistral-small-2507", "magistral-small-latest"},
		},
		"voxtral-mini-2507": {
			Label: "voxtral-mini-2507",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  false,
				Tools:       false,
			},
			Versions: []string{"voxtral-mini-2507", "voxtral-mini-latest"},
		},
		"voxtral-small-2507": {
			Label: "voxtral-small-2507",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"voxtral-small-2507", "voxtral-small-latest"},
		},

		"mistral-moderation-2411": {
			Label: "mistral-moderation-2411",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       false,
				Multiturn:   false,
				SystemRole:  false,
				ToolChoice:  false,
				Tools:       false,
			},
			Versions: []string{"mistral-moderation-2411", "mistral-moderation-latest"},
		},
		"mistral-ocr-2503": {
			Label: "mistral-ocr-2503",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-ocr-2503"},
		},
		"mistral-ocr-2505": {
			Label: "mistral-ocr-2505",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  true,
				Tools:       true,
			},
			Versions: []string{"mistral-ocr-2505", "mistral-ocr-latest"},
		},
		"voxtral-mini-transcribe-2507": {
			Label: "voxtral-mini-transcribe-2507",
			Stage: ai.ModelStageStable,
			Supports: &ai.ModelSupports{
				Constrained: ai.ConstrainedSupportAll,
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				ToolChoice:  false,
				Tools:       false,
			},
			Versions: []string{"voxtral-mini-transcribe-2507"},
		},
	}

	embeddingModels = []string{
		"mistral-embed",
		"codestral-embed",
	}
)

type Plugin struct {
	sync.Mutex

	APIKey string
	Client mistralclient.Client

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

func (p *Plugin) Init(ctx context.Context) []api.Action {
	if p.Client == nil {
		p.Client = mistralclient.NewClientWithConfig(p.APIKey, &p.config.Client)
	}

	p.Lock()
	defer p.Unlock()

	var actions []api.Action

	for name, info := range llmModels {
		models := defineModel(p.Client, name, info)
		for _, model := range models {
			actions = append(actions, model.(api.Action))
		}
	}
	actions = append(actions, defineFakeModel().(api.Action))

	for _, name := range embeddingModels {
		actions = append(actions, defineEmbedder(p.Client, name).(api.Action))
	}
	actions = append(actions, defineFakeEmbedder().(api.Action))

	return actions
}

var _ api.Plugin = &Plugin{}
