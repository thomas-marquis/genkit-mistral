package mistralclient

type UsageInfo struct {
	CompletionTokens   int `json:"completion_tokens"`
	PromptAudioSeconds int `json:"prompt_audio_seconds"`
	PromptTokens       int `json:"prompt_tokens"`
	TotalTokens        int `json:"total_tokens"`
}
