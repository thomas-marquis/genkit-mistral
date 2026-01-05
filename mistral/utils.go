package mistral

import (
	"log"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/thomas-marquis/genkit-mistral/mistral/internal/mapping"
)

var (
	logger = log.New(os.Stdout, "mistral-client: ", log.LstdFlags|log.Lshortfile)
)

// StringFromParts returns the content of a multi-parts message as a string.
// The multiple parts are concatenated with a newline character.
func StringFromParts(content []*ai.Part) string {
	msg, err := mapping.StringFromParts(content)
	if err != nil {
		logger.Printf("Failed to convert message content to string: %v\n", err)
	}

	return msg
}

// SanitizeToolName formats a function name to be used as a reference in a tool call.
func SanitizeToolName(name string) string {
	runes := []rune(name)

	isAllowed := func(r rune) bool {
		return (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '_' || r == '-'
	}

	var b strings.Builder
	b.Grow(len(runes))

	for i := 0; i < len(runes); {
		r := runes[i]
		switch {
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_':
			b.WriteRune(r)
			i++
		case r == '-':
			// If pattern is "-<disallowed>-" convert to "_-" (skip the middle disallowed runes)
			j := i + 1
			skipped := 0
			for j < len(runes) && !isAllowed(runes[j]) {
				j++
				skipped++
			}
			if j < len(runes) && runes[j] == '-' && skipped > 0 {
				b.WriteRune('_')
				b.WriteRune('-')
				i = j + 1
			} else {
				b.WriteRune('-')
				i++
			}
		default:
			// drop any other disallowed characters
			i++
		}
	}

	result := b.String()
	if len(result) > 256 {
		result = result[:256]
	}
	return result
}

func mapResponseFromText(mr *ai.ModelRequest, resp string) *ai.ModelResponse {
	return &ai.ModelResponse{
		Request: mr,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(resp)},
		},
	}
}
