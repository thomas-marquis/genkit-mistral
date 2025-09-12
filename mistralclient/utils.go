package mistralclient

import (
	"encoding/json"
	"log"
	"os"
	"strconv"
	"strings"
)

var (
	logger = log.New(os.Stdout, "genkit-mistral: ", log.LstdFlags|log.Lshortfile)
)

// jsonMap unmarshal from either an object or a JSON string containing an object.
type jsonMap map[string]any

func (jm *jsonMap) UnmarshalJSON(b []byte) error {
	t := strings.TrimSpace(string(b))
	if t == "null" || t == "" {
		*jm = nil
		return nil
	}

	// Try a direct object first.
	var m map[string]any
	if err := json.Unmarshal(b, &m); err == nil {
		*jm = m
		return nil
	}

	// Fallback: the field is a quoted JSON string.
	s, err := strconv.Unquote(t)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return err
	}
	*jm = m
	return nil
}
