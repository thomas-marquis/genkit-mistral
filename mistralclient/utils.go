package mistralclient

import (
	"encoding/json"
	"log"
	"os"
	"strconv"
	"strings"
)

var (
	logger = log.New(os.Stdout, "mistral-client: ", log.LstdFlags|log.Lshortfile)
)

// JsonMap unmarshal from either an object or a JSON string containing an object.
type JsonMap map[string]any

func (jm *JsonMap) UnmarshalJSON(data []byte) error {
	trimmedData := strings.TrimSpace(string(data))
	if trimmedData == "null" {
		*jm = nil
		return nil
	}

	var resMap map[string]any
	if err := json.Unmarshal(data, &resMap); err == nil {
		*jm = resMap
		return nil
	}

	unquotedData, err := strconv.Unquote(trimmedData)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(unquotedData), &resMap); err != nil {
		return err
	}
	*jm = resMap
	return nil
}

type JsonSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema"`
	Strict      bool   `json:"strict,omitempty"`
}

func mapToStruct(from map[string]any, to any) error {
	j, err := json.Marshal(from)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(j, to); err != nil {
		return err
	}
	return nil
}
