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

func (jm *JsonMap) UnmarshalJSON(b []byte) error {
	t := strings.TrimSpace(string(b))
	if t == "null" {
		*jm = nil
		return nil
	}

	var m map[string]any
	if err := json.Unmarshal(b, &m); err == nil {
		*jm = m
		return nil
	}

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
