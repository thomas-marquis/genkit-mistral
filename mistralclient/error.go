package mistralclient

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ErrorResponseDetail struct {
	Type  string   `json:"type"`
	Loc   []string `json:"loc"`
	Msg   string   `json:"msg"`
	Input bool     `json:"input"`
}

type ErrorResponseMessage struct {
	Detail []ErrorResponseDetail `json:"detail"`
}

type ErrorResponse struct {
	Object  string               `json:"object"`
	Message ErrorResponseMessage `json:"message"`
	Type    string               `json:"type"`
	Param   interface{}          `json:"param"`
	Code    interface{}          `json:"code"`
}

var _ error = (*ErrorResponse)(nil)
var _ json.Unmarshaler = (*ErrorResponse)(nil)

func (e *ErrorResponse) UnmarshalJSON(data []byte) error {
	var value map[string]any
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value == nil {
		return nil
	}
	e.Object = value["object"].(string)
	e.Type = value["type"].(string)
	e.Param = value["param"]
	e.Code = value["code"]

	rawMsg := value["message"]
	if rawMsg == nil {
		return nil
	}

	if val, ok := rawMsg.(string); ok {
		e.Message.Detail = append(e.Message.Detail, ErrorResponseDetail{Msg: val})
	} else if val, ok := rawMsg.(map[string]any); ok {
		var errMsg ErrorResponseMessage
		if err := mapToStruct(val, &errMsg); err != nil {
			return err
		}
		e.Message = errMsg
	} else {
		return fmt.Errorf("unexpected error response message format: %v", value)
	}
	return nil
}

func (e *ErrorResponse) Error() string {
	msg := strings.Builder{}
	msg.WriteString(e.Type)

	if len(e.Message.Detail) == 0 {
		return msg.String()
	}

	msg.WriteString(":")
	for i, detail := range e.Message.Detail {
		msg.WriteString(" ")
		if detail.Type != "" {
			msg.WriteString(detail.Type)
			msg.WriteString(": ")
		}
		msg.WriteString(detail.Msg)
		if len(detail.Loc) > 0 {
			msg.WriteString(" ")
			msg.WriteString("(" + strings.Join(detail.Loc, ".") + ")")
		}
		if i < len(e.Message.Detail)-1 {
			msg.WriteString(";")
		}
	}
	return msg.String()
}
