package mistralclient

import "strings"

type ErrorResponseDetail struct {
	Type  string `json:"type"`
	Loc   []string
	Msg   string
	Input bool
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

func (e ErrorResponse) Error() string {
	msg := strings.Builder{}
	msg.WriteString(e.Type)

	if len(e.Message.Detail) == 0 {
		return msg.String()
	}

	msg.WriteString(":")
	for _, detail := range e.Message.Detail {
		msg.WriteString(" ")
		msg.WriteString(detail.Type)
		msg.WriteString(": ")
		msg.WriteString(detail.Msg)
		if len(detail.Loc) > 0 {
			msg.WriteString(" ")
			msg.WriteString("(" + strings.Join(detail.Loc, ".") + ")")
		}
		msg.WriteString(";")
	}
	return msg.String()
}
