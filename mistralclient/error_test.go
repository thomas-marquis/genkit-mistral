package mistralclient_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

func TestErrorResponse(t *testing.T) {
	t.Run("should format error message", func(t *testing.T) {
		resp := mistralclient.ErrorResponse{
			Object: "error",
			Message: mistralclient.ErrorResponseMessage{
				Detail: []mistralclient.ErrorResponseDetail{
					{
						Type:  "extra_forbidden",
						Loc:   []string{"body", "parallel_tool_calls"},
						Msg:   "Extra inputs are not permitted",
						Input: true,
					},
				},
			},
			Type:  "invalid_request_error",
			Param: nil,
			Code:  nil,
		}

		assert.Equal(t, "invalid_request_error: extra_forbidden: Extra inputs are not permitted (body.parallel_tool_calls);", resp.Error())
	})

	t.Run("should format error essage with multiple details", func(t *testing.T) {
		resp := mistralclient.ErrorResponse{
			Object: "error",
			Message: mistralclient.ErrorResponseMessage{
				Detail: []mistralclient.ErrorResponseDetail{
					{
						Type:  "extra_forbidden",
						Loc:   []string{"body", "parallel_tool_calls"},
						Msg:   "Extra inputs are not permitted",
						Input: true,
					},
					{
						Type:  "missing_required",
						Loc:   []string{"body", "messages"},
						Msg:   "Missing required property: messages",
						Input: false,
					},
				},
			},
			Type:  "invalid_request_error",
			Param: nil,
			Code:  nil,
		}

		assert.Equal(t, "invalid_request_error: extra_forbidden: Extra inputs are not permitted (body.parallel_tool_calls); missing_required: Missing required property: messages (body.messages);", resp.Error())
	})
}
