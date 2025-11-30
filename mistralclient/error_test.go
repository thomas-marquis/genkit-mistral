package mistralclient_test

import (
	"encoding/json"
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

		assert.Equal(t, "invalid_request_error: extra_forbidden: Extra inputs are not permitted (body.parallel_tool_calls)", resp.Error())
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

		assert.Equal(t, "invalid_request_error: extra_forbidden: Extra inputs are not permitted (body.parallel_tool_calls); missing_required: Missing required property: messages (body.messages)", resp.Error())
	})

	t.Run("should unmarshall error response from json with array message", func(t *testing.T) {
		expected := mistralclient.ErrorResponse{
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

		j := `{
			"object": "error",
			"message": {
				"detail": [
					{
						"type": "extra_forbidden",
						"loc": [
							"body",
							"parallel_tool_calls"
						],
						"msg": "Extra inputs are not permitted",
						"input": true
					},
					{
						"type": "missing_required",
						"loc": [
							"body",
							"messages"
						],
						"msg": "Missing required property: messages",
						"input": false
					}
				]
			},
			"type": "invalid_request_error",
			"param": null,
			"code": null
		}`

		var resp mistralclient.ErrorResponse
		assert.NoError(t, json.Unmarshal([]byte(j), &resp))
		assert.Equal(t, expected, resp)
	})

	t.Run("should unmarshall error response from json with string message", func(t *testing.T) {
		j := `{
			"object": "error",
			"message": "This model does not support output_dimension.",
			"type": "invalid_request_invalid_args",
			"param": null,
			"code": "3051"
		}`
		var resp mistralclient.ErrorResponse
		assert.NoError(t, json.Unmarshal([]byte(j), &resp))
		assert.Equal(t, "invalid_request_invalid_args: This model does not support output_dimension.", resp.Error())
	})
}
