package mistralclient_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// timeoutNetError implements net.Error with Timeout() = true
type timeoutNetError struct{}

func (timeoutNetError) Error() string   { return "timeout" }
func (timeoutNetError) Timeout() bool   { return true }
func (timeoutNetError) Temporary() bool { return true } // for legacy checks

// flakyRoundTripper fails with a timeout once, then returns a successful response.
type flakyRoundTripper struct {
	failuresLeft int32
	successBody  []byte
}

func (f *flakyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if atomic.AddInt32(&f.failuresLeft, -1) >= 0 {
		return nil, timeoutNetError{}
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header:     make(http.Header),
		Body:       io.NopCloser(bytes.NewReader(f.successBody)),
		Request:    req,
	}
	resp.Header.Set("Content-Type", "application/json")
	return resp, nil
}

// makeMockServerWithCapture creates an HTTP test server that returns a fixed JSON response
// and also captures the JSON request body. The captured JSON is pretty-printed to make
// assertions easy to read in tests.
func makeMockServerWithCapture(t *testing.T, method, path, jsonResponse string, responseCode int, capturedBody *string) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == method && r.URL.Path == path {
			// Capture and pretty-print the incoming JSON body for assertions
			if r.Body != nil {
				defer r.Body.Close()
				raw, _ := io.ReadAll(r.Body)
				var anyJSON any
				if len(bytes.TrimSpace(raw)) > 0 && json.Unmarshal(raw, &anyJSON) == nil {
					pretty, _ := json.MarshalIndent(anyJSON, "", "  ")
					*capturedBody = string(pretty)
				} else {
					*capturedBody = string(raw)
				}
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(responseCode)
			_, _ = w.Write([]byte(jsonResponse))
			return
		}
		http.NotFound(w, r)
	}))
}
