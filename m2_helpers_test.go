package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// m2Client builds a ChatClient of the given format against a server that
// answers every request with body (SSE when stream is true).
func m2Client(t *testing.T, format Format, id, body string, stream bool) *ChatClient {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return newTestClient(t, ProviderConfig{ID: id, Format: format, BaseURL: srv.URL, APIKey: "k"}, srv)
}

// m2Run executes the request buffered or streaming and returns the result
// plus every delta (streaming only).
func m2Run(t *testing.T, c *ChatClient, req *ChatRequest, stream bool) (*ChatResult, []Delta) {
	t.Helper()
	var ds []Delta
	var res *ChatResult
	var err error
	if stream {
		res, err = c.CallStream(context.Background(), req, func(d Delta) error { ds = append(ds, d); return nil })
	} else {
		res, err = c.Call(context.Background(), req)
	}
	if err != nil {
		t.Fatalf("call (stream=%v): %v", stream, err)
	}
	return res, ds
}

func m2Req() *ChatRequest {
	return &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools:    []ToolDef{{Name: "f", Description: "d", Parameters: []byte(`{"type":"object"}`)}},
	}
}
