package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// m3Wire is what a test server captured from one request.
type m3Wire struct {
	mu      sync.Mutex
	hits    int
	escPath string
	rawQ    string
	body    []byte
}

func (w *m3Wire) snapshot() (int, string, string, []byte) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.hits, w.escPath, w.rawQ, w.body
}

// m3Reply canned success bodies per wire format (buffered).
func m3Reply(format Format) string {
	switch format {
	case FormatAnthropic:
		return `{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	case FormatGemini:
		return `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`
	default:
		return `{"id":"c","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	}
}

// m3Server records every request and answers 200 with body (SSE when sse).
func m3Server(t *testing.T, body string, sse bool) (*httptest.Server, *m3Wire) {
	t.Helper()
	rec := &m3Wire{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		rec.mu.Lock()
		rec.hits++
		rec.escPath, rec.rawQ, rec.body = r.URL.EscapedPath(), r.URL.RawQuery, b
		rec.mu.Unlock()
		if sse {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

// m3Chat sends req through a ChatClient of the given format and returns the
// call error plus the captured wire request.
func m3Chat(t *testing.T, format Format, req *ChatRequest) (*m3Wire, error) {
	t.Helper()
	srv, rec := m3Server(t, m3Reply(format), false)
	c := newTestClient(t, ProviderConfig{ID: string(format) + "-m3", Format: format, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := c.Call(context.Background(), req)
	return rec, err
}

func m3JSON(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("captured body is not JSON: %v\n%s", err, b)
	}
	return m
}
