package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// streamRejected is a heuristic classifier; these cases pin its intended
// shape: a 400 that rejects streaming itself, not any 400 that merely
// mentions the word "stream".
func TestStreamRejectedClassification(t *testing.T) {
	cases := []struct {
		msg  string
		want bool
	}{
		{"Streaming is not supported for this model.", true},
		{"This model does not support streaming via the API.", true},
		{"streaming unsupported for model", true},
		{"Streaming rejected for this model", true},
		{"Context length exceeded: the streaming limit is 8192 tokens.", false},
		{"Reasoning effort cannot be combined with tools while streaming.", false},
		{"'stream_options.include_usage' is not supported by this model.", false}, // separate learn-once class
		{"max_tokens is too large for this model.", false},
	}
	for _, tc := range cases {
		e := &APIError{Status: http.StatusBadRequest, Message: tc.msg}
		if got := streamRejected(e); got != tc.want {
			t.Errorf("streamRejected(%q) = %v, want %v", tc.msg, got, tc.want)
		}
	}
	if streamRejected(&APIError{Status: http.StatusInternalServerError, Message: "streaming is not supported"}) {
		t.Error("non-400 must never classify as stream rejection")
	}
}

// Contract: partial output followed by a mid-stream failure is returned as
// the partial result plus a wrapped error, and is never retried — a silent
// retry would duplicate user-visible output.
func TestCallStreamPartialFailureNotRetried(t *testing.T) {
	var reqs int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&reqs, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n")
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler) // hard connection drop mid-stream
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	deltas := 0
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		deltas++
		return nil
	})
	if err == nil {
		t.Fatal("expected a mid-stream failure error")
	}
	if res == nil {
		t.Fatalf("partial result lost (err = %v)", err)
	}
	if res.Content != "ab" {
		t.Errorf("partial content = %q, want \"ab\"", res.Content)
	}
	if deltas != 2 {
		t.Errorf("deltas delivered = %d, want 2", deltas)
	}
	if got := atomic.LoadInt32(&reqs); got != 1 {
		t.Errorf("requests = %d, want 1 (never retry after partial output)", got)
	}
	if strings.Contains(err.Error(), "retry exhausted") {
		t.Errorf("partial failure misclassified as retry exhaustion: %v", err)
	}
}

// Contract: a stream that stays 429 through the whole retry budget surfaces
// as *RateLimitError with Attempts = maxRetries+1.
func TestCallStreamPersistent429ReturnsRateLimitError(t *testing.T) {
	var mu sync.Mutex
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		mu.Unlock()
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(429)
		fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	var rl *RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("err = %v (%T), want *RateLimitError", err, err)
	}
	if rl.Attempts != maxRetries+1 {
		t.Errorf("Attempts = %d, want %d", rl.Attempts, maxRetries+1)
	}
	mu.Lock()
	defer mu.Unlock()
	if n != maxRetries+1 {
		t.Errorf("server saw %d requests, want %d", n, maxRetries+1)
	}
}

// A 429 followed by a healthy stream must succeed after exactly one retry.
func TestCallStream429ThenSuccess(t *testing.T) {
	var mu sync.Mutex
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		first := n == 1
		mu.Unlock()
		if first {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(429)
			fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
			return
		}
		sse(w,
			`{"choices":[{"delta":{"content":"ok"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`,
			"[DONE]",
		)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "ok" || res.Usage.PromptTokens != 3 {
		t.Errorf("res = %q usage %+v", res.Content, res.Usage)
	}
	mu.Lock()
	defer mu.Unlock()
	if n != 2 {
		t.Errorf("requests = %d, want 2", n)
	}
}

// The wall-clock deadline after partial output must return the partial
// result wrapped in the deadline error — not a bare context error, and not
// a retry.
func TestCallStreamDeadlineAfterPartialKeepsPartial(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w, `{"choices":[{"delta":{"content":"a"}}]}`)
		time.Sleep(500 * time.Millisecond) // deadline fires long before
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	cc.SetRequestTimeout(100 * time.Millisecond)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context deadline", err)
	}
	if res == nil || res.Content != "a" {
		t.Errorf("partial result lost after deadline: res=%v err=%v", res, err)
	}
	if strings.Contains(err.Error(), "retry exhausted") {
		t.Errorf("cancellation mislabeled as retry exhaustion: %v", err)
	}
}

// Buffered path: a 429 whose Retry-After outlives the request context must
// still surface as *RateLimitError (the stream path already does this) —
// the caller needs Status/RetryAfter to plan the retry.
func TestCallBuffered429DeadlineKeepsRateLimitError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "3600")
		w.WriteHeader(429)
		fmt.Fprint(w, `{"error":{"message":"quota exhausted, retry in an hour"}}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	_, err := cc.Call(ctx, &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	var rl *RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("err = %v (%T), want *RateLimitError (a bare ctx error hides the 429)", err, err)
	}
	if rl.Attempts != 1 {
		t.Errorf("Attempts = %d, want 1", rl.Attempts)
	}
	if rl.RetryAfter != 3600*time.Second {
		t.Errorf("RetryAfter = %v, want 1h", rl.RetryAfter)
	}
	if rl.Status != http.StatusTooManyRequests {
		t.Errorf("Status = %d, want 429", rl.Status)
	}
}

// RateLimitError embeds APIError; errors.As must reach Status/Retryable.
func TestRateLimitErrorUnwrapsToAPIError(t *testing.T) {
	rl := &RateLimitError{APIError: APIError{Provider: "openai", Status: 429, Message: "slow down"}, Attempts: 3}
	var ae *APIError
	if !errors.As(rl, &ae) || ae.Status != 429 {
		t.Fatalf("errors.As(*APIError) failed on RateLimitError (ae=%v)", ae)
	}
}

// The most common OpenAI error envelope is the nested error object; it must
// be parsed into Message/Code, not left as the raw-body fallback.
func TestHTTPErrorParsesNestedOpenAIError(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "openai", Format: FormatOpenAI}, nil, nil)
	e := pc.httpError(400, []byte(`{"error":{"message":"nested boom","code":"invalid_request_error"}}`))
	if e.Message != "nested boom" || e.Code != "invalid_request_error" {
		t.Fatalf("APIError = %+v, want parsed nested message/code", e)
	}
}

// Unparseable error bodies degrade to the raw preview, capped.
func TestHTTPErrorTruncatesUnparseableBody(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "openai", Format: FormatOpenAI}, nil, nil)
	e := pc.httpError(500, []byte(strings.Repeat("x", 5000)))
	if len(e.Message) != maxErrorBodyPreview {
		t.Fatalf("preview = %d bytes, want %d", len(e.Message), maxErrorBodyPreview)
	}
}

// A 200 + SSE headers response that closes with no events and no completion
// signal is a transport failure (retryable), not a silent empty success.
func TestCallStreamPrematureCloseNoEventsIsRetryable(t *testing.T) {
	var mu sync.Mutex
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		// close: no events, no [DONE]
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err == nil {
		t.Fatalf("premature close returned empty success (res=%+v)", res)
	}
	mu.Lock()
	defer mu.Unlock()
	if n != maxRetries+1 {
		t.Errorf("requests = %d, want %d (retryable: nothing was emitted)", n, maxRetries+1)
	}
}

// Deltas followed by a premature close (no completion signal) must return
// the partial result with a wrapped error, never retried.
func TestCallStreamPrematureCloseAfterDeltasKeepsPartial(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n")
		w.(http.Flusher).Flush()
		// close: no finish_reason, no [DONE]
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if res == nil || res.Content != "a" {
		t.Fatalf("partial result lost after premature close: res=%v err=%v", res, err)
	}
	if err == nil || !strings.Contains(err.Error(), "before completion") {
		t.Fatalf("err = %v, want premature-completion error", err)
	}
}

// The stream path must learn the force-none-effort fallback exactly like
// the buffered path does.
func TestCallStreamLearnsEffortNone(t *testing.T) {
	var mu sync.Mutex
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		mu.Unlock()
		if strings.Contains(string(b), `"reasoning_effort"`) && !strings.Contains(string(b), `"reasoning_effort":"none"`) {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"reasoning_effort is not supported with tools on this model"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{ReasoningEffort: true}}, srv)
	req := &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools:    []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking: "high",
	}
	_, err := cc.CallStream(context.Background(), req, func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(bodies))
	}
	if !strings.Contains(bodies[0], `"reasoning_effort":"high"`) {
		t.Errorf("first body missing effort: %s", bodies[0])
	}
	if !strings.Contains(bodies[1], `"reasoning_effort":"none"`) {
		t.Errorf("second body must pin effort none: %s", bodies[1])
	}
}
