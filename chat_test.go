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
	"testing"
	"time"
)

// newTestClient builds a ChatClient against an httptest server with a
// tiny backoff unit so retries are fast.
func newTestClient(t *testing.T, cfg ProviderConfig, srv *httptest.Server) *ChatClient {
	t.Helper()
	old := backoffUnit
	backoffUnit = time.Millisecond
	t.Cleanup(func() { backoffUnit = old })
	return &ChatClient{
		pc:     newProviderClient(cfg, srv.Client(), srv.Client()),
		model:  "test-model",
		parent: &Provider{cfg: cfg, sdk: New()},
	}
}

func TestCall_RetriesThenSucceeds(t *testing.T) {
	var mu sync.Mutex
	var codes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		codes = append(codes, 500)
		n := len(codes)
		mu.Unlock()
		if n <= 2 {
			w.WriteHeader(500)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.Content != "ok" {
		t.Fatalf("content = %q", res.Content)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(codes) != 3 {
		t.Fatalf("attempts = %d, want 3", len(codes))
	}
}

func TestCall_RateLimitExhaustion(t *testing.T) {
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(429)
		fmt.Fprint(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	var rl *RateLimitError
	if !errors.As(err, &rl) {
		t.Fatalf("err = %v (%T), want *RateLimitError", err, err)
	}
	if rl.Attempts != maxRetries+1 {
		t.Errorf("Attempts = %d, want %d", rl.Attempts, maxRetries+1)
	}
	if !strings.Contains(err.Error(), "rate") || strings.Contains(err.Error(), "Bearer") {
		t.Errorf("error text = %q", err.Error())
	}
}

func TestCall_NonRetryableFailsFast(t *testing.T) {
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(401)
		fmt.Fprint(w, `{"error":{"message":"bad key"}}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	var ae *APIError
	if !errors.As(err, &ae) || ae.Status != 401 {
		t.Fatalf("err = %v, want 401 APIError", err)
	}
	if attempts != 1 {
		t.Errorf("attempts = %d, want 1 (no retries on 401)", attempts)
	}
}

func TestCall_LearnOnceForceNoneEffort(t *testing.T) {
	var bodies []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		n := len(bodies)
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"Unsupported parameter: reasoning_effort is not supported with tools","type":"invalid_request_error"}}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{ReasoningEffort: true}}, srv)
	req := &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools:    []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking: "high",
	}
	res, err := cc.Call(context.Background(), req)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	_ = res
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

func sse(w http.ResponseWriter, chunks ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	flusher := w.(http.Flusher)
	for _, c := range chunks {
		fmt.Fprintf(w, "data: %s\n\n", c)
		flusher.Flush()
	}
}

func TestCallStream_HappyPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ct := r.Header.Get("Accept"); !strings.Contains(ct, "text/event-stream") {
			t.Errorf("Accept = %q", ct)
		}
		sse(w,
			`{"choices":[{"delta":{"reasoning_content":"thinking"}}]}`,
			`{"choices":[{"delta":{"content":"Hel"}}]}`,
			`{"choices":[{"delta":{"content":"lo"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`,
			`[DONE]`,
		)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	var kinds []DeltaKind
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(d Delta) error {
		kinds = append(kinds, d.Kind)
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "Hello" || res.ReasoningContent != "thinking" {
		t.Errorf("result = %+v", res)
	}
	if res.FinishReason != FinishStop || res.Usage.CompletionTokens != 2 {
		t.Errorf("finish/usage = %q %+v", res.FinishReason, res.Usage)
	}
	if len(kinds) != 3 {
		t.Fatalf("deltas = %v", kinds)
	}
}

func TestCallStream_AbortReturnsPartial(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w,
			`{"choices":[{"delta":{"content":"part1"}}]}`,
			`{"choices":[{"delta":{"content":"part2"}}]}`,
			`{"choices":[{"delta":{"content":"part3"}}]}`,
		)
		// Keep the connection open briefly so abort happens mid-stream.
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	sentinel := errors.New("stop now")
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(d Delta) error {
		if d.Text == "part2" {
			return sentinel
		}
		return nil
	})
	var sa *StreamAbortedError
	if !errors.As(err, &sa) || !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want StreamAbortedError wrapping sentinel", err)
	}
	if res == nil || res.Content != "part1part2" {
		t.Fatalf("partial result = %+v, want content accumulated through the aborted fragment", res)
	}
}

func TestCallStream_LearnOnceDropStreamOptions(t *testing.T) {
	var bodies []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		n := len(bodies)
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"stream_options is not supported"}}`)
			return
		}
		sse(w, `{"choices":[{"delta":{"content":"ok"}}]}`, `[DONE]`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "ok" {
		t.Fatalf("content = %q", res.Content)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(bodies))
	}
	if !strings.Contains(bodies[0], "stream_options") {
		t.Errorf("first body must carry stream_options: %s", bodies[0])
	}
	if strings.Contains(bodies[1], "stream_options") {
		t.Errorf("second body must drop stream_options: %s", bodies[1])
	}
}

func TestCallStream_LearnOnceForceBuffered(t *testing.T) {
	var bodies []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		n := len(bodies)
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"this endpoint does not support streaming"}}`)
			return
		}
		// Buffered fallback: plain JSON, no SSE.
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"buffered"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "buffered" {
		t.Fatalf("content = %q", res.Content)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(bodies))
	}
	if !strings.Contains(bodies[0], `"stream":true`) {
		t.Errorf("first body must stream: %s", bodies[0])
	}
	if strings.Contains(bodies[1], `"stream":true`) {
		t.Errorf("fallback body must be buffered: %s", bodies[1])
	}
}

func TestCallStream_NonSSEBodyFallsBack(t *testing.T) {
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if r.Header.Get("Accept") == "text/event-stream" {
			// 200 but plain JSON — the "non-SSE body" fallback.
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"choices":[{"message":{"content":"plain"},"finish_reason":"stop"}]}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"plain"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "plain" {
		t.Fatalf("content = %q", res.Content)
	}
}

func TestCallStream_IdleRetryBeforeFirstDelta(t *testing.T) {
	oldIdle := streamIdleTimeout
	streamIdleTimeout = 80 * time.Millisecond
	t.Cleanup(func() { streamIdleTimeout = oldIdle })

	var attempts int
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts++
		n := attempts
		mu.Unlock()
		if n == 1 {
			// SSE headers, then silence until the watchdog fires.
			w.Header().Set("Content-Type", "text/event-stream")
			w.(http.Flusher).Flush()
			time.Sleep(400 * time.Millisecond)
			return
		}
		sse(w, `{"choices":[{"delta":{"content":"second try"}}]}`, `[DONE]`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "second try" {
		t.Fatalf("content = %q", res.Content)
	}
	mu.Lock()
	defer mu.Unlock()
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2 (idle retry then success)", attempts)
	}
}

func TestCallStream_ConcurrentClients(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w, `{"choices":[{"delta":{"content":"x"}}]}`, `{"choices":[{"delta":{},"finish_reason":"stop"}]}`, `[DONE]`)
	}))
	defer srv.Close()

	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}
	clients := make([]*ChatClient, 8)
	for i := range clients {
		clients[i] = newTestClient(t, cfg, srv) // build before goroutines: helper swaps a package var
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(cc *ChatClient) {
			defer wg.Done()
			if _, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error { return nil }); err != nil {
				t.Errorf("concurrent CallStream: %v", err)
			}
		}(clients[i])
	}
	wg.Wait()
}
