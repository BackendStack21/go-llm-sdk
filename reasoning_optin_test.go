package llm

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// ── request side: IncludeReasoning quirk ──────────────────────────────────

// The quirk must put the gateway opt-in flag on the wire. OpenRouter's
// documented legacy equivalent of `reasoning: {}` is `include_reasoning: true`
// (openrouter.ai/docs/guides/best-practices/reasoning-tokens, Legacy
// Parameters) — the safest portable opt-in across OpenAI-format gateways.
func TestIncludeReasoningQuirkReachesWire(t *testing.T) {
	var bodies [][]byte
	srv := captureJSON(t, &bodies)
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{
		ID: "gw", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k",
	}, srv)
	cc.pc.cfg.Quirks.IncludeReasoning = true
	if _, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(bodies))
	}
	if !strings.Contains(string(bodies[0]), `"include_reasoning":true`) {
		t.Errorf("IncludeReasoning quirk must send include_reasoning:true; body: %s", bodies[0])
	}
}

// Regression guard: providers without the quirk must not see the flag —
// strict OpenAI-format endpoints reject unknown parameters.
func TestNoIncludeReasoningByDefault(t *testing.T) {
	var bodies [][]byte
	srv := captureJSON(t, &bodies)
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{
		ID: "plain", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k",
	}, srv)
	if _, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if strings.Contains(string(bodies[0]), `"include_reasoning"`) {
		t.Errorf("include_reasoning must be absent without the quirk; body: %s", bodies[0])
	}
}

// The quirk must reach streaming request bodies too — gateways see the
// opt-in on the stream call, which is exactly the call that carries the
// reasoning deltas back.
func TestIncludeReasoningQuirkReachesStreamWire(t *testing.T) {
	var mu sync.Mutex
	var bodies [][]byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		mu.Lock()
		bodies = append(bodies, b)
		mu.Unlock()
		sse(w,
			`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
			`[DONE]`,
		)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{
		ID: "gw", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k",
	}, srv)
	cc.pc.cfg.Quirks.IncludeReasoning = true
	if _, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	}); err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(bodies))
	}
	if !strings.Contains(string(bodies[0]), `"include_reasoning":true`) {
		t.Errorf("stream body missing include_reasoning:true; body: %s", bodies[0])
	}
}

// ── response side: documented reasoning shapes ────────────────────────────

// OpenRouter non-streaming: `message.reasoning` (plain string). reasoning_content
// keeps precedence when both arrive.
func TestParseReasoningAliasString(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok","reasoning":"alias think"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.ReasoningContent != "alias think" {
		t.Errorf("ReasoningContent = %q, want %q (message.reasoning string)", res.ReasoningContent, "alias think")
	}
}

// OpenRouter reasoning_details: typed array; text and summary fold in order,
// encrypted entries are skipped, reasoning_content keeps precedence.
func TestParseReasoningDetailsFold(t *testing.T) {
	body := `{"choices":[{"message":{"role":"assistant","content":"ok",` +
		`"reasoning_details":[` +
		`{"type":"reasoning.text","text":"step one"},` +
		`{"type":"reasoning.encrypted","data":"ZZZ"},` +
		`{"type":"reasoning.summary","summary":"step two"}]},` +
		`"finish_reason":"stop"}],"usage":{}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	want := "step one\nstep two"
	if res.ReasoningContent != want {
		t.Errorf("ReasoningContent = %q, want %q", res.ReasoningContent, want)
	}
}

// Precedence: when reasoning_content and reasoning both arrive, the
// standardized field wins.
func TestParseReasoningContentPrecedence(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok","reasoning":"alias","reasoning_content":"standard"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.ReasoningContent != "standard" {
		t.Errorf("ReasoningContent = %q, want %q", res.ReasoningContent, "standard")
	}
}

// OpenRouter streaming: reasoning arrives as delta.reasoning_details chunks.
func TestStreamReasoningDetailsDeltas(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w,
			`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"deep "}]}}]}`,
			`{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.summary","summary":"thought"},{"type":"reasoning.encrypted","data":"X"}]}}]}`,
			`{"choices":[{"delta":{"content":"answer"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`[DONE]`,
		)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	var reasoning, content strings.Builder
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(d Delta) error {
		if d.Kind == DeltaReasoning {
			reasoning.WriteString(d.Text)
		}
		if d.Kind == DeltaContent {
			content.WriteString(d.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if got := reasoning.String(); got != "deep thought" {
		t.Errorf("reasoning deltas = %q, want %q", got, "deep thought")
	}
	if content.String() != "answer" {
		t.Errorf("content deltas = %q", content.String())
	}
	if res.ReasoningContent != "deep thought" {
		t.Errorf("accumulated ReasoningContent = %q, want %q", res.ReasoningContent, "deep thought")
	}
}

// delta.reasoning (plain string) — the alias shape on streams.
func TestStreamReasoningAliasDeltas(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w,
			`{"choices":[{"delta":{"reasoning":"hmm "}}]}`,
			`{"choices":[{"delta":{"reasoning":"ok"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`[DONE]`,
		)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	var reasoning strings.Builder
	if _, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(d Delta) error {
		if d.Kind == DeltaReasoning {
			reasoning.WriteString(d.Text)
		}
		return nil
	}); err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if got := reasoning.String(); got != "hmm ok" {
		t.Errorf("reasoning deltas = %q, want %q", got, "hmm ok")
	}
}

// ── learn-transition observer ─────────────────────────────────────────────

func collectLearnEvents(t *testing.T) (*[]LearnEvent, func()) {
	t.Helper()
	mu := &sync.Mutex{}
	var events []LearnEvent
	learnObserverMu.RLock()
	old := learnObserver
	learnObserverMu.RUnlock()
	SetLearnObserver(func(ev LearnEvent) {
		mu.Lock()
		events = append(events, ev)
		mu.Unlock()
	})
	return &events, func() { SetLearnObserver(old) }
}

// A 2xx JSON response to a streaming request engages the permanent buffered
// downgrade — the client must be able to observe exactly that, once.
func TestLearnObserver_BufferedOnNonSSE(t *testing.T) {
	events, restore := collectLearnEvents(t)
	defer restore()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"buffered"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	for i := 0; i < 2; i++ { // second attempt rides the engaged flag: no new event
		res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
			return nil
		})
		if err != nil {
			t.Fatalf("CallStream: %v", err)
		}
		if res.Content != "buffered" {
			t.Fatalf("content = %q, want buffered", res.Content)
		}
	}
	if len(*events) != 1 {
		t.Fatalf("learn events = %d, want 1: %+v", len(*events), *events)
	}
	ev := (*events)[0]
	if ev.Kind != LearnBuffered {
		t.Errorf("Kind = %q, want %q", ev.Kind, LearnBuffered)
	}
	if ev.Provider != "openai" {
		t.Errorf("Provider = %q, want openai", ev.Provider)
	}
}

// An explicit "streaming not supported" 400 must fire the same observer with
// the provider's message attached.
func TestLearnObserver_BufferedOnStreamRejected(t *testing.T) {
	events, restore := collectLearnEvents(t)
	defer restore()

	var mu sync.Mutex
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		first := n == 1
		mu.Unlock()
		if first {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"streaming is not supported by this deployment"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"buffered"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "buffered" {
		t.Fatalf("content = %q, want buffered (learn retry must succeed)", res.Content)
	}
	if len(*events) != 1 {
		t.Fatalf("learn events = %d, want 1: %+v", len(*events), *events)
	}
	ev := (*events)[0]
	if ev.Kind != LearnBuffered {
		t.Errorf("Kind = %q, want %q", ev.Kind, LearnBuffered)
	}
	if !strings.Contains(ev.Message, "not supported") {
		t.Errorf("Message = %q, want provider rejection text", ev.Message)
	}
}

// Nil observer is the default and must change nothing: buffered fallback
// still returns a parsed result, no panic.
func TestLearnObserver_NilByDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "ok" {
		t.Errorf("content = %q, want ok", res.Content)
	}
}

// Ordinary retries that engage no learn flag must not fire the observer.
func TestLearnObserver_SilentOnPlainRetry(t *testing.T) {
	events, restore := collectLearnEvents(t)
	defer restore()

	var mu sync.Mutex
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		first := n == 1
		mu.Unlock()
		if first {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	if _, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	}); err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if len(*events) != 0 {
		t.Errorf("learn events = %+v, want none", *events)
	}
}
