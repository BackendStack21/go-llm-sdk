package llm

import (
	"context"
	"errors"
	"net/http"
	"sync/atomic"
	"testing"
)

// m6b SL05: close-out tests (RunTools per-handler ctx check, coverage gaps).

// The context is checked before EACH handler: once a handler cancels it, the
// remaining handlers of the turn do not run and RunTools returns the ctx error.
func TestRunTools_CtxCheckedBeforeEachHandler(t *testing.T) {
	f := toolFormats[0] // openai
	srv, _ := newToolServer(t, f.call([2]string{"c1", "a"}, [2]string{"c2", "b"}), f.stop)
	cc := f.client(t, srv)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var ranA, ranB atomic.Int32
	_, _, err := RunTools(ctx, cc, f.req(), map[string]ToolHandler{
		"a": func(context.Context, ToolCall) (string, error) { ranA.Add(1); cancel(); return "ok", nil },
		"b": func(context.Context, ToolCall) (string, error) { ranB.Add(1); return "ok", nil },
	}, ToolLoopOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if ranA.Load() != 1 {
		t.Errorf("handler a ran %d times, want 1", ranA.Load())
	}
	if ranB.Load() != 0 {
		t.Errorf("handler b ran %d times after cancellation, want 0", ranB.Load())
	}
}

// A Gemini tool result carrying an image URL without a MIME type is rejected.
func TestGemini_ToolResultImageURLWithoutMIME(t *testing.T) {
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "go"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "wx", Arguments: "{}"}}},
		{Role: RoleTool, ToolCallID: "c1", ToolName: "wx", Parts: []ContentPart{ImageURLPart("https://x/y.png")}},
	}}
	_, err := buildGeminiRequest(req, "gemini-x", false)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v, want *ConfigError", err)
	}
}

// CountTokens with a reserved header fails before any I/O.
func TestCountTokens_ReservedHeaderBeforeIO(t *testing.T) {
	srv, rec := countServer(t, obsStep{status: 200, body: anCountOK})
	cc := countClient(t, "anthropic", FormatAnthropic, "claude-x", srv)
	req := countReq()
	req.Headers = map[string]string{"x-api-key": "evil"}
	_, err := cc.CountTokens(context.Background(), req)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v, want *ConfigError", err)
	}
	if rec.n() != 0 {
		t.Errorf("requests = %d, want 0", rec.n())
	}
}

// A request only the Anthropic builder rejects errors before any I/O.
func TestCountTokens_AnthropicBuilderRejection(t *testing.T) {
	srv, rec := countServer(t, obsStep{status: 200, body: anCountOK})
	cc := countClient(t, "anthropic", FormatAnthropic, "claude-x", srv)
	req := countReq()
	req.Thinking = "enabled"
	req.ThinkingBudget = 8000
	req.MaxTokens = 2000
	_, err := cc.CountTokens(context.Background(), req)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v, want *ConfigError", err)
	}
	if rec.n() != 0 {
		t.Errorf("requests = %d, want 0", rec.n())
	}
}

type closeoutSentinel struct{}

func (closeoutSentinel) Error() string { return "dial failed using " + obsKey }

type closeoutRT struct{ err error }

func (r closeoutRT) RoundTrip(*http.Request) (*http.Response, error) { return nil, r.err }

// A masked AttemptEvent.Err still unwraps to the original error.
func TestAttemptEvent_MaskedErrUnwraps(t *testing.T) {
	fastBackoff(t)
	orig := closeoutSentinel{}
	col := &obsCollector{}
	s := New(WithAttemptObserver(col.fn), WithTransport(closeoutRT{err: orig}),
		WithRetryPolicy(RetryPolicy{MaxAttempts: 1}),
		WithProvider("openai", WithAPIKey(obsKey), WithBaseURL("http://127.0.0.1:1")))
	cc, err := s.Chat("openai", "gpt-4o")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	evs := col.all()
	if len(evs) == 0 || evs[0].Err == nil {
		t.Fatalf("events = %+v", evs)
	}
	e := evs[0].Err
	obsNoKey(t, evs[0])
	if !errors.Is(e, orig) {
		t.Errorf("errors.Is(event err, original) = false: %v", e)
	}
	var cs closeoutSentinel
	if !errors.As(e, &cs) {
		t.Errorf("errors.As to original type failed: %v", e)
	}
}
