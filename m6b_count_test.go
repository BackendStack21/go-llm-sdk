package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// m6b SL03: CountTokens (AC-4).

type countRec struct {
	mu      sync.Mutex
	paths   []string
	bodies  []map[string]any
	headers []http.Header
}

func (r *countRec) n() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.paths)
}

// countServer answers request i with steps[i] (last repeats) and records it.
func countServer(t *testing.T, steps ...obsStep) (*httptest.Server, *countRec) {
	t.Helper()
	rec := &countRec{}
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var b map[string]any
		_ = json.Unmarshal(raw, &b)
		rec.mu.Lock()
		rec.paths = append(rec.paths, r.URL.Path)
		rec.bodies = append(rec.bodies, b)
		rec.headers = append(rec.headers, r.Header.Clone())
		rec.mu.Unlock()
		i := min(int(n.Add(1))-1, len(steps)-1)
		s := steps[i]
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(s.status)
		_, _ = w.Write([]byte(s.body))
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

const (
	anCountOK = `{"input_tokens":42}`
	gmCountOK = `{"totalTokens":17}`
)

func countClient(t *testing.T, id string, f Format, model string, srv *httptest.Server) *ChatClient {
	t.Helper()
	cc := newTestClient(t, ProviderConfig{ID: id, Format: f, BaseURL: srv.URL, APIKey: "sk-count-secret"}, srv)
	cc.model = model
	return cc
}

func countReq() *ChatRequest {
	return &ChatRequest{
		Messages:    []Message{{Role: RoleUser, Content: "hello"}},
		System:      []SystemBlock{{Text: "be brief"}},
		Tools:       []ToolDef{{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
		MaxTokens:   99,
		Temperature: 0.5,
		TopP:        0.9,
	}
}

func TestCountTokens_AnthropicWire(t *testing.T) {
	srv, rec := countServer(t, obsStep{status: 200, body: anCountOK})
	cc := countClient(t, "anthropic", FormatAnthropic, "claude-x", srv)
	req := countReq()
	req.Thinking = "enabled"
	req.ThinkingBudget = 2048
	req.MaxTokens = 8192
	n, err := cc.CountTokens(context.Background(), req)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if n != 42 {
		t.Errorf("count = %d, want 42", n)
	}
	if rec.n() != 1 || rec.paths[0] != "/v1/messages/count_tokens" {
		t.Fatalf("paths = %v", rec.paths)
	}
	b := rec.bodies[0]
	for _, k := range []string{"stream", "max_tokens", "temperature", "top_p", "top_k"} {
		if _, ok := b[k]; ok {
			t.Errorf("body has %q: %v", k, b)
		}
	}
	for _, k := range []string{"model", "messages", "system", "tools", "thinking"} {
		if _, ok := b[k]; !ok {
			t.Errorf("body missing %q: %v", k, b)
		}
	}
	if b["model"] != "claude-x" {
		t.Errorf("model = %v", b["model"])
	}
	for k := range b {
		switch k {
		case "model", "messages", "system", "tools", "tool_choice", "thinking":
		default:
			t.Errorf("unexpected key %q in count body", k)
		}
	}
	if got := rec.headers[0].Get("x-api-key"); got != "sk-count-secret" {
		t.Errorf("x-api-key = %q", got)
	}
}

func TestCountTokens_AnthropicToolChoiceKept(t *testing.T) {
	srv, rec := countServer(t, obsStep{status: 200, body: anCountOK})
	cc := countClient(t, "anthropic", FormatAnthropic, "claude-x", srv)
	req := countReq()
	req.ToolChoice = &ToolChoice{Mode: ToolChoiceRequired}
	if _, err := cc.CountTokens(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if rec.n() != 1 {
		t.Fatalf("requests = %d, want 1", rec.n())
	}
	if _, ok := rec.bodies[0]["tool_choice"]; !ok {
		t.Errorf("tool_choice missing: %v", rec.bodies[0])
	}
}

func TestCountTokens_GeminiWire(t *testing.T) {
	srv, rec := countServer(t, obsStep{status: 200, body: gmCountOK})
	cc := countClient(t, "gemini", FormatGemini, "gemini-x", srv)
	n, err := cc.CountTokens(context.Background(), countReq())
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if n != 17 {
		t.Errorf("count = %d, want 17", n)
	}
	if rec.n() != 1 || rec.paths[0] != "/v1beta/models/gemini-x:countTokens" {
		t.Fatalf("paths = %v", rec.paths)
	}
	b := rec.bodies[0]
	g, ok := b["generateContentRequest"].(map[string]any)
	if !ok || len(b) != 1 {
		t.Fatalf("body = %v, want only generateContentRequest", b)
	}
	if g["model"] != "models/gemini-x" {
		t.Errorf("model = %v", g["model"])
	}
	for _, k := range []string{"contents", "systemInstruction", "tools"} {
		if _, ok := g[k]; !ok {
			t.Errorf("generateContentRequest missing %q: %v", k, g)
		}
	}
	if _, ok := g["generationConfig"]; ok {
		t.Errorf("generationConfig must not be sent: %v", g)
	}
	if got := rec.headers[0].Get("x-goog-api-key"); got != "sk-count-secret" {
		t.Errorf("x-goog-api-key = %q", got)
	}
}

func TestCountTokens_UnsupportedNoNetwork(t *testing.T) {
	cases := []struct{ name, model string }{
		{"chat", "gpt-4o"},
		{"responses", "gpt-5.6"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, rec := countServer(t, obsStep{status: 200, body: `{}`})
			cc := countClient(t, "openai", FormatOpenAI, tc.model, srv)
			n, err := cc.CountTokens(context.Background(), countReq())
			if !errors.Is(err, ErrUnsupported) {
				t.Fatalf("err = %v, want ErrUnsupported", err)
			}
			var ue *UnsupportedError
			if !errors.As(err, &ue) || ue.Op != "CountTokens" {
				t.Errorf("err = %#v, want *UnsupportedError{Op:CountTokens}", err)
			}
			if n != 0 {
				t.Errorf("count = %d, want 0 (never an estimate)", n)
			}
			if rec.n() != 0 {
				t.Errorf("made %d HTTP requests, want 0", rec.n())
			}
			if strings.Contains(err.Error(), "sk-count-secret") {
				t.Error("error leaks API key")
			}
		})
	}
}

func TestUnsupportedError_Text(t *testing.T) {
	e := &UnsupportedError{Op: "CountTokens", Provider: "openai", Reason: "no token counting endpoint"}
	want := "llm: CountTokens unsupported on openai: no token counting endpoint"
	if e.Error() != want {
		t.Errorf("Error() = %q, want %q", e.Error(), want)
	}
	if !errors.Is(e, ErrUnsupported) {
		t.Error("Unwrap must reach ErrUnsupported")
	}
}

func TestCountTokens_ValidationBeforeNetwork(t *testing.T) {
	for _, f := range []struct {
		id     string
		format Format
		model  string
		ok     string
	}{
		{"anthropic", FormatAnthropic, "claude-x", anCountOK},
		{"gemini", FormatGemini, "gemini-x", gmCountOK},
	} {
		t.Run(f.id, func(t *testing.T) {
			srv, rec := countServer(t, obsStep{status: 200, body: f.ok})
			cc := countClient(t, f.id, f.format, f.model, srv)
			bad := []*ChatRequest{
				nil,
				{Messages: []Message{{Role: "bogus", Content: "x"}}},
				{Messages: []Message{{Role: RoleUser, Content: "x"}}, Headers: map[string]string{"Bad Name": "v"}},
			}
			for i, r := range bad {
				_, err := cc.CountTokens(context.Background(), r)
				var ce *ConfigError
				if !errors.As(err, &ce) {
					t.Errorf("case %d: err = %v, want *ConfigError", i, err)
				}
			}
			cc.model = ""
			_, err := cc.CountTokens(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}})
			var ce *ConfigError
			if !errors.As(err, &ce) {
				t.Errorf("no model: err = %v, want *ConfigError", err)
			}
			if rec.n() != 0 {
				t.Errorf("made %d HTTP requests, want 0", rec.n())
			}
		})
	}
}

func TestCountTokens_ModelFromRequestFallback(t *testing.T) {
	srv, rec := countServer(t, obsStep{status: 200, body: gmCountOK})
	cc := countClient(t, "gemini", FormatGemini, "", srv)
	req := countReq()
	req.Model = "gemini-req"
	if _, err := cc.CountTokens(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if rec.n() != 1 || rec.paths[0] != "/v1beta/models/gemini-req:countTokens" {
		t.Errorf("paths = %v", rec.paths)
	}
}

func TestCountTokens_RetriesServerError(t *testing.T) {
	for _, f := range []struct {
		id     string
		format Format
		model  string
		ok     string
		want   int
	}{
		{"anthropic", FormatAnthropic, "claude-x", anCountOK, 42},
		{"gemini", FormatGemini, "gemini-x", gmCountOK, 17},
	} {
		t.Run(f.id, func(t *testing.T) {
			srv, rec := countServer(t, obsStep{status: 500, body: `{"error":"boom"}`}, obsStep{status: 200, body: f.ok})
			cc := countClient(t, f.id, f.format, f.model, srv)
			n, err := cc.CountTokens(context.Background(), countReq())
			if err != nil {
				t.Fatalf("CountTokens: %v", err)
			}
			if n != f.want || rec.n() != 2 {
				t.Errorf("count = %d attempts = %d, want %d / 2", n, rec.n(), f.want)
			}
		})
	}
}

func TestCountTokens_RetryPolicyBounded(t *testing.T) {
	fastBackoff(t)
	srv, rec := countServer(t, obsStep{status: 503, body: `{}`})
	sdk := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 2}),
		WithProvider("anthropic", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c, err := sdk.Chat("anthropic", "claude-x")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.CountTokens(context.Background(), countReq()); err == nil {
		t.Fatal("want error")
	}
	if rec.n() != 2 {
		t.Errorf("attempts = %d, want 2", rec.n())
	}
}

func TestCountTokens_WholeCallTimeoutBoundsRetries(t *testing.T) {
	srv, rec := countServer(t, obsStep{status: 500, body: `{}`})
	cc := countClient(t, "anthropic", FormatAnthropic, "claude-x", srv)
	old := backoffUnit
	backoffUnit = 40 * time.Millisecond
	t.Cleanup(func() { backoffUnit = old })
	cc.SetRequestTimeout(150 * time.Millisecond)
	start := time.Now()
	_, err := cc.CountTokens(context.Background(), countReq())
	if err == nil {
		t.Fatal("want error")
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Errorf("took %v, whole-call budget not enforced", d)
	}
	if rec.n() >= 8 {
		t.Errorf("attempts = %d, budget should cut retries short", rec.n())
	}
}

func TestCountTokens_HangingServerHitsBudget(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}))
	t.Cleanup(srv.Close)
	cc := countClient(t, "gemini", FormatGemini, "gemini-x", srv)
	cc.SetRequestTimeout(120 * time.Millisecond)
	start := time.Now()
	if _, err := cc.CountTokens(context.Background(), countReq()); err == nil {
		t.Fatal("want deadline error")
	}
	if time.Since(start) > 3*time.Second {
		t.Error("timeout not applied")
	}
}

func TestCountTokens_PerRequestHeaders(t *testing.T) {
	for _, f := range []struct {
		id     string
		format Format
		model  string
		ok     string
	}{
		{"anthropic", FormatAnthropic, "claude-x", anCountOK},
		{"gemini", FormatGemini, "gemini-x", gmCountOK},
	} {
		t.Run(f.id, func(t *testing.T) {
			srv, rec := countServer(t, obsStep{status: 500, body: `{}`}, obsStep{status: 200, body: f.ok})
			cc := countClient(t, f.id, f.format, f.model, srv)
			req := countReq()
			req.Headers = map[string]string{"X-Trace": "t-1"}
			if _, err := cc.CountTokens(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			if rec.n() != 2 {
				t.Fatalf("attempts = %d", rec.n())
			}
			for i, h := range rec.headers {
				if h.Get("X-Trace") != "t-1" {
					t.Errorf("attempt %d: X-Trace = %q", i, h.Get("X-Trace"))
				}
			}
		})
	}
}

func TestCountTokens_AttemptObserverOperation(t *testing.T) {
	for _, f := range []struct {
		id     string
		format Format
		model  string
		ok     string
	}{
		{"anthropic", FormatAnthropic, "claude-x", anCountOK},
		{"gemini", FormatGemini, "gemini-x", gmCountOK},
	} {
		t.Run(f.id, func(t *testing.T) {
			fastBackoff(t)
			srv, _ := countServer(t, obsStep{status: 500, body: `{}`}, obsStep{status: 200, body: f.ok})
			col := &obsCollector{}
			sdk := New(WithAttemptObserver(col.fn),
				WithProvider(f.id, WithAPIKey(obsKey), WithBaseURL(srv.URL)))
			c, err := sdk.Chat(f.id, f.model)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.CountTokens(context.Background(), countReq()); err != nil {
				t.Fatal(err)
			}
			evs := col.all()
			if len(evs) != 2 {
				t.Fatalf("events = %d, want 2", len(evs))
			}
			for i, e := range evs {
				if e.Operation != "count_tokens" || e.Model != f.model || e.Attempt != i+1 || e.Provider != f.id {
					t.Errorf("event %d = %+v", i, e)
				}
				obsNoKey(t, e)
			}
			if evs[0].Status != 500 || evs[1].Status != 200 {
				t.Errorf("statuses = %d, %d", evs[0].Status, evs[1].Status)
			}
		})
	}
}

func TestCountTokens_UnusableSuccessBody(t *testing.T) {
	cases := []struct {
		name string
		f    Format
		id   string
		body string
	}{
		{"anthropic missing", FormatAnthropic, "anthropic", `{}`},
		{"anthropic negative", FormatAnthropic, "anthropic", `{"input_tokens":-3}`},
		{"anthropic bad json", FormatAnthropic, "anthropic", `not json`},
		{"gemini missing", FormatGemini, "gemini", `{"cachedContentTokenCount":1}`},
		{"gemini negative", FormatGemini, "gemini", `{"totalTokens":-1}`},
		{"gemini bad json", FormatGemini, "gemini", `{`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, rec := countServer(t, obsStep{status: 200, body: tc.body})
			cc := countClient(t, tc.id, tc.f, "m", srv)
			n, err := cc.CountTokens(context.Background(), countReq())
			var ae *APIError
			if !errors.As(err, &ae) || ae.Status != 200 {
				t.Fatalf("err = %v, want *APIError{Status:200}", err)
			}
			if n != 0 {
				t.Errorf("count = %d, want 0", n)
			}
			if rec.n() != 1 {
				t.Errorf("attempts = %d, want 1 (terminal, not retried)", rec.n())
			}
			if strings.Contains(err.Error(), "sk-count-secret") {
				t.Error("error leaks API key")
			}
		})
	}
}

func TestCountTokens_ZeroCountIsValid(t *testing.T) {
	srv, _ := countServer(t, obsStep{status: 200, body: `{"input_tokens":0}`})
	cc := countClient(t, "anthropic", FormatAnthropic, "m", srv)
	n, err := cc.CountTokens(context.Background(), countReq())
	if err != nil || n != 0 {
		t.Errorf("n = %d err = %v, want 0 / nil for an explicit zero", n, err)
	}
}

func TestCountTokens_ErrorNeverLeaksKey(t *testing.T) {
	srv, _ := countServer(t, obsStep{status: 401, body: `{"error":{"message":"bad key sk-count-secret"}}`})
	cc := countClient(t, "anthropic", FormatAnthropic, "m", srv)
	_, err := cc.CountTokens(context.Background(), countReq())
	if err == nil {
		t.Fatal("want error")
	}
	var ae *APIError
	if !errors.As(err, &ae) || ae.Status != 401 {
		t.Errorf("err = %v, want 401 APIError", err)
	}
}

func TestCountTokens_RequestNotMutated(t *testing.T) {
	srv, _ := countServer(t, obsStep{status: 200, body: anCountOK})
	cc := countClient(t, "anthropic", FormatAnthropic, "claude-x", srv)
	req := countReq()
	before, _ := json.Marshal(req)
	if _, err := cc.CountTokens(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(req)
	if string(before) != string(after) {
		t.Errorf("request mutated: %s -> %s", before, after)
	}
}
