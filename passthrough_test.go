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
)

// WithHeaders reaches every request the provider makes; an empty value
// removes a header the SDK would otherwise send (e.g. Authorization for
// api-key gateways).
func TestWithHeaders_AllEntryPoints(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]http.Header{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		key := r.URL.Path
		if strings.Contains(r.Header.Get("Accept"), "event-stream") {
			key += "#stream"
		}
		seen[key] = r.Header.Clone()
		mu.Unlock()
		switch {
		case strings.HasSuffix(r.URL.Path, "/models"):
			_, _ = fmt.Fprint(w, `{"data":[{"id":"m"}]}`)
		case strings.HasSuffix(r.URL.Path, "/audio/speech"):
			w.Header().Set("Content-Type", "audio/mpeg")
			_, _ = w.Write([]byte("ID3"))
		case strings.HasSuffix(r.URL.Path, "/audio/transcriptions"):
			_, _ = fmt.Fprint(w, `{"text":"hi"}`)
		case strings.Contains(r.Header.Get("Accept"), "event-stream"):
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
		default:
			_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
		}
	}))
	t.Cleanup(srv.Close)

	sdk := New(WithProvider("gw", WithFormat(FormatOpenAI), WithBaseURL(srv.URL), WithAPIKey("secret-key"),
		WithHeaders(map[string]string{"X-Title": "app", "api-key": "secret-key"}),
		WithHeaders(map[string]string{"Authorization": "", "HTTP-Referer": "https://example.test"})))
	c, err := sdk.Chat("gw", "m")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if _, err := c.Call(ctx, userMsg()); err != nil {
		t.Fatal(err)
	}
	if _, err := c.CallStream(ctx, userMsg(), func(Delta) error { return nil }); err != nil {
		t.Fatal(err)
	}
	p, _ := sdk.Provider("gw")
	if _, err := p.ListModels(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := sdk.Speak(ctx, "gw", "tts", SpeakRequest{Text: "hi", Voice: "v"}); err != nil {
		t.Fatal(err)
	}
	if _, err := sdk.Transcribe(ctx, "gw", "stt", TranscribeRequest{Audio: []byte("a")}); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, path := range []string{"/chat/completions", "/chat/completions#stream", "/models", "/audio/speech", "/audio/transcriptions"} {
		h, ok := seen[path]
		if !ok {
			t.Errorf("%s: no request seen", path)
			continue
		}
		if h.Get("X-Title") != "app" || h.Get("HTTP-Referer") != "https://example.test" || h.Get("api-key") != "secret-key" {
			t.Errorf("%s: custom headers missing: %v", path, h)
		}
		if _, ok := h["Authorization"]; ok {
			t.Errorf("%s: empty value must remove Authorization", path)
		}
	}
	// Headers never leak through String().
	if s := p.Config().String(); strings.Contains(s, "secret-key") || strings.Contains(s, "app") {
		t.Errorf("String() leaks headers: %s", s)
	}
}

// WithHeaders copies its map: later caller mutation never changes the SDK.
func TestWithHeaders_CopiesMap(t *testing.T) {
	h := map[string]string{"X-A": "1"}
	sdk := New(WithProvider("openai", WithHeaders(h)))
	h["X-A"] = "2"
	p, _ := sdk.Provider("openai")
	if got := p.Config().Headers["X-A"]; got != "1" {
		t.Errorf("header = %q, want 1", got)
	}
}

func TestChatRequestExtra(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "x"}},
		Extra:    map[string]any{"user": "u-1", "metadata": map[string]any{"k": "v"}, "max_tokens": 7},
	}
	req.MaxTokens = 100
	for _, f := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
		m, _ := buildBody(t, f, "m", req)
		if m["user"] != "u-1" || m["metadata"].(map[string]any)["k"] != "v" {
			t.Errorf("%s: extra fields missing: %v", f, m)
		}
		if f != FormatGemini && m["max_tokens"] != float64(7) {
			t.Errorf("%s: Extra must override SDK fields, max_tokens = %v", f, m["max_tokens"])
		}
	}
	// Responses path too.
	rs := *req
	rs.Tools = []ToolDef{{Name: "f"}}
	m, url := buildBody(t, FormatOpenAI, "gpt-5.6", &rs)
	if !strings.HasSuffix(url, "/responses") || m["user"] != "u-1" {
		t.Errorf("responses extra: %s %v", url, m)
	}

	bad := []map[string]any{
		{"stream": true}, // owned by Call vs CallStream
		{"x": func() {}}, // not JSON
		{"": 1},          // empty key
	}
	for i, extra := range bad {
		r := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, Extra: extra}
		pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "https://x"}, nil, nil)
		_, _, err := pc.buildChatRequest(r, "m", false)
		var ce *ConfigError
		if !errors.As(err, &ce) {
			t.Errorf("case %d: err = %v, want ConfigError", i, err)
		}
	}
}

// Extra reaches the wire on a real call.
func TestChatRequestExtra_OnTheWire(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)
	c := newTestClient(t, ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: srv.URL}, srv)
	if _, err := c.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, Extra: map[string]any{"service_tier": "flex"}}); err != nil {
		t.Fatal(err)
	}
	if body["service_tier"] != "flex" {
		t.Errorf("body = %v", body)
	}
}
