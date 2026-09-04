package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// httptestNewServer is a shorthand for the handler-shaped server used
// throughout these tests.
func httptestNewServer(h func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(h))
}

// newListModels drives providerClient.listModels against a test server.
func newListModels(baseURL string, f Format) ([]Model, error) {
	pc := newProviderClient(ProviderConfig{ID: "x", Format: f, BaseURL: baseURL, APIKey: "k"}, nil, nil)
	return pc.listModels(context.Background())
}

// ── error rendering ──────────────────────────────────────────────────────

func TestErrorStrings(t *testing.T) {
	if got := (&ConfigError{Msg: "bad"}).Error(); got != "llm: config: bad" {
		t.Errorf("ConfigError = %q", got)
	}
	ae := &APIError{Provider: "openai", Status: 429, Code: "rate_limit", Message: "slow", Retryable: true}
	s := ae.Error()
	for _, want := range []string{"openai", "429", "rate_limit", "slow", "retryable"} {
		if !strings.Contains(s, want) {
			t.Errorf("APIError.Error() = %q, missing %q", s, want)
		}
	}
	minimal := (&APIError{Provider: "x", Status: 500}).Error()
	if strings.Contains(minimal, "(") || strings.Contains(minimal, "[retryable]") {
		t.Errorf("minimal APIError = %q, want bare status form", minimal)
	}
	rl := (&RateLimitError{APIError: APIError{Provider: "x", Message: "m"}, Attempts: 2, RetryAfter: 3 * time.Second}).Error()
	for _, want := range []string{"x", "2 attempts", "3s", "m"} {
		if !strings.Contains(rl, want) {
			t.Errorf("RateLimitError = %q, missing %q", rl, want)
		}
	}
	rlBare := (&RateLimitError{APIError: APIError{Provider: "x"}, Attempts: 1}).Error()
	if strings.Contains(rlBare, "retry after") {
		t.Errorf("RateLimitError without hint = %q", rlBare)
	}
	if got := (&StreamAbortedError{Reason: errors.New("stop")}).Error(); got != "llm: stream aborted by consumer: stop" {
		t.Errorf("StreamAbortedError = %q", got)
	}
	if got := errSSEOversized.Error(); got != "llm: sse: frame exceeds size limit" {
		t.Errorf("sseError = %q", got)
	}
	ca := &consumerAbort{err: errors.New("inner")}
	if !errors.As(func() error { return ca }(), new(*consumerAbort)) || ca.Unwrap().Error() != "inner" {
		t.Error("consumerAbort unwrap broken")
	}
}

// ── options ──────────────────────────────────────────────────────────────

func TestSDKOptions(t *testing.T) {
	rt := &countingRoundTripper{}
	sdk := New(
		WithRequestTimeout(7*time.Second),
		WithModelCacheTTL(0),
		WithTransport(rt),
		WithProvider("mygw",
			WithFormat(FormatOpenAI),
			WithBaseURL("http://127.0.0.1:1"),
			WithAPIKey("k"),
			WithQuirks(Quirks{ReasoningEffort: true}),
			WithEnvKeys("A_KEY", "B_KEY"),
		),
	)
	cc, err := sdk.Chat("mygw", "m")
	if err != nil {
		t.Fatal(err)
	}
	if cc.RequestTimeout() != 7*time.Second {
		t.Errorf("RequestTimeout = %v, want 7s from WithRequestTimeout", cc.RequestTimeout())
	}
	cfg := sdk.Providers()[0].Config()
	if !cfg.Quirks.ReasoningEffort {
		t.Error("WithQuirks not applied")
	}
	if len(cfg.EnvKeys) != 2 || cfg.EnvKeys[0] != "A_KEY" {
		t.Errorf("WithEnvKeys = %v", cfg.EnvKeys)
	}
	// WithTransport: a chat call must flow through the replaced transport.
	// A short context keeps the (retryable) transport failure cheap.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, srvErr := cc.Call(ctx, &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if srvErr == nil {
		t.Fatal("expected transport-level failure from unreachable base URL")
	}
	if rt.calls == 0 {
		t.Error("WithTransport ignored: request bypassed the replaced transport")
	}
	// nil guards must not panic.
	New(WithTransport(nil), WithRequestTimeout(0), WithEnv(nil))
}

type countingRoundTripper struct{ calls int }

func (r *countingRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	r.calls++
	return nil, errors.New("unreachable by design")
}

// ── provider config validation + env resolution ──────────────────────────

func TestValidateProviderConfigTable(t *testing.T) {
	cases := []struct {
		name    string
		cfg     ProviderConfig
		wantErr string
	}{
		{"ok", ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "https://api.example.com"}, ""},
		{"empty id", ProviderConfig{Format: FormatOpenAI, BaseURL: "https://x"}, "ID is empty"},
		{"space id", ProviderConfig{ID: "a b", Format: FormatOpenAI, BaseURL: "https://x"}, "whitespace"},
		{"bad format", ProviderConfig{ID: "x", Format: Format("nope"), BaseURL: "https://x"}, "unknown format"},
		{"empty url", ProviderConfig{ID: "x", Format: FormatOpenAI}, "empty base URL"},
		{"bad scheme", ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "ftp://x"}, "http://"},
	}
	for _, tc := range cases {
		err := validateProviderConfig(tc.cfg)
		if tc.wantErr == "" {
			if err != nil {
				t.Errorf("%s: %v", tc.name, err)
			}
			continue
		}
		if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
			t.Errorf("%s: err = %v, want containing %q", tc.name, err, tc.wantErr)
		}
	}
}

func TestResolveEnvProvider(t *testing.T) {
	base := ProviderConfig{ID: "zai", Format: FormatOpenAI, BaseURL: "https://api.z.ai/api/paas/v4", EnvKeys: []string{"ZAI_API_KEY", "ALIAS_KEY"}}

	if _, ok := resolveEnvProvider(base, nil); ok {
		t.Error("nil lookup must not authenticate")
	}
	if _, ok := resolveEnvProvider(base, func(string) (string, bool) { return "", false }); ok {
		t.Error("missing keys must not authenticate")
	}
	// Primary beats alias; whitespace is trimmed.
	cfg, ok := resolveEnvProvider(base, func(k string) (string, bool) {
		if k == "ZAI_API_KEY" {
			return "  primary  ", true
		}
		return "alias", true
	})
	if !ok || cfg.APIKey != "primary" {
		t.Errorf("primary = %q ok=%v", cfg.APIKey, ok)
	}
	// Alias wins when primary absent; <ID>_BASE_URL override applies.
	cfg, ok = resolveEnvProvider(base, func(k string) (string, bool) {
		switch k {
		case "ALIAS_KEY":
			return "alias-key", true
		case "ZAI_BASE_URL":
			return "https://custom.endpoint/v4", true
		}
		return "", false
	})
	if !ok || cfg.APIKey != "alias-key" || cfg.BaseURL != "https://custom.endpoint/v4" {
		t.Errorf("alias/baseURL = %q %q ok=%v", cfg.APIKey, cfg.BaseURL, ok)
	}
	// Blank-string values are ignored.
	if _, ok = resolveEnvProvider(base, func(k string) (string, bool) { return "   ", true }); ok {
		t.Error("whitespace-only key must not authenticate")
	}
	if envBaseURLKey("zai") != "ZAI_BASE_URL" {
		t.Errorf("envBaseURLKey = %q", envBaseURLKey("zai"))
	}
}

// ── ProviderConfig rendering ─────────────────────────────────────────────

func TestProviderConfigString(t *testing.T) {
	withKey := ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: "https://api.openai.com/v1", APIKey: "secret"}
	if s := withKey.String(); !strings.Contains(s, "Authenticated:true") || strings.Contains(s, "secret") {
		t.Errorf("String() = %q (must flag auth without leaking the key)", s)
	}
	without := ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: "https://api.openai.com/v1"}
	if s := without.String(); !strings.Contains(s, "Authenticated:false") {
		t.Errorf("String() = %q", s)
	}
}

// ── model listing orchestration ──────────────────────────────────────────

func TestListModelsDispatchAndRetries(t *testing.T) {
	old := backoffUnit
	backoffUnit = time.Millisecond
	t.Cleanup(func() { backoffUnit = old })

	// Retryable failures exhaust the 3-attempt budget and wrap the cause.
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		w.WriteHeader(500)
	})
	if _, err := newListModels(srv.URL, FormatOpenAI); err == nil || !strings.Contains(err.Error(), "list models failed") {
		t.Fatalf("err = %v, want exhausted-listing wrap", err)
	}
	if n != 3 {
		t.Errorf("attempts = %d, want 3", n)
	}
	// Non-retryable failures return immediately.
	n = 0
	srv2 := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		w.WriteHeader(401)
	})
	if _, err := newListModels(srv2.URL, FormatOpenAI); err == nil {
		t.Fatal("expected 401 to surface")
	}
	if n != 1 {
		t.Errorf("attempts after 401 = %d, want 1 (no retry on definitive error)", n)
	}
	// Gemini dispatch arm.
	srv3 := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"models":[{"name":"models/gem","displayName":"Gem","inputTokenLimit":10,"outputTokenLimit":5,"supportedGenerationMethods":["generateContent"]}]}`))
	})
	got, err := newListModels(srv3.URL, FormatGemini)
	if err != nil || len(got) != 1 || got[0].ID != "gem" {
		t.Fatalf("gemini listing = %+v err %v", got, err)
	}
	// Anthropic dispatch arm.
	srv4 := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"data":[{"id":"claude-x"}],"has_more":false}`))
	})
	got, err = newListModels(srv4.URL, FormatAnthropic)
	if err != nil || len(got) != 1 || got[0].ID != "claude-x" {
		t.Fatalf("anthropic listing = %+v err %v", got, err)
	}
}

func TestProviderListModelsCache(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		w.Write([]byte(`{"data":[{"id":"m"}]}`))
	})
	sdk := New(WithProvider("openai", WithBaseURL(srv.URL), WithAPIKey("k")))
	p, _ := sdk.Provider("openai")

	if _, err := p.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := p.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("cached: server hits = %d, want 1", n)
	}
	if _, err := p.ListModels(context.Background(), ForceRefresh()); err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("ForceRefresh: server hits = %d, want 2", n)
	}

	// TTL 0 disables caching.
	sdk2 := New(WithModelCacheTTL(0), WithProvider("openai", WithBaseURL(srv.URL), WithAPIKey("k")))
	p2, _ := sdk2.Provider("openai")
	_, _ = p2.ListModels(context.Background())
	_, _ = p2.ListModels(context.Background())
	if n != 4 {
		t.Errorf("ttl0: server hits = %d, want 4", n)
	}

	// Unauthenticated listing is a config error, not a network call.
	sdk3 := New(WithEnv(func(string) (string, bool) { return "", false }))
	p3, _ := sdk3.Provider("openai")
	if _, err := p3.ListModels(context.Background()); err == nil {
		t.Fatal("unauthenticated ListModels must fail")
	}
}
