package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func m5Provider(t *testing.T, id string, srv *httptest.Server, opts ...Option) *Provider {
	t.Helper()
	fastBackoff(t)
	sdk := New(append(opts, WithProvider(id, WithAPIKey("k"), WithBaseURL(srv.URL)))...)
	p, err := sdk.Provider(id)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// M5 bug 2: a bare top-level JSON array (Together AI) is a valid listing.
func TestM5ListModels_OpenAIBareArray(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `[{"id":"a","created":1700000000,"context_length":8192},{"id":"b"}]`)
	}))
	t.Cleanup(srv.Close)
	models, err := m5Provider(t, "openai", srv).ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 2 || models[0].ID != "a" || models[0].ContextWindow != 8192 || models[1].ID != "b" {
		t.Fatalf("models = %+v", models)
	}
}

// M5 bug 3: has_more with no last_id cannot be paged; that is an error,
// never a silent partial success.
func TestM5ListModels_AnthropicHasMoreWithoutLastID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{"data":[{"id":"claude-x","display_name":"X","created_at":"2025-01-01T00:00:00Z"}],"has_more":true,"last_id":""}`)
	}))
	t.Cleanup(srv.Close)
	models, err := m5Provider(t, "anthropic", srv).ListModels(context.Background())
	if err == nil {
		t.Fatalf("want error for has_more without last_id, got partial %+v", models)
	}
	if models != nil {
		t.Errorf("models = %+v alongside error", models)
	}
}

// M5 bug 4a: callers own what ListModels returns; mutating a Capabilities
// slice must never reach the cache or other callers (first and cached read).
func TestM5ListModels_CapabilitiesNotAliased(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{"models":[{"name":"models/g1","displayName":"G","supportedGenerationMethods":["generateContent","embedContent"]}]}`)
	}))
	t.Cleanup(srv.Close)
	p := m5Provider(t, "gemini", srv)
	ctx := context.Background()
	first, err := p.ListModels(ctx)
	if err != nil || len(first) != 1 || len(first[0].Capabilities) != 2 {
		t.Fatalf("first = %+v, %v", first, err)
	}
	first[0].Capabilities[0] = "CORRUPTED"
	second, err := p.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if second[0].Capabilities[0] != "generateContent" {
		t.Errorf("mutation of first result leaked into cache: %v", second[0].Capabilities)
	}
	second[0].Capabilities[1] = "CORRUPTED"
	third, _ := p.ListModels(ctx)
	if third[0].Capabilities[1] != "embedContent" {
		t.Errorf("mutation of cached result leaked into cache: %v", third[0].Capabilities)
	}
}

// M5 bug 4b: an empty successful listing is cached like any other, on every
// format (Anthropic/Gemini build the list from a nil slice).
func TestM5ListModels_EmptyListCached(t *testing.T) {
	cases := []struct{ id, body string }{
		{"openai", `{"data":[]}`},
		{"anthropic", `{"data":[],"has_more":false,"last_id":""}`},
		{"gemini", `{"models":[]}`},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				_, _ = fmt.Fprint(w, tc.body)
			}))
			t.Cleanup(srv.Close)
			p := m5Provider(t, tc.id, srv)
			for i := 0; i < 3; i++ {
				models, err := p.ListModels(context.Background())
				if err != nil || len(models) != 0 {
					t.Fatalf("call %d: %v, %v", i, models, err)
				}
			}
			if n := hits.Load(); n != 1 {
				t.Errorf("upstream hits = %d, want 1 (empty list must be cached)", n)
			}
		})
	}
}

// M5 bug 5: ListModels follows WithRetryPolicy; unconfigured it keeps the
// documented 3 attempts.
func TestM5ListModels_RetryPolicyAttempts(t *testing.T) {
	cases := []struct {
		name string
		opts []Option
		want int32
	}{
		{"default 3 attempts", nil, 3},
		{"MaxAttempts 2", []Option{WithRetryPolicy(RetryPolicy{MaxAttempts: 2})}, 2},
		{"MaxAttempts 5", []Option{WithRetryPolicy(RetryPolicy{MaxAttempts: 5})}, 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, n := countingServer(t, http.StatusInternalServerError, nil)
			_, err := m5Provider(t, "openai", srv, tc.opts...).ListModels(context.Background())
			if err == nil {
				t.Fatal("want error")
			}
			if got := n.Load(); got != tc.want {
				t.Errorf("attempts = %d, want %d", got, tc.want)
			}
		})
	}
}

// M5 bug 5: Retry-After on a retryable status is honoured, not replaced by
// a computed backoff.
func TestM5ListModels_HonoursRetryAfter(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
			return
		}
		_, _ = fmt.Fprint(w, `{"data":[{"id":"a"}]}`)
	}))
	t.Cleanup(srv.Close)
	p := m5Provider(t, "openai", srv)
	start := time.Now()
	models, err := p.ListModels(context.Background())
	if err != nil || len(models) != 1 {
		t.Fatalf("models=%v err=%v", models, err)
	}
	if el := time.Since(start); el < 900*time.Millisecond {
		t.Errorf("retried after %v, want >= ~1s per Retry-After", el)
	}
}

// M5 bug 5: WithRequestTimeout bounds the whole ListModels call, retries
// included.
func TestM5ListModels_RequestTimeoutBoundsCall(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		tm := time.NewTimer(3 * time.Second)
		defer tm.Stop()
		select {
		case <-r.Context().Done():
		case <-tm.C:
		}
	}))
	t.Cleanup(srv.Close)
	p := m5Provider(t, "openai", srv,
		WithRequestTimeout(300*time.Millisecond),
		WithRetryPolicy(RetryPolicy{MaxAttempts: 5}))
	start := time.Now()
	_, err := p.ListModels(context.Background())
	el := time.Since(start)
	if err == nil {
		t.Fatal("want timeout error")
	}
	if el > 1500*time.Millisecond {
		t.Errorf("call took %v, want ~300ms whole-call budget", el)
	}
	if errors.Is(err, context.Canceled) {
		t.Errorf("err = %v", err)
	}
}
