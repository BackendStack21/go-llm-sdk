package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// Regression: learn-once fallbacks must live on the Provider (shared), not
// on each ChatClient — every sdk.Chat() minted a fresh client that re-paid
// the provider's rejection round-trip.
func TestLearnOnceSharedAcrossChatClients(t *testing.T) {
	var mu sync.Mutex
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, string(b))
		n := len(bodies)
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"'stream_options.include_usage' is not supported by this model"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	sdk := New(WithProvider("openai",
		WithFormat(FormatOpenAI), WithBaseURL(srv.URL), WithAPIKey("k")))
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}

	c1, err := sdk.Chat("openai", "test-model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c1.CallStream(context.Background(), req, func(Delta) error { return nil }); err != nil {
		t.Fatalf("first client stream: %v", err)
	}

	c2, err := sdk.Chat("openai", "test-model") // fresh client, same provider
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c2.CallStream(context.Background(), req, func(Delta) error { return nil }); err != nil {
		t.Fatalf("second client stream: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	streamOptionsReqs := 0
	for _, b := range bodies {
		if strings.Contains(b, "stream_options") {
			streamOptionsReqs++
		}
	}
	if streamOptionsReqs != 1 {
		t.Errorf("%d/%d requests carried stream_options, want exactly 1 (the learning request)", streamOptionsReqs, len(bodies))
	}
	if len(bodies) != 3 {
		t.Errorf("requests = %d, want 3 (learn once, then two clean streams)", len(bodies))
	}
}

// Regression: overriding a built-in provider must run the same wiring-time
// validation as a custom registration — config typos should fail loudly at
// startup, not per request.
func TestWithProviderOverrideInvalidConfigRejected(t *testing.T) {
	sdk := New(WithProvider("openai", WithBaseURL("htps://typo.example"), WithAPIKey("k")))
	_, err := sdk.Chat("openai", "gpt-4o")
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v (%T), want *ConfigError at wiring time", err, err)
	}
}

// The unauthenticated-provider hint must name the real env var
// (DEEPSEEK_API_KEY), not the mixed-case literal provider id.
func TestUnauthenticatedHintUsesUppercaseEnvName(t *testing.T) {
	sdk := New(WithEnv(func(string) (string, bool) { return "", false }))
	_, err := sdk.Chat("deepseek", "m")
	if err == nil || !strings.Contains(err.Error(), "DEEPSEEK_API_KEY") {
		t.Fatalf("err = %v, want hint naming DEEPSEEK_API_KEY", err)
	}
}

// FromEnv is the production entry point; it must resolve keys via the real
// environment.
func TestFromEnvRegistersAuthenticatedProviders(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	sdk := New(FromEnv())
	found := false
	for _, p := range sdk.Providers() {
		if p.ID() == "deepseek" {
			found = p.Authenticated()
		}
	}
	if !found {
		t.Fatal("DEEPSEEK_API_KEY set but deepseek provider not authenticated")
	}
}
