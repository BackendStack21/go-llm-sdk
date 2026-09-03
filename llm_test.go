package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestSDK_ListModels_CacheAndForceRefresh(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		fmt.Fprint(w, `{"data":[{"id":"m1"}]}`)
	}))
	defer srv.Close()

	env := map[string]string{"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": srv.URL}
	sdk := New(WithEnv(func(k string) (string, bool) { v, ok := env[k]; return v, ok }))
	p, err := sdk.Provider("openai")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := p.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := p.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("HTTP hits = %d, want 1 (second call served from cache)", got)
	}
	if _, err := p.ListModels(context.Background(), ForceRefresh()); err != nil {
		t.Fatal(err)
	}
	if got := hits.Load(); got != 2 {
		t.Fatalf("HTTP hits = %d, want 2 after ForceRefresh", got)
	}
}

func TestSDK_ListModels_CacheDisabled(t *testing.T) {
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		fmt.Fprint(w, `{"data":[{"id":"m1"}]}`)
	}))
	defer srv.Close()

	sdk := New(
		WithModelCacheTTL(0),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)),
	)
	p, _ := sdk.Provider("openai")
	for i := 0; i < 3; i++ {
		if _, err := p.ListModels(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if got := hits.Load(); got != 3 {
		t.Fatalf("HTTP hits = %d, want 3 (cache disabled)", got)
	}
}

func TestSDK_ChatCustomGateway_EndToEnd(t *testing.T) {
	var gotPath, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"pong"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	sdk := New(WithProvider("my-gateway",
		WithFormat(FormatOpenAI),
		WithBaseURL(srv.URL),
		WithAPIKey("local-key"),
	))
	chat, err := sdk.Chat("my-gateway", "some-model")
	if err != nil {
		t.Fatal(err)
	}
	res, err := chat.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "ping"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "pong" {
		t.Fatalf("content = %q", res.Content)
	}
	if gotPath != "/chat/completions" {
		t.Errorf("path = %s", gotPath)
	}
	if gotAuth != "Bearer local-key" {
		t.Errorf("auth = %q", gotAuth)
	}
}

func TestSDK_GeminiStreaming_EndToEnd(t *testing.T) {
	var gotPath, gotKey, gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hey\"}]}}]}\n\n")
		fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{}]},\"finishReason\":\"STOP\"}]}\n\n")
	}))
	defer srv.Close()

	sdk := New(WithProvider("gemini", WithAPIKey("gk"), WithBaseURL(srv.URL)))
	chat, err := sdk.Chat("gemini", "gemini-2.5-flash")
	if err != nil {
		t.Fatal(err)
	}
	var contents []string
	res, err := chat.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(d Delta) error {
		if d.Kind == DeltaContent {
			contents = append(contents, d.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "hey" || len(contents) != 1 {
		t.Fatalf("result = %+v deltas = %v", res, contents)
	}
	if gotPath != "/v1beta/models/gemini-2.5-flash:streamGenerateContent" {
		t.Errorf("path = %s", gotPath)
	}
	if gotQuery != "alt=sse" {
		t.Errorf("query = %s", gotQuery)
	}
	if gotKey != "gk" {
		t.Errorf("key header = %q", gotKey)
	}
}

func TestSDK_AnthropicHeaders_EndToEnd(t *testing.T) {
	var gotKey, gotVersion string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		fmt.Fprint(w, `{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	sdk := New(WithProvider("anthropic", WithAPIKey("ak"), WithBaseURL(srv.URL)))
	chat, _ := sdk.Chat("anthropic", "claude-sonnet-4")
	res, err := chat.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "q"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "hi" || res.FinishReason != FinishStop {
		t.Fatalf("result = %+v", res)
	}
	if gotKey != "ak" || gotVersion != "2023-06-01" {
		t.Fatalf("headers: key=%q version=%q", gotKey, gotVersion)
	}
}

func TestChatClient_TimeoutAccessors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()

	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	chat, _ := sdk.Chat("openai", "m")
	if chat.RequestTimeout() != DefaultTimeout {
		t.Fatalf("default timeout = %v", chat.RequestTimeout())
	}
	chat.SetRequestTimeout(7 * time.Second)
	if chat.RequestTimeout() != 7*time.Second {
		t.Fatalf("timeout = %v, want 7s", chat.RequestTimeout())
	}
	if chat.Model() != "m" || chat.ProviderID() != "openai" {
		t.Fatalf("accessors: %s %s", chat.ProviderID(), chat.Model())
	}
}

func TestChatClient_NilDeltaHandlerFallsBackToBuffered(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// If the client streamed, Accept would be event-stream; serve JSON.
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"buf"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	chat, _ := sdk.Chat("openai", "m")
	res, err := chat.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "q"}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "buf" {
		t.Fatalf("content = %q", res.Content)
	}
}
