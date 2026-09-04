//go:build e2e

package llm

// End-to-end tests against the live DeepSeek API. Tag-gated so they never
// build into CI or a plain `go test ./...`:
//
//	go test -tags e2e -run 'TestE2E' -timeout 15m -v .
//
// Credentials: DEEPSEEK_API_KEY from the process environment, or a
// repo-root .env file (KEY=VALUE lines, optional `export ` prefix and
// double quotes). The file is parsed by loadDotEnv; neither the file nor
// the key is ever logged — the SDK guarantees API keys stay out of all
// error text. Tests skip cleanly when no key resolves.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// loadDotEnv parses a .env file and sets any variable not already present
// in the process environment (environment wins over file). Values may be
// wrapped in single or double quotes; `KEY=VALUE` and `export KEY=VALUE`
// forms are accepted; `#` starts a comment.
func loadDotEnv(t *testing.T, path string) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		return // no .env: environment only
	}
	for i, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "export "))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			t.Logf(".env line %d: not KEY=VALUE, ignored", i+1)
			continue
		}
		k = strings.TrimSpace(k)
		v = strings.Trim(strings.TrimSpace(v), `"'`)
		if k == "" {
			continue
		}
		if _, exists := os.LookupEnv(k); !exists {
			if err := os.Setenv(k, v); err != nil {
				t.Fatalf("setenv %s: %v", k, err)
			}
		}
	}
}

// e2eKey resolves the DeepSeek credentials, skipping the test when absent.
func e2eKey(t *testing.T) string {
	t.Helper()
	loadDotEnv(t, ".env")
	key := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY"))
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY not set (env or .env); skipping live e2e")
	}
	return key
}

// e2eChat builds a chat client against the live DeepSeek endpoint via the
// production FromEnv path.
func e2eChat(t *testing.T, model string) *ChatClient {
	t.Helper()
	e2eKey(t)
	sdk := New(FromEnv())
	cc, err := sdk.Chat("deepseek", model)
	if err != nil {
		t.Fatalf("Chat(deepseek, %s): %v", model, err)
	}
	return cc
}

func e2eCtx(t *testing.T, d time.Duration) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), d)
	t.Cleanup(cancel)
	return ctx
}

// FromEnv + registry discovery: the .env key must land in the deepseek
// provider through the production entry point.
func TestE2EDiscoveryAndListModels(t *testing.T) {
	e2eKey(t)
	sdk := New(FromEnv())
	p, err := sdk.Provider("deepseek")
	if err != nil {
		t.Fatal(err)
	}
	if !p.Authenticated() {
		t.Fatal("deepseek provider resolved but not authenticated")
	}
	models, err := p.ListModels(e2eCtx(t, 30*time.Second))
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("expected at least one accessible model")
	}
	for _, m := range models {
		if m.ID == "" {
			t.Errorf("model with empty ID: %+v", m)
		}
	}
}

// Buffered chat: content, finish reason, and live token accounting.
func TestE2EBufferedChat(t *testing.T) {
	cc := e2eChat(t, "deepseek-chat")
	res, err := cc.Call(e2eCtx(t, 120*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "Reply with exactly: OK"}},
		MaxTokens: 20,
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(strings.ToUpper(res.Content), "OK") {
		t.Errorf("content = %q, want it to contain OK", res.Content)
	}
	if res.FinishReason != FinishStop {
		t.Errorf("finish = %q, want stop", res.FinishReason)
	}
	if res.Usage.PromptTokens == 0 || res.Usage.CompletionTokens == 0 {
		t.Errorf("usage = %+v, want live token counts", res.Usage)
	}
}

// Streaming: deltas fold into the final result, usage arrives via the
// final chunk, finish reason is canonical.
func TestE2EStreamingChat(t *testing.T) {
	cc := e2eChat(t, "deepseek-chat")
	var content strings.Builder
	sawContent := false
	res, err := cc.CallStream(e2eCtx(t, 120*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "Count from 1 to 5, digits only."}},
		MaxTokens: 60,
	}, func(d Delta) error {
		if d.Kind == DeltaContent {
			sawContent = true
			content.WriteString(d.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if !sawContent {
		t.Error("no content deltas received")
	}
	if res.Content == "" {
		t.Fatal("empty result content")
	}
	if res.Content != content.String() {
		t.Errorf("folded deltas %q != final result %q", content.String(), res.Content)
	}
	if res.FinishReason != FinishStop {
		t.Errorf("finish = %q, want stop", res.FinishReason)
	}
	if res.Usage.CompletionTokens == 0 {
		t.Errorf("streaming usage missing (stream_options.include_usage): %+v", res.Usage)
	}
}

// Tool calling, buffered: the model must emit a well-formed tool call, and
// a follow-up turn carrying the tool result must complete the loop.
func TestE2EToolCallRoundTrip(t *testing.T) {
	cc := e2eChat(t, "deepseek-chat")
	weather := ToolDef{
		Name:        "get_weather",
		Description: "Current weather for a city",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
	question := "What's the weather in Lisbon? Use the tool."
	ctx := e2eCtx(t, 120*time.Second)

	res, err := cc.Call(ctx, &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: question}},
		Tools:     []ToolDef{weather},
		MaxTokens: 200,
	})
	if err != nil {
		t.Fatalf("tool call: %v", err)
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("want exactly 1 tool call, got %d (finish %q, content %q)", len(res.ToolCalls), res.FinishReason, res.Content)
	}
	tc := res.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", tc.Name)
	}
	if !json.Valid([]byte(tc.Arguments)) {
		t.Errorf("arguments not valid JSON: %q", tc.Arguments)
	}

	follow := &ChatRequest{
		Messages: []Message{
			{Role: RoleUser, Content: question},
			{Role: RoleAssistant, Content: res.Content, ToolCalls: res.ToolCalls},
			{Role: RoleTool, ToolCallID: tc.ID, ToolName: tc.Name, Content: `{"temp_c":24,"condition":"sunny"}`},
		},
		Tools:     []ToolDef{weather},
		MaxTokens: 200,
	}
	res2, err := cc.Call(ctx, follow)
	if err != nil {
		t.Fatalf("follow-up: %v", err)
	}
	if res2.Content == "" {
		t.Fatal("model did not answer using the tool result")
	}
}

// Tool calling, streamed: argument fragments must assemble into a valid call.
func TestE2EToolCallStreaming(t *testing.T) {
	cc := e2eChat(t, "deepseek-chat")
	res, err := cc.CallStream(e2eCtx(t, 120*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "What's the weather in Tokyo? Use the tool."}},
		Tools:     []ToolDef{{Name: "get_weather", Description: "Current weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}},
		MaxTokens: 200,
	}, func(Delta) error { return nil })
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("want exactly 1 streamed tool call, got %d", len(res.ToolCalls))
	}
	if !json.Valid([]byte(res.ToolCalls[0].Arguments)) {
		t.Errorf("streamed arguments not valid JSON: %q", res.ToolCalls[0].Arguments)
	}
}

// deepseek-reasoner: reasoning content is captured and the answer is right.
func TestE2EReasonerThinking(t *testing.T) {
	cc := e2eChat(t, "deepseek-reasoner")
	res, err := cc.Call(e2eCtx(t, 180*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "What is 17*23? Answer with the number only."}},
		MaxTokens: 1000,
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.ReasoningContent == "" {
		t.Error("expected non-empty reasoning content from deepseek-reasoner")
	}
	if !strings.Contains(res.Content, "391") {
		t.Errorf("content = %q, want it to contain 391", res.Content)
	}
}

// Reasoner streaming: reasoning deltas flow through DeltaReasoning.
func TestE2EReasonerStreaming(t *testing.T) {
	cc := e2eChat(t, "deepseek-reasoner")
	var sawReasoning bool
	res, err := cc.CallStream(e2eCtx(t, 180*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "What is 12+12? Answer with the number only."}},
		MaxTokens: 1000,
	}, func(d Delta) error {
		if d.Kind == DeltaReasoning && d.Text != "" {
			sawReasoning = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if !sawReasoning {
		t.Error("no reasoning deltas received")
	}
	if !strings.Contains(res.Content, "24") {
		t.Errorf("content = %q, want it to contain 24", res.Content)
	}
}

// Live error taxonomy: a bad key must be a non-retryable 401 *APIError and
// the error text must never echo credentials.
func TestE2EBadKeyErrorTaxonomy(t *testing.T) {
	e2eKey(t)
	sdk := New(WithProvider("deepseek",
		WithBaseURL("https://api.deepseek.com"),
		WithAPIKey("sk-e2e-invalid-probe"),
	))
	cc, err := sdk.Chat("deepseek", "deepseek-chat")
	if err != nil {
		t.Fatal(err)
	}
	_, err = cc.Call(e2eCtx(t, 30*time.Second), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	var ae *APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %v (%T), want *APIError", err, err)
	}
	if ae.Status != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", ae.Status)
	}
	if ae.Retryable {
		t.Error("401 must not be retryable")
	}
	if strings.Contains(err.Error(), "sk-e2e-invalid-probe") {
		t.Error("error text leaked the API key")
	}
}
