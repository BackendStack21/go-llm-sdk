package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// ── Anthropic extended-thinking round-trip ───────────────────────────────

// Anthropic requires an assistant turn that ended in tool_use to replay its
// thinking block (with signature) as the first block. The canonical types
// must therefore carry the signature through results AND back into requests.
func TestAnthropicThinkingRoundTripBuffered(t *testing.T) {
	body := []byte(`{"content":[{"type":"thinking","thinking":"hmm","signature":"SIG1"},{"type":"text","text":"hi"}],"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":5}}`)
	res, err := parseAnthropicResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if res.ReasoningContent != "hmm" {
		t.Fatalf("ReasoningContent = %q, want %q", res.ReasoningContent, "hmm")
	}
	if res.ThinkingSignature != "SIG1" {
		t.Fatalf("ThinkingSignature = %q, want %q (signature must survive the round trip)", res.ThinkingSignature, "SIG1")
	}

	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, Content: "hi", ReasoningContent: "hmm", ThinkingSignature: "SIG1",
			ToolCalls: []ToolCall{{ID: "tu_1", Name: "f", Arguments: `{}`}}},
		{Role: RoleTool, ToolCallID: "tu_1", Content: "r"},
	}}
	b, err := buildAnthropicRequest(req, "m", false)
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type      string `json:"type"`
				Thinking  string `json:"thinking,omitempty"`
				Signature string `json:"signature,omitempty"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	var asst *struct {
		Role    string `json:"role"`
		Content []struct {
			Type      string `json:"type"`
			Thinking  string `json:"thinking,omitempty"`
			Signature string `json:"signature,omitempty"`
		} `json:"content"`
	}
	for i := range out.Messages {
		if out.Messages[i].Role == "assistant" {
			asst = &out.Messages[i]
		}
	}
	if asst == nil || len(asst.Content) == 0 {
		t.Fatalf("no assistant message in %s", b)
	}
	if asst.Content[0].Type != "thinking" {
		t.Fatalf("first assistant block = %s, want thinking replayed first (Anthropic contract)", asst.Content[0].Type)
	}
	if asst.Content[0].Thinking != "hmm" || asst.Content[0].Signature != "SIG1" {
		t.Fatalf("thinking block = %+v, want thinking/signature replayed", asst.Content[0])
	}
}

// Streaming: signature_delta must be captured into the result's
// ThinkingSignature, mirroring the buffered path.
func TestAnthropicStreamSignatureCapture(t *testing.T) {
	acc := newStreamAccum()
	evs := []string{
		`{"type":"message_start","message":{"usage":{"input_tokens":2}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG9"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
		`{"type":"message_stop"}`,
	}
	for _, e := range evs {
		if _, _, err := mapAnthropicStreamEvent([]byte(e), acc); err != nil {
			t.Fatal(err)
		}
	}
	res := acc.result()
	if res.ReasoningContent != "let me" {
		t.Fatalf("ReasoningContent = %q", res.ReasoningContent)
	}
	if res.ThinkingSignature != "SIG9" {
		t.Fatalf("ThinkingSignature = %q, want %q (signature_delta dropped)", res.ThinkingSignature, "SIG9")
	}
}

// ── role validation ─────────────────────────────────────────────────────

// A zero-value (or unknown) Role must be rejected loudly at the SDK
// boundary, not silently dropped (Anthropic/Gemini) or reinterpreted as a
// user message (OpenAI).
func TestUnknownRoleRejectedAtBoundary(t *testing.T) {
	var reached atomicBoolFlag
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached.Store(true)
		w.WriteHeader(500)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{
		{Role: Role("wizard"), Content: "hi"},
	}})
	var ce *ConfigError
	if !errors.As(err, &ce) || !strings.Contains(err.Error(), "role") {
		t.Fatalf("err = %v (%T), want *ConfigError naming the bad role", err, err)
	}
	if reached.Load() {
		t.Error("request reached the network despite invalid role")
	}
}

// atomicBoolFlag is a tiny helper so tests can flag handler entry safely.
type atomicBoolFlag struct {
	mu sync.Mutex
	v  bool
}

func (b *atomicBoolFlag) Store(v bool) { b.mu.Lock(); b.v = v; b.mu.Unlock() }
func (b *atomicBoolFlag) Load() bool   { b.mu.Lock(); defer b.mu.Unlock(); return b.v }

// ── Gemini fidelity ─────────────────────────────────────────────────────

// A trailing usage-only chunk (no candidates) must still update usage.
func TestGeminiUsageOnlyChunkRetained(t *testing.T) {
	acc := newStreamAccum()
	if _, _, err := mapGeminiStreamEvent([]byte(`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`), acc); err != nil {
		t.Fatal(err)
	}
	if _, _, err := mapGeminiStreamEvent([]byte(`{"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3}}`), acc); err != nil {
		t.Fatal(err)
	}
	if acc.usage.PromptTokens != 7 || acc.usage.CompletionTokens != 3 {
		t.Fatalf("usage = %+v, want {7 3 0} (usage-only chunk dropped)", acc.usage)
	}
}

// Gemini functionResponse requires the function NAME; consumers ported from
// the OpenAI format set only ToolCallID, so the name must be recovered from
// the assistant tool_call — and a truly unresolvable name must be a loud
// config error instead of a guaranteed provider 400.
func TestGeminiToolResultNameFallback(t *testing.T) {
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "weather?"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_0", Name: "get_weather", Arguments: `{"city":"SF"}`}}},
		{Role: RoleTool, ToolCallID: "call_0", Content: `{"temp":70}`},
	}}
	b, err := buildGeminiRequest(req, "gemini-2.5-pro", false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"name":"get_weather"`) {
		t.Fatalf("functionResponse name unresolved in %s", b)
	}

	bad := &ChatRequest{Messages: []Message{
		{Role: RoleTool, ToolCallID: "ghost", Content: "x"},
	}}
	if _, err := buildGeminiRequest(bad, "m", false); err == nil {
		t.Fatal("unresolvable tool result name must error, not silently 400 at the provider")
	}
}

// ── canonical finish reasons ────────────────────────────────────────────

// Provider-specific stop reasons must not leak into the canonical
// FinishReason; unknown values map to "" (unknown), like documented gaps.
func TestUnknownStopReasonsMapToCanonicalEmpty(t *testing.T) {
	if got := mapAnthropicStopReason("pause_turn"); got != "" {
		t.Errorf(`anthropic "pause_turn" = %q, want ""`, got)
	}
	if got := mapGeminiFinishReason("OTHER"); got != "" {
		t.Errorf(`gemini "OTHER" = %q, want ""`, got)
	}
	if mapAnthropicStopReason("tool_use") != FinishToolCalls {
		t.Error("anthropic tool_use regression")
	}
	if mapGeminiFinishReason("STOP") != FinishStop {
		t.Error("gemini STOP regression")
	}
}

// ── SetRequestTimeout race safety ───────────────────────────────────────

// SetRequestTimeout concurrent with RequestTimeout must be race-free (the
// buffered client is swapped atomically, not field-assigned).
//
// NOTE: this deliberately uses a deadline loop instead of time.After +
// select-default spinning — that pattern was observed to never observe its
// timer on this machine (Go 1.26.5 darwin/arm64, reproducible in a
// standalone program), which would hang the test unrelated to the SDK.
func TestSetRequestTimeoutRacesAreSafe(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1"}, nil, nil)
	cc := &ChatClient{pc: pc, model: "m", parent: &Provider{cfg: pc.cfg, sdk: New()}}
	deadline := time.Now().Add(80 * time.Millisecond)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for time.Now().Before(deadline) {
			cc.SetRequestTimeout(30 * time.Millisecond)
		}
	}()
	go func() {
		defer wg.Done()
		for time.Now().Before(deadline) {
			_ = cc.RequestTimeout()
		}
	}()
	wg.Wait()
}
