package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseGeminiResponse_CapturesThoughtSignatures(t *testing.T) {
	res, err := parseGeminiResponse([]byte(`{"candidates":[{"content":{"parts":[
		{"text":"thinking","thought":true},
		{"functionCall":{"name":"f","args":{}},"thoughtSignature":"SIG1"},
		{"functionCall":{"name":"g","args":{}}}]},"finishReason":"STOP"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 2 || res.ToolCalls[0].Signature != "SIG1" || res.ToolCalls[1].Signature != "" {
		t.Fatalf("tool calls = %+v", res.ToolCalls)
	}
	res, _ = parseGeminiResponse([]byte(`{"candidates":[{"content":{"parts":[{"text":"hi","thoughtSignature":"TXT"}]},"finishReason":"STOP"}]}`))
	if res.ThinkingSignature != "TXT" {
		t.Errorf("text signature = %q", res.ThinkingSignature)
	}
}

func TestMapGeminiStreamEvent_SignatureOnlyPart(t *testing.T) {
	acc := newStreamAccum()
	for _, c := range []string{
		`{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"END"}]},"finishReason":"STOP"}]}`,
	} {
		if _, _, err := mapGeminiStreamEvent([]byte(c), acc); err != nil {
			t.Fatal(err)
		}
	}
	if res := acc.result(); res.ThinkingSignature != "END" || res.Content != "hi" {
		t.Errorf("result = %+v", res)
	}
}

func TestBuildGeminiRequest_ReplaysThoughtSignatures(t *testing.T) {
	b, err := buildGeminiRequest(&ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_0", Name: "f", Arguments: "{}", Signature: "SIG1"}, {ID: "call_1", Name: "g", Arguments: "{}"}}},
		{Role: RoleTool, ToolCallID: "call_0", Content: "r"},
		{Role: RoleTool, ToolCallID: "call_1", Content: "r"},
		{Role: RoleAssistant, Content: "done", ThinkingSignature: "TXT"},
		{Role: RoleAssistant, ThinkingSignature: "ONLY"},
	}}, "gemini", false)
	if err != nil {
		t.Fatal(err)
	}
	var r struct {
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	calls := r.Contents[1].Parts
	if calls[0]["thoughtSignature"] != "SIG1" {
		t.Errorf("first call part = %v", calls[0])
	}
	if _, ok := calls[1]["thoughtSignature"]; ok {
		t.Errorf("second call must carry no signature: %v", calls[1])
	}
	if p := r.Contents[3].Parts[0]; p["text"] != "done" || p["thoughtSignature"] != "TXT" {
		t.Errorf("text part = %v", p)
	}
	if p := r.Contents[4].Parts[0]; p["thoughtSignature"] != "ONLY" {
		t.Errorf("signature-only turn = %v", p)
	}
}

// A function-call response is a tool_calls turn on every format; Gemini
// reports STOP for it.
func TestGeminiFinishReason_ToolCalls(t *testing.T) {
	res, err := parseGeminiResponse([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != FinishToolCalls {
		t.Errorf("buffered finish = %q, want tool_calls", res.FinishReason)
	}
	acc := newStreamAccum()
	_, _, _ = mapGeminiStreamEvent([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]}}]}`), acc)
	_, _, _ = mapGeminiStreamEvent([]byte(`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}`), acc)
	if r := acc.result(); r.FinishReason != FinishToolCalls {
		t.Errorf("stream finish = %q, want tool_calls", r.FinishReason)
	}
	// MAX_TOKENS keeps length even with a (truncated) call.
	res, _ = parseGeminiResponse([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"MAX_TOKENS"}]}`))
	if res.FinishReason != FinishLength {
		t.Errorf("max tokens finish = %q", res.FinishReason)
	}
}

func geminiStreamServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A Gemini stream that ends without any finishReason is truncated, not a
// silent partial success (invariant 3).
func TestCallStream_GeminiPrematureClose(t *testing.T) {
	cfg := ProviderConfig{ID: "gemini", Format: FormatGemini}
	srv := geminiStreamServer(t, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n")
	cfg.BaseURL = srv.URL
	c := newTestClient(t, cfg, srv)
	res, err := c.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(Delta) error { return nil })
	if !errors.Is(err, errPrematureClose) {
		t.Fatalf("err = %v, want premature close", err)
	}
	if res == nil || res.Content != "Hel" {
		t.Errorf("partial = %+v", res)
	}
}

// An unmapped (non-canonical) finish reason still completes the stream.
func TestCallStream_UnmappedFinishReasonCompletes(t *testing.T) {
	cfg := ProviderConfig{ID: "gemini", Format: FormatGemini}
	srv := geminiStreamServer(t, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}]},\"finishReason\":\"OTHER\"}]}\n\n")
	cfg.BaseURL = srv.URL
	c := newTestClient(t, cfg, srv)
	res, err := c.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(Delta) error { return nil })
	if err != nil || res.FinishReason != "" || res.Content != "x" {
		t.Fatalf("res=%+v err=%v", res, err)
	}

	oa := ProviderConfig{ID: "oa", Format: FormatOpenAI}
	srv2 := geminiStreamServer(t, "data: {\"choices\":[{\"delta\":{\"content\":\"y\"},\"finish_reason\":\"weird\"}]}\n\n")
	oa.BaseURL = srv2.URL
	c2 := newTestClient(t, oa, srv2)
	res, err = c2.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(Delta) error { return nil })
	if err != nil || res.Content != "y" {
		t.Fatalf("openai unmapped finish: res=%+v err=%v", res, err)
	}
}

func TestGeminiModelPathEscaped(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "g", Format: FormatGemini, BaseURL: "https://x"}, nil, nil)
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}
	for _, stream := range []bool{false, true} {
		_, url, err := pc.buildChatRequest(req, "../../v1/files?x=", stream)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(url, "/../") || strings.Contains(url, "?x=") {
			t.Errorf("model not escaped: %s", url)
		}
	}
	_, url, _ := pc.buildChatRequest(req, "gemini-2.5-flash", false)
	if url != "https://x/v1beta/models/gemini-2.5-flash:generateContent" {
		t.Errorf("plain model url = %s", url)
	}
}

// A prompt blocked by Gemini safety (promptFeedback.blockReason, no
// candidates) is a completed content_filter turn, not a truncated stream.
func TestGeminiPromptBlocked(t *testing.T) {
	cfg := ProviderConfig{ID: "gemini", Format: FormatGemini}
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n++
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}\n\n")
	}))
	t.Cleanup(srv.Close)
	cfg.BaseURL = srv.URL
	c := newTestClient(t, cfg, srv)
	res, err := c.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(Delta) error { return nil })
	if err != nil || res.FinishReason != FinishContentFilter || n != 1 {
		t.Fatalf("res=%+v err=%v attempts=%d", res, err, n)
	}
	r, err := parseGeminiResponse([]byte(`{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"}}`))
	if err != nil || r.FinishReason != FinishContentFilter {
		t.Fatalf("buffered: res=%+v err=%v", r, err)
	}
}
