package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ── callStream arms ──────────────────────────────────────────────────────

// A stream transport failure (dropped connection) retries and succeeds.
func TestCallStreamTransportErrorRetries(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			panic(http.ErrAbortHandler)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil || res.Content != "ok" {
		t.Fatalf("res=%q err=%v", res.Content, err)
	}
	if n != 2 {
		t.Errorf("requests = %d, want 2", n)
	}
}

// Persistent transport failure exhausts the budget with a wrapped cause.
func TestCallStreamTransportExhausted(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		panic(http.ErrAbortHandler)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	// The final attempt's raw transport error surfaces as-is.
	if err == nil || !errors.Is(err, io.EOF) {
		t.Fatalf("err = %v, want the terminal transport error", err)
	}
}

// A non-retryable status during streaming surfaces immediately.
func TestCallStreamImmediateNonRetryable(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		w.WriteHeader(401)
		fmt.Fprint(w, `{"error":{"message":"bad key"}}`)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	var ae *APIError
	if !errors.As(err, &ae) || ae.Status != 401 {
		t.Fatalf("err = %v, want 401 APIError", err)
	}
	if n != 1 {
		t.Errorf("attempts = %d, want 1", n)
	}
}

// The wall-clock deadline fires while the provider never responds: the
// deadline error surfaces (no partial output, nothing to retry).
func TestCallStreamDeadlineNoOutput(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	cc.SetRequestTimeout(120 * time.Millisecond)
	_, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err == nil {
		t.Fatal("expected deadline error")
	}
}

// Anthropic and Gemini streaming dispatch arms via the real client path.
func TestCallStreamDispatchArms(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"A\"}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n")
			fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
		})
		defer srv.Close()
		cc := newTestClient(t, ProviderConfig{ID: "a", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k"}, srv)
		res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
			return nil
		})
		if err != nil || res.Content != "A" || res.FinishReason != FinishStop {
			t.Fatalf("res=%+v err=%v", res, err)
		}
	})
	t.Run("gemini", func(t *testing.T) {
		srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"G\"}]},\"finishReason\":\"STOP\"}]}\n\n")
		})
		defer srv.Close()
		cc := newTestClient(t, ProviderConfig{ID: "g", Format: FormatGemini, BaseURL: srv.URL, APIKey: "k"}, srv)
		res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
			return nil
		})
		if err != nil || res.Content != "G" || res.FinishReason != FinishStop {
			t.Fatalf("res=%+v err=%v", res, err)
		}
	})
}

// ── small unit arms ──────────────────────────────────────────────────────

func TestReasoningEffortRejectedShape(t *testing.T) {
	if reasoningEffortRejected(errors.New("plain")) {
		t.Error("plain error must not classify")
	}
	if reasoningEffortRejected(&APIError{Status: 500, Message: "reasoning_effort"}) {
		t.Error("non-400 must not classify")
	}
	if !reasoningEffortRejected(&APIError{Status: 400, Message: "reasoning_effort unsupported"}) {
		t.Error("400 + reasoning_effort must classify")
	}
}

func TestReasoningEffortNonePatchedClearsThinking(t *testing.T) {
	withThinking := reasoningEffortNonePatched(oaRequest{ReasoningEffort: "high", Thinking: &oaThinking{Type: "enabled"}})
	if withThinking.ReasoningEffort != "none" || withThinking.Thinking != nil {
		t.Errorf("patched = %+v, want effort none and thinking cleared", withThinking)
	}
	plain := reasoningEffortNonePatched(oaRequest{ReasoningEffort: "high"})
	if plain.ReasoningEffort != "none" || plain.Thinking != nil {
		t.Errorf("patched = %+v", plain)
	}
}

func TestNewProviderClientDefaults(t *testing.T) {
	pc := newProviderClientWithLearn(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1"}, nil, nil, nil)
	if pc.buffered() == nil {
		t.Fatal("nil buffered client must default to a usable client")
	}
	if pc.learn == nil {
		t.Fatal("nil learn state must default to fresh state")
	}
}

func TestSetRequestTimeoutIgnoresNonPositive(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI}, nil, nil)
	before := pc.buffered()
	cc := &ChatClient{pc: pc, model: "m", parent: &Provider{cfg: pc.cfg, sdk: New()}}
	cc.SetRequestTimeout(0)
	cc.SetRequestTimeout(-1 * time.Second)
	if pc.buffered() != before {
		t.Error("non-positive timeout must not swap the client")
	}
}

// ── builders: remaining arms ─────────────────────────────────────────────

func TestBuildAnthropicRequestArms(t *testing.T) {
	req := &ChatRequest{
		System: []SystemBlock{{Text: "be terse", Cache: true}},
		Messages: []Message{
			{Role: RoleUser, Content: "q"},
			{Role: RoleAssistant, Content: " "}, // empty assistant → placeholder
			{Role: RoleTool, ToolCallID: "tu_1", Content: "r1"},
			{Role: RoleTool, ToolCallID: "tu_2", Content: "r2"},
		},
		Tools:          []ToolDef{{Name: "f"}}, // empty schema → default {}
		Thinking:       "enabled",
		ThinkingBudget: 4096,
		Temperature:    -1,
	}
	b, err := buildAnthropicRequest(req, "claude-x", false)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{
		"be terse", "cache_control", // system block cache marking
		`"budget_tokens":4096`, // explicit budget (maxInt arm)
		"tool_result", `"tool_use_id":"tu_1"`, `"tool_use_id":"tu_2"`,
		`"input_schema":{"type":"object"}`, // defaulted schema
	} {
		if !strings.Contains(s, want) {
			t.Errorf("anthropic request missing %q:\n%s", want, s)
		}
	}
	if !strings.Contains(s, `"temperature":0`) {
		t.Error("negative temperature must normalize to an explicit 0")
	}
}

func TestBuildGeminiRequestArms(t *testing.T) {
	cases := []struct {
		thinking string
		budget   int
		want     string
		notWant  string
	}{
		{"enabled", 0, `"thinkingConfig"`, ""},
		{"high", 0, `"thinkingConfig"`, ""},
		{"disabled", 0, `"thinkingConfig":{"thinkingBudget":0}`, ""}, // Gemini encodes "off" as budget 0
		{"", 0, "", `"thinkingConfig"`},
	}
	for _, tc := range cases {
		req := &ChatRequest{
			Messages:       []Message{{Role: RoleSystem, Content: "sys"}, {Role: RoleUser, Content: "q"}},
			Thinking:       tc.thinking,
			ThinkingBudget: tc.budget,
			Temperature:    -1,
			MaxTokens:      55,
		}
		b, err := buildGeminiRequest(req, "gemini-2.5-pro", false)
		if err != nil {
			t.Fatal(err)
		}
		s := string(b)
		if tc.want != "" && !strings.Contains(s, tc.want) {
			t.Errorf("thinking %q: missing %q in %s", tc.thinking, tc.want, s)
		}
		if tc.notWant != "" && strings.Contains(s, tc.notWant) {
			t.Errorf("thinking %q: unexpected %q in %s", tc.thinking, tc.notWant, s)
		}
		if !strings.Contains(s, "sys") || !strings.Contains(s, `"maxOutputTokens":55`) {
			t.Errorf("system fold / maxOutputTokens missing: %s", s)
		}
	}
}

// ── model listing edges ──────────────────────────────────────────────────

func TestModelsGetEdgePaths(t *testing.T) {
	// Malformed base URL → request-build ConfigError.
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "ht tp://bad", APIKey: "k"}, nil, nil)
	if _, _, err := pc.get(context.Background(), pc.base+"/models"); err == nil {
		t.Fatal("malformed URL must error")
	}
	// Unreachable endpoint → transport error.
	pc2 := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1", APIKey: "k"}, nil, nil)
	if _, _, err := pc2.get(context.Background(), pc2.base+"/models"); err == nil {
		t.Fatal("unreachable endpoint must error")
	}
}

func TestListModelsGeminiPagination(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			fmt.Fprint(w, `{"models":[{"name":"models/a"}],"nextPageToken":"PAGE2"}`)
			return
		}
		if !strings.Contains(r.URL.RawQuery, "pageToken=PAGE2") {
			t.Errorf("second page missing pageToken: %s", r.URL.RawQuery)
		}
		fmt.Fprint(w, `{"models":[{"name":"models/b"}]}`)
	})
	got, err := newListModels(srv.URL, FormatGemini)
	if err != nil || len(got) != 2 {
		t.Fatalf("models = %+v err %v", got, err)
	}
}

// ── final tail: reachable-but-uncovered arms ─────────────────────────────

func TestAnthropicThinkingBudgetArms(t *testing.T) {
	// Explicit budget at the floor: maxInt takes the >= branch.
	b, err := buildAnthropicRequest(&ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "q"}},
		Thinking: "enabled", ThinkingBudget: 1024,
	}, "claude-x", false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"budget_tokens":1024`) {
		t.Errorf("floor budget missing: %s", b)
	}
}

func TestParseAnthropicResponseArms(t *testing.T) {
	res, err := parseAnthropicResponse([]byte(`{"content":[{"type":"tool_use","id":"tu_9","name":"f","input":{"a":1}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "tu_9" || res.ToolCalls[0].Arguments != `{"a":1}` {
		t.Fatalf("tool call parse = %+v", res.ToolCalls)
	}
	if _, err := parseAnthropicResponse([]byte(`{not-json`)); err == nil {
		t.Fatal("unparseable body must error")
	}
}

func TestListModelsAnthropicErrorAndBadDates(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
	})
	if _, err := newListModels(srv.URL, FormatAnthropic); err == nil {
		t.Fatal("401 listing must error")
	}
	srv2 := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[{"id":"x","created_at":"not-a-date"}],"has_more":false}`)
	})
	got, err := newListModels(srv2.URL, FormatAnthropic)
	if err != nil || len(got) != 1 || !got[0].CreatedAt.IsZero() {
		t.Fatalf("bad date handling = %+v err %v", got, err)
	}
}

func TestListModelsOpenAIBadJSON(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{oops`)
	})
	if _, err := newListModels(srv.URL, FormatOpenAI); err == nil {
		t.Fatal("bad listing JSON must error")
	}
}

func TestListModelsGeminiError(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(403)
	})
	if _, err := newListModels(srv.URL, FormatGemini); err == nil {
		t.Fatal("403 listing must error")
	}
}

func TestListModelsCancelledContext(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1", APIKey: "k"}, nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := pc.listModels(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestModelsGetOversizedBody(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.Write(make([]byte, maxModelsResponseSize+2))
	})
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv.Client(), srv.Client())
	if _, _, err := pc.get(context.Background(), pc.base+"/models"); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("err = %v, want size-cap error", err)
	}
}

func TestBufferedCallPostBuildError(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1", APIKey: "k"}, nil, nil)
	// NUL is rejected by net/url at request-build time, before any dial.
	_, _, err := pc.post(context.Background(), pc.buffered(), "http://127.0.0.1:1/\x00chat", nil)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v (%T), want request-build ConfigError", err, err)
	}
}

func TestBufferedCallCancelledContext(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) { t.Error("must not dial") })
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cc.Call(ctx, &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestCallStreamRetryable500ThenSuccess(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			w.WriteHeader(500)
			fmt.Fprint(w, `{"error":{"message":"blip"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil || res.Content != "ok" || n != 2 {
		t.Fatalf("res=%q n=%d err=%v", res.Content, n, err)
	}
}

func TestMapOpenAIStreamEventBadJSON(t *testing.T) {
	acc := newStreamAccum()
	if _, _, err := mapOpenAIStreamEvent([]byte(`{bad`), acc); err == nil {
		t.Fatal("bad chunk JSON must error")
	}
}

func TestParseGeminiResponseBadJSON(t *testing.T) {
	if _, err := parseGeminiResponse([]byte(`{bad`)); err == nil {
		t.Fatal("bad body must error")
	}
}

func TestConsumerAbortError(t *testing.T) {
	ca := &consumerAbort{err: errors.New("inner")}
	if ca.Error() != "inner" {
		t.Errorf("consumerAbort.Error() = %q", ca.Error())
	}
}

func TestBuildOpenAIRequestThinkingArms(t *testing.T) {
	// GLM forced-thinking models upgrade disabled → enabled + low effort.
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}, Thinking: "disabled"}
	b, err := json.Marshal(buildOpenAIRequest(ProviderConfig{
		ID: "zai", Format: FormatOpenAI,
		Quirks: Quirks{ThinkingObject: true, ForceThinking: []string{"glm"}, ReasoningEffort: true},
	}, req, "glm-5.3", false, false))
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, `"thinking":{"type":"enabled"}`) || !strings.Contains(s, `"reasoning_effort":"low"`) {
		t.Errorf("forced-thinking upgrade missing: %s", s)
	}
	// Providers with no thinking support ignore the field entirely.
	b2, err := json.Marshal(buildOpenAIRequest(ProviderConfig{ID: "kimi", Format: FormatOpenAI}, req, "kimi-x", false, false))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b2), "thinking") || strings.Contains(string(b2), "reasoning_effort") {
		t.Errorf("no-quirk provider must not receive thinking fields: %s", b2)
	}
}

func TestGeminiThinkingExplicitBudget(t *testing.T) {
	b, err := buildGeminiRequest(&ChatRequest{
		Messages:       []Message{{Role: RoleUser, Content: "q"}},
		Thinking:       "enabled",
		ThinkingBudget: 2048,
	}, "gemini-2.5-pro", false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"thinkingBudget":2048`) {
		t.Errorf("explicit budget missing: %s", b)
	}
}

func TestHTTPErrorTopLevelOpenAIAndEmptyNested(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "o", Format: FormatOpenAI}, nil, nil)
	e := pc.httpError(400, []byte(`{"message":"top","code":"c1"}`))
	if e.Message != "top" || e.Code != "c1" {
		t.Fatalf("top-level = %+v", e)
	}
	e = pc.httpError(400, []byte(`{"error":{"type":"x"}}`))
	if e.Message != `{"error":{"type":"x"}}` {
		t.Fatalf("empty nested must fall back to raw body: %+v", e)
	}
}

// A build-time rejection (unknown role) inside CallStream surfaces before
// any network activity.
func TestCallStreamUnknownRoleRejected(t *testing.T) {
	var dialed bool
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		dialed = true
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: Role("wizard"), Content: "hi"}}}, func(Delta) error {
		return nil
	})
	var ce *ConfigError
	if !errors.As(err, &ce) || dialed {
		t.Fatalf("err = %v dialed=%v, want pre-dial ConfigError", err, dialed)
	}
}

func TestParseAnthropicResponseUnknownBlockIgnored(t *testing.T) {
	res, err := parseAnthropicResponse([]byte(`{"content":[{"type":"tool_result","content":"weird"},{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "ok" {
		t.Fatalf("res = %+v", res)
	}
}

func TestAnthropicThinkingBudgetBelowFloor(t *testing.T) {
	b, err := buildAnthropicRequest(&ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "q"}},
		Thinking: "enabled", ThinkingBudget: 200, // below floor → clamped to 1024
	}, "claude-x", false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"budget_tokens":1024`) {
		t.Errorf("floor clamp missing: %s", b)
	}
}

// ── third tail: small reachable arms ─────────────────────────────────────

func TestChatModelFallbackArms(t *testing.T) {
	var seenModel string
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &req)
		seenModel = req.Model
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)

	// Empty ChatClient model: falls back to ChatRequest.Model.
	fallback := &ChatClient{pc: cc.pc, model: "", parent: cc.parent}
	if _, err := fallback.Call(context.Background(), &ChatRequest{Model: "req-model", Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err != nil {
		t.Fatal(err)
	}
	if seenModel != "req-model" {
		t.Errorf("model = %q, want req-model fallback", seenModel)
	}
	// Neither set: loud config error.
	if _, err := fallback.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err == nil || !strings.Contains(err.Error(), "no model set") {
		t.Fatalf("err = %v, want no-model ConfigError", err)
	}
}

func TestCallCtxCancelDuringRetryableSleep(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	backoffUnit = 500 * time.Millisecond // first retry backoff (1s) outlives ctx
	t.Cleanup(func() { backoffUnit = time.Second })
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err := cc.Call(ctx, &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context deadline", err)
	}
}

func TestBufferedResponseOversized(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.Write(make([]byte, maxResponseSize+2))
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("err = %v, want response size-cap error", err)
	}
}

func TestCallStreamBuildErrorNULURL(t *testing.T) {
	old := backoffUnit
	backoffUnit = time.Millisecond
	t.Cleanup(func() { backoffUnit = old })
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1/\x00", APIKey: "k"}, nil, nil)
	cc := &ChatClient{pc: pc, model: "m", parent: &Provider{cfg: pc.cfg, sdk: New()}}
	_, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v (%T), want request-build ConfigError", err, err)
	}
}

func TestOpenAIParseBadJSON(t *testing.T) {
	if _, err := parseOpenAIResponse([]byte(`{oops`)); err == nil {
		t.Fatal("bad body must error")
	}
}

func TestCallStreamLearnsBufferedFromStreamRejected400(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		var req struct {
			Stream bool `json:"stream"`
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &req)
		if req.Stream {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"Streaming is not supported for this model"}}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"buffered"},"finish_reason":"stop"}]}`)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil || res.Content != "buffered" || n != 2 {
		t.Fatalf("res=%q n=%d err=%v", res.Content, n, err)
	}
}

func TestGeminiThinkingMediumArm(t *testing.T) {
	b, err := buildGeminiRequest(&ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "q"}},
		Thinking: "medium",
	}, "gemini-2.5-pro", false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"thinkingBudget":8192`) {
		t.Errorf("medium budget missing: %s", b)
	}
}

func TestListModelsGeminiBadJSON(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{oops`)
	})
	if _, err := newListModels(srv.URL, FormatGemini); err == nil {
		t.Fatal("bad listing JSON must error")
	}
}

func TestParseSSELeadingBlankLines(t *testing.T) {
	done := make(chan struct{})
	ch := make(chan sseItem, 4)
	go parseSSEStream(strings.NewReader("\n\n\ndata: x\n\n"), ch, done)
	it := <-ch
	if it.kind != sseData || string(it.data) != "x" {
		t.Fatalf("leading blank lines produced %v", it)
	}
	if next := <-ch; next.kind != sseEnd {
		t.Fatalf("expected clean end, got %v", next)
	}
}

func TestStreamAccumResultCarriesToolCalls(t *testing.T) {
	acc := newStreamAccum()
	c := acc.call(0)
	c.id, c.name = "call_0", "f"
	c.args.WriteString(`{"a":1}`)
	res := acc.result()
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "f" || res.ToolCalls[0].Arguments != `{"a":1}` {
		t.Fatalf("result tool calls = %+v", res.ToolCalls)
	}
}

// ── fourth tail: last reachable arms ─────────────────────────────────────

func TestBuildAnthropicRequestArmsTail(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleSystem, Content: "in-band system"},
			{Role: RoleUser, Content: "q"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "tu_1", Name: "f"}}}, // empty args → {}
		},
		Thinking: "enabled", // no explicit budget → 5000 default
	}
	b, err := buildAnthropicRequest(req, "claude-x", false)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"budget_tokens":5000`, "in-band system", `"input":{}`} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in %s", want, s)
		}
	}
}

func TestParseAnthropicResponseArmsTail(t *testing.T) {
	res, err := parseAnthropicResponse([]byte(`{"content":[{"type":"tool_use","id":"t","name":"f"}],"stop_reason":"tool_use"}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != "{}" {
		t.Fatalf("empty input must default to {}: %+v", res.ToolCalls)
	}
}

func TestBuildGeminiRequestArmsTail(t *testing.T) {
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_0", Name: "f"}}}, // empty content + empty args
	}}
	b, err := buildGeminiRequest(req, "gemini-2.5-pro", false)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, `"name":"f"`) || !strings.Contains(s, `"args":{}`) {
		t.Errorf("empty args must default to {}: %s", s)
	}
	if strings.Count(s, `"text":" "`) != 0 && !strings.Contains(s, `"functionCall"`) {
		t.Errorf("placeholder displaced the function call: %s", s)
	}
}

func TestMapAnthropicStreamEventBadJSON(t *testing.T) {
	acc := newStreamAccum()
	if _, _, err := mapAnthropicStreamEvent([]byte(`{bad`), acc); err == nil {
		t.Fatal("bad event JSON must error")
	}
}

func TestListModelsAnthropicBadJSON(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{oops`)
	})
	if _, err := newListModels(srv.URL, FormatAnthropic); err == nil {
		t.Fatal("bad listing JSON must error")
	}
}

func TestListModelsAnthropicPageLimitExit(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[{"id":"x"}],"has_more":true,"last_id":"x"}`)
	})
	got, err := newListModels(srv.URL, FormatAnthropic)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 10 {
		t.Errorf("models = %d, want 10 (page cap)", len(got))
	}
}

func TestProviderListModelsErrorPropagates(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(401)
	})
	sdk := New(WithProvider("openai", WithBaseURL(srv.URL), WithAPIKey("k")))
	p, _ := sdk.Provider("openai")
	if _, err := p.ListModels(context.Background()); err == nil {
		t.Fatal("listing failure must propagate")
	}
}

func TestFoldGeminiPartsEmptyArgs(t *testing.T) {
	acc := newStreamAccum()
	chunk := []byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f"}}]}}]}`)
	if _, _, err := mapGeminiStreamEvent(chunk, acc); err != nil {
		t.Fatal(err)
	}
	if len(acc.calls) != 1 || acc.calls[0].args.String() != "{}" {
		t.Fatalf("empty functionCall args must default to {}: %+v", acc.calls)
	}
}

// ── fifth tail: last reachable edges ─────────────────────────────────────

// EOF with a pending un-terminated event must still deliver it.
func TestParseSSETrailingEventWithoutBlankLine(t *testing.T) {
	done := make(chan struct{})
	ch := make(chan sseItem, 4)
	go parseSSEStream(strings.NewReader("data: trailing"), ch, done)
	it := <-ch
	if it.kind != sseData || string(it.data) != "trailing" {
		t.Fatalf("trailing event = %v (%q)", it.kind, it.data)
	}
	if next := <-ch; next.kind != sseEnd {
		t.Fatalf("expected end after trailing event, got %v", next)
	}
}

// Some OpenAI-compatible providers wrap listings in "models" instead of "data".
func TestListModelsOpenAIModelsWrapper(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"models":[{"id":"wrapped","context_length":4096}]}`)
	})
	got, err := newListModels(srv.URL, FormatOpenAI)
	if err != nil || len(got) != 1 || got[0].ID != "wrapped" || got[0].ContextWindow != 4096 {
		t.Fatalf("wrapped listing = %+v err %v", got, err)
	}
}

func TestBuildOpenAIRequestSystemAndToolRoles(t *testing.T) {
	req := &ChatRequest{
		System:   []SystemBlock{{Text: "sys"}},
		Messages: []Message{{Role: RoleSystem, Content: "band"}, {Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: "a"}, {Role: RoleTool, ToolCallID: "t1", Content: "r"}},
	}
	b, err := json.Marshal(buildOpenAIRequest(ProviderConfig{ID: "openai", Format: FormatOpenAI}, req, "m", false, false))
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, `"role":"system","content":"sys\nband"`) {
		t.Errorf("system fold missing: %s", s)
	}
	if !strings.Contains(s, `"role":"tool","content":"r","tool_call_id":"t1"`) {
		t.Errorf("tool role missing: %s", s)
	}
}

func TestListModelsRetryBreakOnCancelledContext(t *testing.T) {
	old := backoffUnit
	backoffUnit = time.Second
	t.Cleanup(func() { backoffUnit = old })
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	})
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv.Client(), srv.Client())
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err := pc.listModels(ctx)
	if err == nil || !strings.Contains(err.Error(), "list models failed") {
		t.Fatalf("err = %v, want exhausted-listing wrap", err)
	}
}

func TestPumpSSEKeepaliveWithoutWatchdog(t *testing.T) {
	before := runtime.NumGoroutine()
	err := pumpSSE(context.Background(), io.NopCloser(strings.NewReader(": ping\n\ndata: x\n\n")), 0, func([]byte) error {
		return nil
	})
	if err != nil {
		t.Fatalf("pump: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= before {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("parser goroutine did not exit")
}

// Buffered: persistent transport failure exhausts the budget with the
// wrapped "retry exhausted" error.
func TestCallTransportExhausted(t *testing.T) {
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		panic(http.ErrAbortHandler)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err == nil || !strings.Contains(err.Error(), "retry exhausted") {
		t.Fatalf("err = %v, want exhausted wrap", err)
	}
}

// Stream: a transport failure whose backoff consumes the remaining wall
// clock surfaces the deadline (or the exhausted wrap — either is honest).
func TestCallStreamDeadlineBetweenAttempts(t *testing.T) {
	old := backoffUnit
	backoffUnit = 80 * time.Millisecond
	t.Cleanup(func() { backoffUnit = old })

	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			panic(http.ErrAbortHandler) // fast transport error
		}
		select { // hang until the wall clock kills the attempt
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	// AFTER newTestClient (which pins a 1ms unit): a 80ms unit makes the
	// post-error backoff outlive the 150ms wall clock.
	backoffUnit = 80 * time.Millisecond
	cc.SetRequestTimeout(150 * time.Millisecond)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err == nil || res != nil {
		t.Fatalf("res=%v err=%v, want an error with no result", res, err)
	}
}

// Anthropic streaming error events surface as errors pre-delta (retryable).
func TestCallStreamAnthropicErrorEvent(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"boom\"}}\n\n")
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"A\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "a", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil || res.Content != "A" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	if n != 2 {
		t.Errorf("requests = %d, want 2 (error event retried pre-delta)", n)
	}
}

func TestMapGeminiStreamEventBadJSON(t *testing.T) {
	acc := newStreamAccum()
	if _, _, err := mapGeminiStreamEvent([]byte(`{bad`), acc); err == nil {
		t.Fatal("bad chunk JSON must error")
	}
}

// parseSSEStream propagates reader errors verbatim (non-EOF).
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("socket gone") }

func TestParseSSEStreamReadError(t *testing.T) {
	done := make(chan struct{})
	ch := make(chan sseItem, 2)
	go parseSSEStream(errReader{}, ch, done)
	it := <-ch
	if it.kind != sseErr || it.err == nil || strings.Contains(fmt.Sprint(it.err), "EOF") {
		t.Fatalf("item = %+v, want reader error", it)
	}
}

func TestBackoffDelayNegativeClamp(t *testing.T) {
	if d := backoffDelay(-1); d != 0 {
		t.Errorf("backoffDelay(-1) = %v, want 0", d)
	}
}

func TestParseOpenAIResponseToolCalls(t *testing.T) {
	res, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"completion_tokens_details":{"reasoning_tokens":7}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "f" || res.FinishReason != FinishToolCalls {
		t.Fatalf("res = %+v", res)
	}
	if res.Usage.ReasoningTokens != 7 {
		t.Errorf("reasoning tokens = %d", res.Usage.ReasoningTokens)
	}
}

func TestListModelsAnthropicMidPageError(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			fmt.Fprint(w, `{"data":[{"id":"a"}],"has_more":true,"last_id":"a"}`)
			return
		}
		w.WriteHeader(500)
	})
	_, err := newListModels(srv.URL, FormatAnthropic)
	var ae *APIError
	if err == nil || !errors.As(err, &ae) {
		t.Fatalf("err = %v, want the mid-page APIError", err)
	}
}

func TestListModelsGeminiMidPageError(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			fmt.Fprint(w, `{"models":[{"name":"models/a"}],"nextPageToken":"P"}`)
			return
		}
		w.WriteHeader(500)
	})
	if _, err := newListModels(srv.URL, FormatGemini); err == nil {
		t.Fatal("mid-page failure must error")
	}
}
