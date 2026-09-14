package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// openaiReqMap decodes a captured request body into a generic map.
func openaiReqMap(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("decode request: %v\nbody: %s", err, body)
	}
	return m
}

func sampleRequest() *ChatRequest {
	return &ChatRequest{
		System: []SystemBlock{{Text: "Be terse."}},
		Messages: []Message{
			{Role: RoleUser, Content: "Hi"},
		},
		Tools: []ToolDef{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		}},
	}
}

func TestBuildOpenAIRequest_Golden(t *testing.T) {
	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI, Quirks: Quirks{ReasoningEffort: true}}
	req := sampleRequest()
	req.Temperature = -1
	req.MaxTokens = 512

	oa := buildOpenAIRequest(cfg, req, "gpt-4o", false, true)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m := openaiReqMap(t, body)

	if m["model"] != "gpt-4o" {
		t.Errorf("model = %v", m["model"])
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages len = %d, want 2 (system + user)", len(msgs))
	}
	sys := msgs[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "Be terse." {
		t.Errorf("system message = %v", sys)
	}
	if m["max_tokens"].(float64) != 512 {
		t.Errorf("max_tokens = %v", m["max_tokens"])
	}
	if _, ok := m["temperature"]; !ok {
		t.Error("temperature missing, want explicit 0 (negative canonical → 0)")
	} else if m["temperature"].(float64) != 0 {
		t.Errorf("temperature = %v, want 0", m["temperature"])
	}
	if _, ok := m["stream"]; ok {
		t.Error("stream must be omitted for buffered calls")
	}
	if _, ok := m["stream_options"]; ok {
		t.Error("stream_options must be omitted for buffered calls")
	}
	tools := m["tools"].([]any)
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("tool name = %v", fn["name"])
	}
}

func TestBuildOpenAIRequest_ToolCacheNotSerialized(t *testing.T) {
	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI}
	req := &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "Hi"}},
		Tools: []ToolDef{{
			Name:       "echo",
			Parameters: json.RawMessage(`{"type":"object"}`),
			Cache:      true,
		}},
	}
	oa := buildOpenAIRequest(cfg, req, "gpt-4o", false, true)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "cache_control") {
		t.Errorf("OpenAI-format tools must not carry cache_control: %s", body)
	}
}

func TestBuildOpenAIRequest_SeparateSystemMessages(t *testing.T) {
	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI}
	req := &ChatRequest{
		System: []SystemBlock{{Text: "stable-base"}, {Text: "volatile-memory"}},
		Messages: []Message{
			{Role: RoleSystem, Content: "skill-block"},
			{Role: RoleUser, Content: "Hi"},
		},
	}
	oa := buildOpenAIRequest(cfg, req, "gpt-4o", false, false)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m := openaiReqMap(t, body)
	msgs := m["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("messages len = %d, want 4 (3 system + user)", len(msgs))
	}
	want := []string{"stable-base", "volatile-memory", "skill-block", "Hi"}
	for i, w := range want {
		got := msgs[i].(map[string]any)
		role := "system"
		if i == 3 {
			role = "user"
		}
		if got["role"] != role || got["content"] != w {
			t.Errorf("messages[%d] = %v, want role=%s content=%q", i, got, role, w)
		}
	}
}

func TestBuildOpenAIRequest_TemperatureForbiddenModels(t *testing.T) {
	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI, Quirks: Quirks{ReasoningEffort: true}}
	req := sampleRequest()
	req.Temperature = 0.3

	for _, model := range []string{"gpt-5.6", "o3-mini", "kimi-for-coding-v1", "k3-max"} {
		oa := buildOpenAIRequest(cfg, req, model, false, true)
		body, _ := json.Marshal(oa)
		m := openaiReqMap(t, body)
		if _, ok := m["temperature"]; ok {
			t.Errorf("%s: temperature present, must be omitted", model)
		}
	}
	// Control: allowed model keeps the field.
	oa := buildOpenAIRequest(cfg, req, "gpt-4o", false, true)
	body, _ := json.Marshal(oa)
	if _, ok := openaiReqMap(t, body)["temperature"]; !ok {
		t.Error("gpt-4o: temperature missing for allowed model")
	}
}

func TestBuildOpenAIRequest_ThinkingVariants(t *testing.T) {
	base := sampleRequest()

	// OpenAI: reasoning_effort mapping, never a thinking object.
	openai := ProviderConfig{ID: "openai", Format: FormatOpenAI, Quirks: Quirks{ReasoningEffort: true}}
	cases := []struct {
		thinking string
		wantEff  string
	}{
		{"enabled", "medium"}, {"low", "low"}, {"medium", "medium"}, {"high", "high"}, {"max", "high"}, {"disabled", "none"}, {"", ""},
	}
	for _, c := range cases {
		r := *base
		r.Thinking = c.thinking
		oa := buildOpenAIRequest(openai, &r, "gpt-5.6", false, true)
		if oa.ReasoningEffort != c.wantEff {
			t.Errorf("openai thinking=%q → effort %q, want %q", c.thinking, oa.ReasoningEffort, c.wantEff)
		}
		if oa.Thinking != nil {
			t.Errorf("openai must never send thinking object, got %+v", oa.Thinking)
		}
	}
	// Non-5.6 OpenAI: disabled still omits the field (provider default).
	r := *base
	r.Thinking = "disabled"
	oa := buildOpenAIRequest(openai, &r, "gpt-4o", false, true)
	if oa.ReasoningEffort != "" {
		t.Errorf("gpt-4o disabled → effort %q, want omit", oa.ReasoningEffort)
	}

	// DeepSeek: thinking object, no effort.
	deepseek := ProviderConfig{ID: "deepseek", Format: FormatOpenAI, Quirks: Quirks{ThinkingObject: true}}
	r = *base
	r.Thinking = "enabled"
	oa = buildOpenAIRequest(deepseek, &r, "deepseek-v4-pro", false, true)
	if oa.Thinking == nil || oa.Thinking.Type != "enabled" {
		t.Errorf("deepseek thinking = %+v, want {type:enabled}", oa.Thinking)
	}
	if oa.ReasoningEffort != "" {
		t.Errorf("deepseek must not send reasoning_effort, got %q", oa.ReasoningEffort)
	}

	// GLM forced-thinking: disabled on glm-5.3* flips to enabled+low.
	zai := ProviderConfig{ID: "zai", Format: FormatOpenAI, Quirks: Quirks{ThinkingObject: true, ReasoningEffort: true, ForceThinking: []string{"glm-5.3"}}}
	r = *base
	r.Thinking = "disabled"
	oa = buildOpenAIRequest(zai, &r, "glm-5.3", false, true)
	if oa.Thinking == nil || oa.Thinking.Type != "enabled" {
		t.Errorf("glm-5.3 disabled → thinking %+v, want {type:enabled}", oa.Thinking)
	}
	if oa.ReasoningEffort != "low" {
		t.Errorf("glm-5.3 disabled → effort %q, want low", oa.ReasoningEffort)
	}
	// Same request on a non-forced GLM keeps disabled.
	oa = buildOpenAIRequest(zai, &r, "glm-4.6", false, true)
	if oa.Thinking == nil || oa.Thinking.Type != "disabled" {
		t.Errorf("glm-4.6 disabled → thinking %+v, want {type:disabled}", oa.Thinking)
	}
	// Levels on GLM: enabled + effort.
	r = *base
	r.Thinking = "high"
	oa = buildOpenAIRequest(zai, &r, "glm-5.3", false, true)
	if oa.Thinking == nil || oa.ReasoningEffort != "high" {
		t.Errorf("glm high → %+v effort %q, want enabled+high", oa.Thinking, oa.ReasoningEffort)
	}
}

// GLM has no "medium" effort level and accepts "max". Mapping applies only
// when both ThinkingObject and ReasoningEffort are set.
func TestBuildOpenAIRequest_GLMThinkingMediumAndMax(t *testing.T) {
	zai := ProviderConfig{ID: "zai", Format: FormatOpenAI, Quirks: Quirks{ThinkingObject: true, ReasoningEffort: true, ForceThinking: []string{"glm-5.3"}}}
	base := sampleRequest()

	r := *base
	r.Thinking = "medium"
	oa := buildOpenAIRequest(zai, &r, "glm-5.3", false, true)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m := openaiReqMap(t, body)
	if m["reasoning_effort"] != "high" {
		t.Errorf("thinking medium → reasoning_effort %v, want high (GLM has no medium)", m["reasoning_effort"])
	}
	th, _ := m["thinking"].(map[string]any)
	if th == nil || th["type"] != "enabled" {
		t.Errorf("thinking medium → thinking %v, want {type:enabled}", m["thinking"])
	}

	r = *base
	r.Thinking = "max"
	oa = buildOpenAIRequest(zai, &r, "glm-5.3", false, true)
	body, err = json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m = openaiReqMap(t, body)
	if m["reasoning_effort"] != "max" {
		t.Errorf("thinking max → reasoning_effort %v, want max", m["reasoning_effort"])
	}
	th, _ = m["thinking"].(map[string]any)
	if th == nil || th["type"] != "enabled" {
		t.Errorf("thinking max → thinking %v, want {type:enabled}", m["thinking"])
	}
}

func TestBuildOpenAIRequest_ReasoningContentReplay(t *testing.T) {
	cfg := ProviderConfig{ID: "deepseek", Format: FormatOpenAI, Quirks: Quirks{ThinkingObject: true}}
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, Content: "a", ReasoningContent: "internal thoughts"},
		{Role: RoleUser, Content: "again"},
	}}
	oa := buildOpenAIRequest(cfg, req, "deepseek-reasoner", false, false)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m := openaiReqMap(t, body)
	msgs := m["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want 3", len(msgs))
	}
	asst := msgs[1].(map[string]any)
	if asst["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", asst["role"])
	}
	if asst["reasoning_content"] != "internal thoughts" {
		t.Errorf("reasoning_content = %v, want %q (DeepSeek tool loops require echo)", asst["reasoning_content"], "internal thoughts")
	}
	if asst["content"] != "a" {
		t.Errorf("content = %v, want a", asst["content"])
	}
}

// DeepSeek thinking mode with tools: the reasoning_content key must survive on
// every replayed assistant turn — including turns where the provider returned
// no reasoning of its own (documented elision; see e2e_test.go). DeepSeek's
// documented contract: "for requests carrying the tools parameter, the
// reasoning_content must be fully passed back to the API in all subsequent
// requests — even for turns where the model did not perform a tool call. If
// your code does not correctly pass back reasoning_content, the API will
// return a 400 error." An omitted key is a hard 400 on every later request of
// the loop, not just the offending turn.
func TestBuildOpenAIRequest_EchoesEmptyReasoningWithTools(t *testing.T) {
	cfg := ProviderConfig{
		ID:     "deepseek",
		Format: FormatOpenAI,
		Quirks: Quirks{ThinkingObject: true, EchoReasoningWithTools: true},
	}
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleUser, Content: "weather?"},
			// Tool-call turn for which the provider elided reasoning.
			{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Berlin"}`},
			}},
			{Role: RoleTool, ToolCallID: "call_1", ToolName: "get_weather", Content: `{"temp":20}`},
		},
		Tools:    []ToolDef{{Name: "get_weather"}},
		Thinking: "enabled",
	}
	oa := buildOpenAIRequest(cfg, req, "deepseek-chat", false, false)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m := openaiReqMap(t, body)
	asst := m["messages"].([]any)[1].(map[string]any)
	v, present := asst["reasoning_content"]
	if !present {
		t.Fatal("reasoning_content key absent on a tool-call assistant turn: DeepSeek returns 400 for every later request in the loop")
	}
	if v != "" {
		t.Errorf("reasoning_content = %v, want empty string", v)
	}
}

// Without tools the key must stay absent when there is nothing to echo:
// DeepSeek ignores the field there, and OpenAI-compatible providers that do
// not know it must never see it.
func TestBuildOpenAIRequest_NoReasoningKeyWithoutTools(t *testing.T) {
	cfg := ProviderConfig{
		ID:     "deepseek",
		Format: FormatOpenAI,
		Quirks: Quirks{ThinkingObject: true, EchoReasoningWithTools: true},
	}
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleUser, Content: "q"},
			{Role: RoleAssistant, Content: "a"},
		},
		Thinking: "enabled",
	}
	oa := buildOpenAIRequest(cfg, req, "deepseek-chat", false, false)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m := openaiReqMap(t, body)
	asst := m["messages"].([]any)[1].(map[string]any)
	if _, present := asst["reasoning_content"]; present {
		t.Errorf("reasoning_content must not be sent without tools: %v", asst["reasoning_content"])
	}
}

// The echo follows the tools, not the thinking knob: a tool-bearing DeepSeek
// request echoes the key whether thinking is enabled, disabled or unset, and
// on every assistant turn — not only the one that triggered the fix. Pinning
// it here makes the wider blast radius a decision rather than a side effect.
func TestBuildOpenAIRequest_EchoFollowsToolsNotThinking(t *testing.T) {
	cfg := ProviderConfig{
		ID:     "deepseek",
		Format: FormatOpenAI,
		Quirks: Quirks{ThinkingObject: true, EchoReasoningWithTools: true},
	}
	// Two assistant turns: one with reasoning, one elided (empty).
	msgs := []Message{
		{Role: RoleUser, Content: "weather?"},
		{Role: RoleAssistant, Content: "Checking.", ReasoningContent: "need the forecast",
			ToolCalls: []ToolCall{{ID: "c1", Name: "f", Arguments: "{}"}}},
		{Role: RoleTool, ToolCallID: "c1", ToolName: "f", Content: "{}"},
		{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{{ID: "c2", Name: "f", Arguments: "{}"}}},
		{Role: RoleTool, ToolCallID: "c2", ToolName: "f", Content: "{}"},
	}
	for _, thinking := range []string{"", "disabled", "enabled", "high"} {
		req := &ChatRequest{Messages: msgs, Tools: []ToolDef{{Name: "f"}}, Thinking: thinking}
		oa := buildOpenAIRequest(cfg, req, "deepseek-chat", false, false)
		body, err := json.Marshal(oa)
		if err != nil {
			t.Fatal(err)
		}
		m := openaiReqMap(t, body)
		got := m["messages"].([]any)
		for _, idx := range []int{1, 3} {
			asst := got[idx].(map[string]any)
			if _, present := asst["reasoning_content"]; !present {
				t.Errorf("thinking %q: assistant msg %d is missing reasoning_content (must follow tools, not the thinking setting)", thinking, idx)
			}
		}
	}
}

// Registry -> wire composition: the built-in deepseek entry must actually put
// the empty echo on the outbound body. Every other wire test hand-builds its own
// ProviderConfig, so without this one the registry flag and the serializer are
// only ever tested apart, and a flag that stops reaching the wire ships silently.
func TestChatClient_DeepSeekRegistryEchoReachesWire(t *testing.T) {
	var bodies [][]byte
	srv := captureJSON(t, &bodies)
	defer srv.Close()
	t.Setenv("DEEPSEEK_API_KEY", "k")
	sdk := New(FromEnv(), WithProvider("deepseek", WithBaseURL(srv.URL)))
	cc, err := sdk.Chat("deepseek", "deepseek-chat")
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, err := cc.Call(context.Background(), emptyEchoProbeRequest()); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(bodies))
	}
	if !strings.Contains(string(bodies[0]), `"reasoning_content":""`) {
		t.Errorf("registry deepseek must echo an empty reasoning_content; body: %s", bodies[0])
	}
}

// WithQuirks REPLACES the quirks struct rather than merging into it, so
// re-registering the built-in deepseek id with explicit quirks silently drops
// the echo flag. Pinned observably (and documented in README) so the behaviour
// is a decision: making WithQuirks merge must be a deliberate change that fails
// here first.
func TestWithProvider_QuirksReplaceDropsEcho(t *testing.T) {
	var bodies [][]byte
	srv := captureJSON(t, &bodies)
	defer srv.Close()
	t.Setenv("DEEPSEEK_API_KEY", "k")
	sdk := New(FromEnv(),
		WithProvider("deepseek", WithBaseURL(srv.URL), WithQuirks(Quirks{ThinkingObject: true})))
	cc, err := sdk.Chat("deepseek", "deepseek-chat")
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, err := cc.Call(context.Background(), emptyEchoProbeRequest()); err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(bodies))
	}
	if strings.Contains(string(bodies[0]), `"reasoning_content"`) {
		t.Errorf("WithQuirks(Quirks{ThinkingObject}) must lose the echo (replaces, not merges); body: %s", bodies[0])
	}
}

// captureJSON is an httptest server that records request bodies and answers a
// minimal valid chat completion.
func captureJSON(t *testing.T, bodies *[][]byte) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		*bodies = append(*bodies, b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`)
	}))

	return srv
}

// emptyEchoProbeRequest is a minimal tool-bearing request whose assistant turn
// carries no reasoning: the state the echo exists for.
func emptyEchoProbeRequest() *ChatRequest {
	return &ChatRequest{
		Tools: []ToolDef{{Name: "f"}},
		Messages: []Message{
			{Role: RoleUser, Content: "q"},
			{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{{ID: "c1", Name: "f", Arguments: "{}"}}},
			{Role: RoleTool, ToolCallID: "c1", ToolName: "f", Content: "{}"},
		},
	}
}

// A provider without the echo quirk keeps today's wire shape exactly: no key
// when there is no reasoning to replay.
func TestBuildOpenAIRequest_NoEchoWithoutQuirk(t *testing.T) {
	cfg := ProviderConfig{ID: "kimi", Format: FormatOpenAI}
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleUser, Content: "weather?"},
			{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Berlin"}`},
			}},
			{Role: RoleTool, ToolCallID: "call_1", ToolName: "get_weather", Content: `{"temp":20}`},
		},
		Tools: []ToolDef{{Name: "get_weather"}},
	}
	oa := buildOpenAIRequest(cfg, req, "kimi-k2", false, false)
	body, err := json.Marshal(oa)
	if err != nil {
		t.Fatal(err)
	}
	m := openaiReqMap(t, body)
	asst := m["messages"].([]any)[1].(map[string]any)
	if _, present := asst["reasoning_content"]; present {
		t.Errorf("unexpected reasoning_content for a non-echo provider: %v", asst["reasoning_content"])
	}
}

func TestBuildOpenAIRequest_ToolMessages(t *testing.T) {
	cfg := ProviderConfig{ID: "kimi", Format: FormatOpenAI}
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "weather?"},
		{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Berlin"}`}}},
		{Role: RoleTool, ToolCallID: "call_1", ToolName: "get_weather", Content: `{"temp":20}`},
		{Role: RoleAssistant, Content: "20°C"},
	}}
	oa := buildOpenAIRequest(cfg, req, "kimi-k2", false, true)
	if len(oa.Messages) != 4 {
		t.Fatalf("messages = %d, want 4", len(oa.Messages))
	}
	tool := oa.Messages[2]
	if tool.Role != "tool" || tool.ToolCallID != "call_1" {
		t.Fatalf("tool message = %+v", tool)
	}
	if tool.Content == nil || *tool.Content != `{"temp":20}` {
		t.Fatalf("tool content = %v", tool.Content)
	}
	asst := oa.Messages[1]
	if len(asst.ToolCalls) != 1 || asst.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("assistant tool_calls = %+v", asst.ToolCalls)
	}
}

func TestParseOpenAIResponse_ToolCallsAndUsage(t *testing.T) {
	body := []byte(`{
		"choices": [{
			"message": {
				"role": "assistant",
				"content": "Checking.",
				"reasoning_content": "hmm",
				"tool_calls": [{
					"id": "call_9", "type": "function",
					"function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 10, "completion_tokens": 5,
			"completion_tokens_details": {"reasoning_tokens": 3}
		}
	}`)
	res, err := parseOpenAIResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "Checking." || res.ReasoningContent != "hmm" {
		t.Errorf("content fields: %+v", res)
	}
	if res.FinishReason != FinishToolCalls {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "call_9" || res.ToolCalls[0].Arguments != `{"city":"Oslo"}` {
		t.Errorf("tool calls: %+v", res.ToolCalls)
	}
	if res.Usage.PromptTokens != 10 || res.Usage.CompletionTokens != 5 || res.Usage.ReasoningTokens != 3 {
		t.Errorf("usage: %+v", res.Usage)
	}
}

func TestParseOpenAIResponse_NoChoices(t *testing.T) {
	if _, err := parseOpenAIResponse([]byte(`{"choices":[]}`)); err == nil {
		t.Fatal("expected error for empty choices")
	}
}

func TestMapOpenAIStreamEvent_Accumulation(t *testing.T) {
	acc := newStreamAccum()
	chunks := []string{
		`{"choices":[{"delta":{"reasoning_content":"think "}}]}`,
		`{"choices":[{"delta":{"reasoning_content":"hard"}}]}`,
		`{"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"content":"lo"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{\"a\""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":7,"completion_tokens":4,"completion_tokens_details":{"reasoning_tokens":2}}}`,
	}
	var all []Delta
	for _, c := range chunks {
		ds, _, err := mapOpenAIStreamEvent([]byte(c), acc)
		if err != nil {
			t.Fatalf("chunk %s: %v", c, err)
		}
		all = append(all, ds...)
	}
	res := acc.result()
	if res.ReasoningContent != "think hard" {
		t.Errorf("reasoning = %q", res.ReasoningContent)
	}
	if res.Content != "Hello" {
		t.Errorf("content = %q", res.Content)
	}
	if res.FinishReason != FinishToolCalls {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "f" || res.ToolCalls[0].Arguments != `{"a":1}` {
		t.Errorf("tool calls: %+v", res.ToolCalls)
	}
	if res.Usage.PromptTokens != 7 || res.Usage.ReasoningTokens != 2 {
		t.Errorf("usage: %+v", res.Usage)
	}
	// First tool delta carries id+name.
	var toolFirst Delta
	for _, d := range all {
		if d.Kind == DeltaToolArgs {
			toolFirst = d
			break
		}
	}
	if toolFirst.ToolID != "call_1" || toolFirst.ToolName != "f" {
		t.Errorf("first tool delta = %+v", toolFirst)
	}
}

func TestListModelsOpenAI_ContextFields(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Write([]byte(`{"data":[
			{"id":"m1","context_length":131072},
			{"id":"m2","max_context":65536},
			{"id":"m3","max_input_tokens":32768},
			{"id":"m4"}
		]}`))
	}))
	defer srv.Close()

	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"},
		srv.Client(), srv.Client())
	models, err := listModelsOpenAI(context.Background(), pc)
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer k" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	wantCtx := map[string]int{"m1": 131072, "m2": 65536, "m3": 32768, "m4": 0}
	if len(models) != 4 {
		t.Fatalf("models = %d, want 4", len(models))
	}
	for _, m := range models {
		if m.ContextWindow != wantCtx[m.ID] {
			t.Errorf("%s ContextWindow = %d, want %d", m.ID, m.ContextWindow, wantCtx[m.ID])
		}
	}
}

func TestListModelsOpenAI_ModelsWrapper(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/models") {
			t.Errorf("path = %s, want .../models", r.URL.Path)
		}
		w.Write([]byte(`{"models":[{"id":"alt"}]}`))
	}))
	defer srv.Close()

	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"},
		srv.Client(), srv.Client())
	models, err := listModelsOpenAI(context.Background(), pc)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "alt" {
		t.Fatalf("models = %+v", models)
	}
}

// Models in the o-series/gpt-5 family reject max_tokens in favor of
// max_completion_tokens; everyone else keeps the classic parameter.
func TestBuildOpenAIRequestTokenParamRouting(t *testing.T) {
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}, MaxTokens: 100}
	for _, model := range []string{"o3", "o4-mini", "gpt-5-mini"} {
		b, err := json.Marshal(buildOpenAIRequest(ProviderConfig{ID: "openai", Format: FormatOpenAI}, req, model, false, false))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `"max_completion_tokens":100`) {
			t.Errorf("%s: body missing max_completion_tokens: %s", model, b)
		}
		if strings.Contains(string(b), `"max_tokens"`) {
			t.Errorf("%s: body must not send max_tokens: %s", model, b)
		}
	}
	for _, model := range []string{"gpt-4o", "deepseek-v4"} {
		b, err := json.Marshal(buildOpenAIRequest(ProviderConfig{ID: "openai", Format: FormatOpenAI}, req, model, false, false))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `"max_tokens":100`) {
			t.Errorf("%s: body missing max_tokens: %s", model, b)
		}
		if strings.Contains(string(b), "max_completion_tokens") {
			t.Errorf("%s: body must not send max_completion_tokens: %s", model, b)
		}
	}
}
