package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestChatCompletionsRejectsReasoningWithTools(t *testing.T) {
	for _, m := range []string{"gpt-5.6", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "GPT-5.6-Luna", "gpt-5.7-preview", "gpt-6-astra"} {
		if !chatCompletionsRejectsReasoningWithTools(m) {
			t.Errorf("%s must use Responses for tools+reasoning", m)
		}
	}
	for _, m := range []string{"gpt-5", "gpt-5-mini", "gpt-5.5", "gpt-5.4", "gpt-4o", "o3-mini"} {
		if chatCompletionsRejectsReasoningWithTools(m) {
			t.Errorf("%s must not proactively force Responses", m)
		}
	}
}

func TestUseResponsesAPI(t *testing.T) {
	tools := []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}}
	learn := &learnOnce{}

	if useResponsesAPI(learn, FormatOpenAI, "gpt-5.6-luna", &ChatRequest{Tools: tools, Thinking: "medium"}) != true {
		t.Error("gpt-5.6-luna + tools + medium must use Responses")
	}
	if useResponsesAPI(learn, FormatOpenAI, "gpt-5.6-luna", &ChatRequest{Tools: tools}) != true {
		t.Error("gpt-5.6-luna + tools + empty thinking still defaults to medium; must use Responses")
	}
	if useResponsesAPI(learn, FormatOpenAI, "gpt-5.6-luna", &ChatRequest{Tools: tools, Thinking: "disabled"}) != false {
		t.Error("disabled thinking stays on chat completions")
	}
	if useResponsesAPI(learn, FormatOpenAI, "gpt-5.6-luna", &ChatRequest{Thinking: "medium"}) != false {
		t.Error("no tools → chat completions")
	}
	if useResponsesAPI(learn, FormatAnthropic, "gpt-5.6-luna", &ChatRequest{Tools: tools, Thinking: "medium"}) != false {
		t.Error("anthropic format must not route to Responses")
	}
	if useResponsesAPI(learn, FormatOpenAI, "gpt-4o", &ChatRequest{Tools: tools, Thinking: "medium"}) != false {
		t.Error("gpt-4o must not proactively use Responses")
	}
	if useResponsesAPI(learn, FormatOpenAI, "gpt-5.5", &ChatRequest{Tools: tools, Thinking: "high"}) != true {
		t.Error("gpt-5.5 + explicit thinking + tools uses Responses")
	}
	if useResponsesAPI(learn, FormatOpenAI, "gpt-5.5", &ChatRequest{Tools: tools}) != false {
		t.Error("gpt-5.5 + empty thinking stays on chat completions")
	}

	learn.forceResponses.Store(true)
	if useResponsesAPI(learn, FormatOpenAI, "gpt-4o", &ChatRequest{Tools: tools, Thinking: "high"}) != true {
		t.Error("learned forceResponses must win for any OpenAI-format model with tools")
	}
	learn = &learnOnce{}
	learn.forceNoneEffort.Store(true)
	if useResponsesAPI(learn, FormatOpenAI, "gpt-5.6-luna", &ChatRequest{Tools: tools, Thinking: "medium"}) != false {
		t.Error("learned forceNoneEffort must keep chat completions")
	}
}

func TestBuildResponsesRequest_Shape(t *testing.T) {
	req := &ChatRequest{
		System:    []SystemBlock{{Text: "Be terse."}},
		Messages:  []Message{{Role: RoleUser, Content: "hi"}},
		Tools:     []ToolDef{{Name: "echo", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking:  "medium",
		MaxTokens: 2048,
	}
	body, err := json.Marshal(buildResponsesRequest(req, "gpt-5.6-luna", false))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatal(err)
	}
	if m["model"] != "gpt-5.6-luna" {
		t.Errorf("model = %v", m["model"])
	}
	if m["instructions"] != "Be terse." {
		t.Errorf("instructions = %v", m["instructions"])
	}
	if m["max_output_tokens"] != float64(2048) {
		t.Errorf("max_output_tokens = %v", m["max_output_tokens"])
	}
	if store, ok := m["store"].(bool); !ok || store {
		t.Errorf("store = %v, want false", m["store"])
	}
	inc, _ := m["include"].([]any)
	if len(inc) != 1 || inc[0] != "reasoning.encrypted_content" {
		t.Errorf("include = %v", m["include"])
	}
	rsn, _ := m["reasoning"].(map[string]any)
	if rsn["effort"] != "medium" || rsn["summary"] != "auto" {
		t.Errorf("reasoning = %v", rsn)
	}
	tools, _ := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", m["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "echo" {
		t.Errorf("tool = %v", tool)
	}
	if _, ok := tool["function"]; ok {
		t.Error("Responses tools must be flat, not nested under function")
	}
	if _, ok := m["reasoning_effort"]; ok {
		t.Error("Responses must not send reasoning_effort")
	}
}

func TestBuildResponsesInput_ToolLoopReplay(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleUser, Content: "weather?"},
			{
				Role:              RoleAssistant,
				ReasoningContent:  "need a tool",
				ThinkingSignature: "enc-1",
				ToolCalls:         []ToolCall{{ID: "call_1", Name: "get_weather", Arguments: `{"q":"sf"}`}},
			},
			{Role: RoleTool, ToolCallID: "call_1", Content: "72F"},
		},
	}
	_, input := buildResponsesInput(req)
	body, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	var items []map[string]any
	if err := json.Unmarshal(body, &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 4 {
		t.Fatalf("items = %d (%s), want 4", len(items), body)
	}
	if items[0]["role"] != "user" {
		t.Errorf("item0 = %v", items[0])
	}
	if items[1]["type"] != "reasoning" || items[1]["encrypted_content"] != "enc-1" {
		t.Errorf("item1 = %v", items[1])
	}
	if items[2]["type"] != "function_call" || items[2]["call_id"] != "call_1" {
		t.Errorf("item2 = %v", items[2])
	}
	if items[3]["type"] != "function_call_output" || items[3]["output"] != "72F" {
		t.Errorf("item3 = %v", items[3])
	}
}

func TestParseResponsesAPI_TextToolsReasoning(t *testing.T) {
	raw := `{
		"status":"completed",
		"output":[
			{"type":"reasoning","encrypted_content":"enc-9","summary":[{"type":"summary_text","text":"plan"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]},
			{"type":"function_call","call_id":"c1","name":"echo","arguments":"{}"}
		],
		"usage":{"input_tokens":20,"output_tokens":8,"input_tokens_details":{"cached_tokens":4},"output_tokens_details":{"reasoning_tokens":3}}
	}`
	res, err := parseResponsesAPI([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "hi" || res.ReasoningContent != "plan" || res.ThinkingSignature != "enc-9" {
		t.Errorf("res = %+v", res)
	}
	if res.FinishReason != FinishToolCalls {
		t.Errorf("finish = %q, want tool_calls", res.FinishReason)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "c1" || res.ToolCalls[0].Name != "echo" {
		t.Errorf("tools = %+v", res.ToolCalls)
	}
	if res.Usage.PromptTokens != 16 || res.Usage.CacheReadTokens != 4 || res.Usage.ReasoningTokens != 3 {
		t.Errorf("usage = %+v", res.Usage)
	}
}

func TestParseResponsesAPI_IncompleteLength(t *testing.T) {
	raw := `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","content":[{"type":"output_text","text":"cut"}]}]}`
	res, err := parseResponsesAPI([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != FinishLength || res.Content != "cut" {
		t.Errorf("res = %+v", res)
	}
}

func TestParseResponsesAPI_ProviderError(t *testing.T) {
	_, err := parseResponsesAPI([]byte(`{"error":{"message":"nope"}}`))
	if err == nil || !strings.Contains(err.Error(), "nope") {
		t.Fatalf("err = %v", err)
	}
}

func TestMapResponsesStreamEvent(t *testing.T) {
	acc := newStreamAccum()
	events := []string{
		`{"type":"response.reasoning_summary_text.delta","delta":"think "}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"hard"}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"echo"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{}"}`,
		`{"type":"response.output_item.done","item":{"type":"reasoning","encrypted_content":"enc"}}`,
		`{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":2,"output_tokens_details":{"reasoning_tokens":1}}}}`,
	}
	var kinds []DeltaKind
	var lastDone bool
	for _, e := range events {
		ds, done, err := mapResponsesStreamEvent([]byte(e), acc)
		if err != nil {
			t.Fatalf("event %s: %v", e, err)
		}
		for _, d := range ds {
			kinds = append(kinds, d.Kind)
		}
		lastDone = done
	}
	if !lastDone {
		t.Fatal("completed event must signal done")
	}
	res := acc.result()
	if res.ReasoningContent != "think hard" {
		t.Errorf("reasoning = %q", res.ReasoningContent)
	}
	if res.ThinkingSignature != "enc" {
		t.Errorf("signature = %q", res.ThinkingSignature)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "c1" || res.ToolCalls[0].Arguments != "{}" {
		t.Errorf("tools = %+v", res.ToolCalls)
	}
	if res.FinishReason != FinishToolCalls {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if res.Usage.ReasoningTokens != 1 {
		t.Errorf("usage = %+v", res.Usage)
	}
	if len(kinds) < 3 || kinds[0] != DeltaReasoning || kinds[len(kinds)-1] != DeltaToolArgs {
		t.Errorf("kinds = %v", kinds)
	}
}

func TestCall_GPT56ToolsUseResponses(t *testing.T) {
	var path string
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		body, _ = io.ReadAll(r.Body)
		fmt.Fprint(w, `{"status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"plan"}],"encrypted_content":"enc"},{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":3,"output_tokens":2,"output_tokens_details":{"reasoning_tokens":4}}}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{ReasoningEffort: true}}, srv)
	cc.model = "gpt-5.6-luna"
	res, err := cc.Call(context.Background(), &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools:    []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking: "medium",
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.HasSuffix(path, "/responses") {
		t.Fatalf("path = %q, want .../responses", path)
	}
	if !strings.Contains(string(body), `"effort":"medium"`) || strings.Contains(string(body), "reasoning_effort") {
		t.Errorf("body = %s", body)
	}
	if res.Content != "ok" || res.ReasoningContent != "plan" || res.ThinkingSignature != "enc" {
		t.Errorf("res = %+v", res)
	}
	if res.Usage.ReasoningTokens != 4 {
		t.Errorf("usage = %+v", res.Usage)
	}
}

func TestCall_GPT56DisabledStaysChatCompletions(t *testing.T) {
	var path, payload string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		payload = string(b)
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{ReasoningEffort: true}}, srv)
	cc.model = "gpt-5.6-luna"
	_, err := cc.Call(context.Background(), &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools:    []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking: "disabled",
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.HasSuffix(path, "/chat/completions") {
		t.Fatalf("path = %q, want chat/completions", path)
	}
	if !strings.Contains(payload, `"reasoning_effort":"none"`) {
		t.Errorf("disabled gpt-5.6 must pin effort none, body=%s", payload)
	}
}

func TestCall_LearnOnceResponsesAPI(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		paths = append(paths, r.URL.Path+" "+string(b[:min(len(b), 40)]))
		n := len(paths)
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-5.4 in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'."}}`)
			return
		}
		fmt.Fprint(w, `{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{ReasoningEffort: true}}, srv)
	cc.model = "gpt-4o"
	res, err := cc.Call(context.Background(), &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools:    []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking: "high",
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.Content != "ok" {
		t.Errorf("content = %q", res.Content)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 2 {
		t.Fatalf("requests = %d (%v), want 2", len(paths), paths)
	}
	if !strings.Contains(paths[0], "/chat/completions") {
		t.Errorf("first path = %s", paths[0])
	}
	if !strings.Contains(paths[1], "/responses") {
		t.Errorf("second path = %s, want /responses (not effort none)", paths[1])
	}
}

func TestCallStream_GPT56ToolsUseResponses(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"plan\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":2}}}}\n\n")
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{ReasoningEffort: true}}, srv)
	cc.model = "gpt-5.6-luna"
	var kinds []DeltaKind
	res, err := cc.CallStream(context.Background(), &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools:    []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking: "medium",
	}, func(d Delta) error {
		kinds = append(kinds, d.Kind)
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if !strings.HasSuffix(path, "/responses") {
		t.Fatalf("path = %q", path)
	}
	if res.ReasoningContent != "plan" || res.Content != "ok" {
		t.Errorf("res = %+v", res)
	}
	if len(kinds) != 2 || kinds[0] != DeltaReasoning || kinds[1] != DeltaContent {
		t.Errorf("kinds = %v", kinds)
	}
	if res.Usage.ReasoningTokens != 2 {
		t.Errorf("usage = %+v", res.Usage)
	}
}

func TestResponsesRequiredShape(t *testing.T) {
	if responsesRequired(fmt.Errorf("plain")) {
		t.Error("plain must not classify")
	}
	if responsesRequired(&APIError{Status: 400, Message: "reasoning_effort is not supported with tools"}) {
		t.Error("legacy effort-rejected must not force Responses")
	}
	msg := "Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'."
	if !responsesRequired(&APIError{Status: 400, Message: msg}) {
		t.Error("gpt-5.6 400 must classify as responses-required")
	}
}
