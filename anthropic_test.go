package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestBuildAnthropicRequest_Golden(t *testing.T) {
	req := &ChatRequest{
		System: []SystemBlock{{Text: "Be terse."}, {Text: "cached", Cache: true}},
		Messages: []Message{
			{Role: RoleUser, Content: "Hi"},
			{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{{ID: "t1", Name: "f", Arguments: `{"a":1}`}}},
			{Role: RoleTool, ToolCallID: "t1", ToolName: "f", Content: "42"},
			{Role: RoleTool, ToolCallID: "t2", ToolName: "g", Content: "43"},
		},
		Tools:          []ToolDef{{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking:       "enabled",
		ThinkingBudget: 2000,
	}
	body, err := buildAnthropicRequest(req, "claude-sonnet-4", false)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("decode: %v\n%s", err, body)
	}

	// system: two blocks, second carries cache_control.
	sys := m["system"].([]any)
	if len(sys) != 2 {
		t.Fatalf("system blocks = %d, want 2", len(sys))
	}
	if sys[0].(map[string]any)["text"] != "Be terse." {
		t.Errorf("sys[0] = %v", sys[0])
	}
	if sys[1].(map[string]any)["cache_control"] == nil {
		t.Error("sys[1] missing cache_control")
	}

	// messages: user, assistant(tool_use), merged tool_result user turn.
	msgs := m["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want 3 (tool results merged into one user turn)", len(msgs))
	}
	asst := msgs[1].(map[string]any)
	blocks := asst["content"].([]any)
	if len(blocks) != 1 {
		t.Fatalf("assistant blocks = %d, want 1 tool_use", len(blocks))
	}
	tu := blocks[0].(map[string]any)
	if tu["type"] != "tool_use" || tu["id"] != "t1" {
		t.Errorf("tool_use block = %v", tu)
	}
	toolTurn := msgs[2].(map[string]any)
	if toolTurn["role"] != "user" {
		t.Errorf("tool turn role = %v, want user", toolTurn["role"])
	}
	results := toolTurn["content"].([]any)
	if len(results) != 2 {
		t.Fatalf("tool_result blocks = %d, want 2 (merged consecutive tool messages)", len(results))
	}
	for i, wantID := range []string{"t1", "t2"} {
		tr := results[i].(map[string]any)
		if tr["type"] != "tool_result" || tr["tool_use_id"] != wantID {
			t.Errorf("result[%d] = %v, want tool_result %s", i, tr, wantID)
		}
	}

	// max_tokens default + thinking budget.
	if m["max_tokens"].(float64) != 8192 {
		t.Errorf("max_tokens = %v, want default 8192", m["max_tokens"])
	}
	th := m["thinking"].(map[string]any)
	if th["type"] != "enabled" || th["budget_tokens"].(float64) != 2000 {
		t.Errorf("thinking = %v", th)
	}
	// tools carry input_schema.
	tool := m["tools"].([]any)[0].(map[string]any)
	if tool["input_schema"] == nil {
		t.Error("tool missing input_schema")
	}
}

func TestBuildAnthropicRequest_UserCacheMarker(t *testing.T) {
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "cached turn", Cache: true},
		{Role: RoleUser, Content: "plain turn"},
	}}
	body, err := buildAnthropicRequest(req, "claude", false)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("decode: %v\n%s", err, body)
	}
	msgs := m["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2", len(msgs))
	}
	cached := msgs[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if cached["text"] != "cached turn" {
		t.Errorf("cached text = %v", cached["text"])
	}
	cc, ok := cached["cache_control"].(map[string]any)
	if !ok || cc["type"] != "ephemeral" {
		t.Errorf("cached user block cache_control = %v, want {type:ephemeral}", cached["cache_control"])
	}
	plain := msgs[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if _, ok := plain["cache_control"]; ok {
		t.Errorf("plain user block must omit cache_control, got %v", plain["cache_control"])
	}
}

func TestBuildAnthropicRequest_EmptyAssistantPlaceholder(t *testing.T) {
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant},
		{Role: RoleUser, Content: "again"},
	}}
	body, _ := buildAnthropicRequest(req, "claude", false)
	var m map[string]any
	json.Unmarshal(body, &m)
	msgs := m["messages"].([]any)
	asst := msgs[1].(map[string]any)
	blocks := asst["content"].([]any)
	if len(blocks) != 1 || blocks[0].(map[string]any)["type"] != "text" {
		t.Fatalf("empty assistant blocks = %v, want one placeholder text", blocks)
	}
}

func TestBuildAnthropicRequest_ThinkingLevels(t *testing.T) {
	cases := map[string]int{"low": 1024, "medium": 8192, "high": 16384}
	for level, want := range cases {
		req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, Thinking: level}
		body, _ := buildAnthropicRequest(req, "claude", false)
		var m map[string]any
		json.Unmarshal(body, &m)
		th := m["thinking"].(map[string]any)
		if th["budget_tokens"].(float64) != float64(want) {
			t.Errorf("%s budget = %v, want %d", level, th["budget_tokens"], want)
		}
	}
	// disabled → no thinking field
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, Thinking: "disabled"}
	body, _ := buildAnthropicRequest(req, "claude", false)
	var m map[string]any
	json.Unmarshal(body, &m)
	if _, ok := m["thinking"]; ok {
		t.Error("disabled thinking must omit the field")
	}
}

func TestParseAnthropicResponse(t *testing.T) {
	body := []byte(`{
		"content": [
			{"type":"thinking","thinking":"pondering"},
			{"type":"text","text":"Answer"},
			{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}
		],
		"stop_reason":"tool_use",
		"usage":{"input_tokens":5,"output_tokens":9}
	}`)
	res, err := parseAnthropicResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "Answer" || res.ReasoningContent != "pondering" {
		t.Errorf("content: %+v", res)
	}
	if res.FinishReason != FinishToolCalls {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "t1" || res.ToolCalls[0].Arguments != `{"a":1}` {
		t.Errorf("tool calls: %+v", res.ToolCalls)
	}
	if res.Usage.PromptTokens != 5 || res.Usage.CompletionTokens != 9 {
		t.Errorf("usage: %+v", res.Usage)
	}
}

func anthropicStreamFixture() []string {
	return []string{
		`{"type":"message_start","message":{"usage":{"input_tokens":11}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"f","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}`,
		`{"type":"message_stop"}`,
	}
}

func TestMapAnthropicStreamEvent_FullFlow(t *testing.T) {
	acc := newStreamAccum()
	var contentDeltas, toolDeltas int
	done := false
	for _, ev := range anthropicStreamFixture() {
		ds, d, err := mapAnthropicStreamEvent([]byte(ev), acc)
		if err != nil {
			t.Fatalf("%s: %v", ev, err)
		}
		done = done || d
		for _, dd := range ds {
			switch dd.Kind {
			case DeltaContent:
				contentDeltas++
			case DeltaToolArgs:
				toolDeltas++
			}
		}
	}
	if !done {
		t.Fatal("message_stop must set done")
	}
	res := acc.result()
	if res.Content != "Hi" {
		t.Errorf("content = %q", res.Content)
	}
	if res.Usage.PromptTokens != 11 || res.Usage.CompletionTokens != 6 {
		t.Errorf("usage = %+v (input from message_start, output from message_delta)", res.Usage)
	}
	if res.FinishReason != FinishToolCalls {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != `{"a":1}` || res.ToolCalls[0].Name != "f" {
		t.Errorf("tool calls: %+v", res.ToolCalls)
	}
	if contentDeltas != 1 || toolDeltas != 3 { // start + 2 arg fragments
		t.Errorf("deltas: content=%d tool=%d", contentDeltas, toolDeltas)
	}
}

func TestMapAnthropicStreamEvent_Error(t *testing.T) {
	acc := newStreamAccum()
	_, _, err := mapAnthropicStreamEvent([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"boom"}}`), acc)
	if err == nil {
		t.Fatal("expected error event")
	}
}

func TestListModelsAnthropic_Pagination(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RequestURI())
		if r.Header.Get("x-api-key") != "k" || r.Header.Get("anthropic-version") != "2023-06-01" {
			w.WriteHeader(400)
			return
		}
		if r.URL.Query().Get("after_id") == "" {
			fmt.Fprint(w, `{"data":[{"id":"claude-a","display_name":"Claude A","created_at":"2026-01-02T15:04:05Z"}],"has_more":true,"last_id":"claude-a"}`)
			return
		}
		fmt.Fprint(w, `{"data":[{"id":"claude-b"}],"has_more":false}`)
	}))
	defer srv.Close()

	pc := newProviderClient(ProviderConfig{ID: "anthropic", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{AnthropicVersion: "2023-06-01"}},
		srv.Client(), srv.Client())
	models, err := listModelsAnthropic(context.Background(), pc)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "claude-a" || models[1].ID != "claude-b" {
		t.Fatalf("models = %+v", models)
	}
	if models[0].CreatedAt.IsZero() {
		t.Errorf("CreatedAt = %v, want parsed RFC3339", models[0].CreatedAt)
	}
	if len(paths) != 2 || paths[1] != "/v1/models?limit=100&after_id=claude-a" {
		t.Errorf("requests = %v, want after_id follow-up", paths)
	}
}
