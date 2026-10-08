package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// buildBody renders req for a format (chat-completions unless model routes
// to /responses) and decodes the JSON body.
func buildBody(t *testing.T, f Format, model string, req *ChatRequest) (map[string]any, string) {
	t.Helper()
	pc := newProviderClient(ProviderConfig{ID: string(f), Format: f, BaseURL: "https://x", Quirks: Quirks{ReasoningEffort: f == FormatOpenAI}}, nil, nil)
	b, url, err := pc.buildChatRequest(req, model, false)
	if err != nil {
		t.Fatalf("%s build: %v", f, err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m, url
}

func toolReq(choice *ToolChoice) *ChatRequest {
	return &ChatRequest{
		Messages:   []Message{{Role: RoleUser, Content: "x"}},
		Tools:      []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}, {Name: "g"}},
		ToolChoice: choice,
	}
}

func jsonEq(t *testing.T, name string, got any, want string) {
	t.Helper()
	var w any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, w) {
		g, _ := json.Marshal(got)
		t.Errorf("%s = %s, want %s", name, g, want)
	}
}

func TestToolChoice_PerFormat(t *testing.T) {
	cases := []struct {
		choice                               *ToolChoice
		openai, responses, anthropic, gemini string
	}{
		{&ToolChoice{Mode: ToolChoiceAuto}, `"auto"`, `"auto"`, `{"type":"auto"}`, `{"functionCallingConfig":{"mode":"AUTO"}}`},
		{&ToolChoice{Mode: ToolChoiceNone}, `"none"`, `"none"`, `{"type":"none"}`, `{"functionCallingConfig":{"mode":"NONE"}}`},
		{&ToolChoice{Mode: ToolChoiceRequired}, `"required"`, `"required"`, `{"type":"any"}`, `{"functionCallingConfig":{"mode":"ANY"}}`},
		{ToolChoiceNamed("f"), `{"type":"function","function":{"name":"f"}}`, `{"type":"function","name":"f"}`,
			`{"type":"tool","name":"f"}`, `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["f"]}}`},
	}
	for _, c := range cases {
		m, _ := buildBody(t, FormatOpenAI, "gpt-4o", toolReq(c.choice))
		jsonEq(t, "openai tool_choice", m["tool_choice"], c.openai)
		m, url := buildBody(t, FormatOpenAI, "gpt-5.6", toolReq(c.choice))
		if !strings.HasSuffix(url, "/responses") {
			t.Fatalf("gpt-5.6 + tools must route to /responses: %s", url)
		}
		jsonEq(t, "responses tool_choice", m["tool_choice"], c.responses)
		m, _ = buildBody(t, FormatAnthropic, "claude", toolReq(c.choice))
		jsonEq(t, "anthropic tool_choice", m["tool_choice"], c.anthropic)
		m, _ = buildBody(t, FormatGemini, "gemini", toolReq(c.choice))
		jsonEq(t, "gemini toolConfig", m["toolConfig"], c.gemini)
	}
	// Unset: no field anywhere.
	for _, f := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
		m, _ := buildBody(t, f, "m", toolReq(nil))
		if _, ok := m["tool_choice"]; ok {
			t.Errorf("%s: tool_choice must be absent when unset", f)
		}
		if _, ok := m["toolConfig"]; ok {
			t.Errorf("%s: toolConfig must be absent when unset", f)
		}
	}
}

func TestRequestControls_Validation(t *testing.T) {
	bad := []*ChatRequest{
		{ToolChoice: &ToolChoice{Mode: ToolChoiceAuto}}, // no tools
		toolReq(&ToolChoice{Mode: "sometimes"}),
		toolReq(&ToolChoice{Mode: ToolChoiceTool}),                  // no name
		toolReq(&ToolChoice{Mode: ToolChoiceTool, Name: "missing"}), // unknown tool
		toolReq(&ToolChoice{Mode: ToolChoiceAuto, Name: "f"}),       // name on non-tool mode
		{ResponseFormat: &ResponseFormat{Type: "yaml"}},             // unknown type
		{ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema}}, // no schema
		{ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: json.RawMessage(`[1]`)}},
		{ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: json.RawMessage(`{bad`)}},
		{ResponseFormat: &ResponseFormat{Type: ResponseJSONObject, Schema: json.RawMessage(`{}`)}},
		{ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Name: "has space", Schema: json.RawMessage(`{}`)}},
	}
	for i, req := range bad {
		if req.Messages == nil {
			req.Messages = []Message{{Role: RoleUser, Content: "x"}}
		}
		for _, f := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
			pc := newProviderClient(ProviderConfig{ID: "x", Format: f, BaseURL: "https://x"}, nil, nil)
			_, _, err := pc.buildChatRequest(req, "m", false)
			var ce *ConfigError
			if !errors.As(err, &ce) {
				t.Errorf("case %d %s: err = %v, want ConfigError", i, f, err)
			}
		}
	}
}

func TestParallelToolCallsAndSeed(t *testing.T) {
	no := false
	seed := 42
	req := toolReq(nil)
	req.ParallelToolCalls = &no
	req.Seed = &seed

	m, _ := buildBody(t, FormatOpenAI, "gpt-4o", req)
	if m["parallel_tool_calls"] != false || m["seed"] != float64(42) {
		t.Errorf("openai: parallel=%v seed=%v", m["parallel_tool_calls"], m["seed"])
	}
	m, _ = buildBody(t, FormatOpenAI, "gpt-5.6", req)
	if m["parallel_tool_calls"] != false {
		t.Errorf("responses: parallel=%v", m["parallel_tool_calls"])
	}
	m, _ = buildBody(t, FormatAnthropic, "claude", req)
	jsonEq(t, "anthropic tool_choice", m["tool_choice"], `{"type":"auto","disable_parallel_tool_use":true}`)
	req.ToolChoice = &ToolChoice{Mode: ToolChoiceRequired}
	m, _ = buildBody(t, FormatAnthropic, "claude", req)
	jsonEq(t, "anthropic tool_choice", m["tool_choice"], `{"type":"any","disable_parallel_tool_use":true}`)
	m, _ = buildBody(t, FormatGemini, "gemini", req)
	if m["generationConfig"].(map[string]any)["seed"] != float64(42) {
		t.Errorf("gemini seed: %v", m["generationConfig"])
	}

	// Without tools, parallel_tool_calls is never sent (OpenAI rejects it).
	plain := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, ParallelToolCalls: &no}
	m, _ = buildBody(t, FormatOpenAI, "gpt-4o", plain)
	if _, ok := m["parallel_tool_calls"]; ok {
		t.Error("parallel_tool_calls without tools must be omitted")
	}
	m, _ = buildBody(t, FormatAnthropic, "claude", plain)
	if _, ok := m["tool_choice"]; ok {
		t.Error("anthropic: no tool_choice without tools")
	}
}

// Forced tool use is incompatible with Anthropic extended thinking.
func TestAnthropic_ForcedToolWithThinkingRejected(t *testing.T) {
	for _, choice := range []*ToolChoice{{Mode: ToolChoiceRequired}, ToolChoiceNamed("f")} {
		req := toolReq(choice)
		req.Thinking = "low"
		_, err := buildAnthropicRequest(req, "claude", false)
		var ce *ConfigError
		if !errors.As(err, &ce) {
			t.Errorf("%+v: err = %v, want ConfigError", choice, err)
		}
	}
	req := toolReq(&ToolChoice{Mode: ToolChoiceAuto})
	req.Thinking = "low"
	if _, err := buildAnthropicRequest(req, "claude", false); err != nil {
		t.Errorf("auto + thinking must pass: %v", err)
	}
}

var testSchema = json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`)

func TestResponseFormat_PerFormat(t *testing.T) {
	schemaReq := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Name: "answer", Schema: testSchema, Strict: true}}
	objReq := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONObject}}

	m, _ := buildBody(t, FormatOpenAI, "gpt-4o", schemaReq)
	jsonEq(t, "openai response_format", m["response_format"],
		`{"type":"json_schema","json_schema":{"name":"answer","schema":`+string(testSchema)+`,"strict":true}}`)
	m, _ = buildBody(t, FormatOpenAI, "gpt-4o", objReq)
	jsonEq(t, "openai json_object", m["response_format"], `{"type":"json_object"}`)

	// Responses API (tools + gpt-5.6 routes there).
	rs := *schemaReq
	rs.Tools = []ToolDef{{Name: "f"}}
	m, _ = buildBody(t, FormatOpenAI, "gpt-5.6", &rs)
	jsonEq(t, "responses text.format", m["text"],
		`{"format":{"type":"json_schema","name":"answer","schema":`+string(testSchema)+`,"strict":true}}`)
	ro := *objReq
	ro.Tools = []ToolDef{{Name: "f"}}
	m, _ = buildBody(t, FormatOpenAI, "gpt-5.6", &ro)
	jsonEq(t, "responses json_object", m["text"], `{"format":{"type":"json_object"}}`)

	m, _ = buildBody(t, FormatGemini, "gemini", schemaReq)
	gc := m["generationConfig"].(map[string]any)
	if gc["responseMimeType"] != "application/json" {
		t.Errorf("gemini mime = %v", gc["responseMimeType"])
	}
	jsonEq(t, "gemini responseJsonSchema", gc["responseJsonSchema"], string(testSchema))
	m, _ = buildBody(t, FormatGemini, "gemini", objReq)
	gc = m["generationConfig"].(map[string]any)
	if gc["responseMimeType"] != "application/json" || gc["responseJsonSchema"] != nil {
		t.Errorf("gemini json_object = %v", gc)
	}

	// Anthropic: a forced synthetic tool carries the schema.
	m, _ = buildBody(t, FormatAnthropic, "claude", schemaReq)
	jsonEq(t, "anthropic tool_choice", m["tool_choice"], `{"type":"tool","name":"answer"}`)
	tools := m["tools"].([]any)
	last := tools[len(tools)-1].(map[string]any)
	if last["name"] != "answer" {
		t.Fatalf("anthropic json tool = %v", last)
	}
	jsonEq(t, "anthropic json tool schema", last["input_schema"], string(testSchema))
	m, _ = buildBody(t, FormatAnthropic, "claude", objReq)
	tools = m["tools"].([]any)
	if tools[0].(map[string]any)["name"] != anthropicJSONToolDefault {
		t.Errorf("anthropic default json tool name = %v", tools[0])
	}

	// Text is the default: nothing is sent.
	txt := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, ResponseFormat: &ResponseFormat{Type: ResponseText}}
	for _, f := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
		m, _ := buildBody(t, f, "m", txt)
		if m["response_format"] != nil || m["tools"] != nil || m["generationConfig"] != nil {
			t.Errorf("%s: text format must send nothing: %v", f, m)
		}
	}
}

func TestResponseFormat_AnthropicConflicts(t *testing.T) {
	rf := &ResponseFormat{Type: ResponseJSONSchema, Name: "f", Schema: testSchema}
	cases := []*ChatRequest{
		{Tools: []ToolDef{{Name: "f"}}, ResponseFormat: rf},                                                // name collision
		{Tools: []ToolDef{{Name: "g"}}, ToolChoice: &ToolChoice{Mode: ToolChoiceAuto}, ResponseFormat: rf}, // forced tool vs ToolChoice
		{Thinking: "low", ResponseFormat: &ResponseFormat{Type: ResponseJSONObject}},                       // forced tool vs thinking
	}
	for i, req := range cases {
		req.Messages = []Message{{Role: RoleUser, Content: "x"}}
		_, err := buildAnthropicRequest(req, "claude", false)
		var ce *ConfigError
		if !errors.As(err, &ce) {
			t.Errorf("case %d: err = %v, want ConfigError", i, err)
		}
	}
}

// The JSON tool call is folded back into Content on both paths.
func TestResponseFormat_AnthropicResultFolding(t *testing.T) {
	srvBody := `{"content":[{"type":"tool_use","id":"t","name":"answer","input":{"a":1}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}}`
	sse := strings.Join([]string{
		`event: message_start` + "\n" + `data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"answer"}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n") + "\n\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.Header.Get("Accept"), "event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w, sse)
			return
		}
		_, _ = fmt.Fprint(w, srvBody)
	}))
	t.Cleanup(srv.Close)
	c := newTestClient(t, ProviderConfig{ID: "anthropic", Format: FormatAnthropic, BaseURL: srv.URL}, srv)
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Name: "answer", Schema: testSchema}}

	res, err := c.Call(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != `{"a":1}` || len(res.ToolCalls) != 0 || res.FinishReason != FinishStop {
		t.Errorf("buffered = %+v", res)
	}

	var deltas []Delta
	res, err = c.CallStream(context.Background(), req, func(d Delta) error { deltas = append(deltas, d); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != `{"a":1}` || len(res.ToolCalls) != 0 || res.FinishReason != FinishStop {
		t.Errorf("stream = %+v", res)
	}
	for _, d := range deltas {
		if d.Kind != DeltaContent {
			t.Errorf("JSON mode must stream content deltas only, got %+v", d)
		}
	}
}

// A real tool call alongside JSON mode is kept as a tool call.
func TestFoldAnthropicJSON_KeepsOtherCalls(t *testing.T) {
	res := &ChatResult{FinishReason: FinishToolCalls, ToolCalls: []ToolCall{{Name: "answer", Arguments: `{"a":1}`}, {Name: "g", Arguments: "{}"}}}
	foldAnthropicJSON(res, "answer")
	if res.Content != `{"a":1}` || len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "g" || res.FinishReason != FinishToolCalls {
		t.Errorf("folded = %+v", res)
	}
	foldAnthropicJSON(nil, "answer") // nil-safe
	same := &ChatResult{Content: "x"}
	foldAnthropicJSON(same, "")
	if same.Content != "x" {
		t.Error("inactive JSON mode must not touch the result")
	}
}
