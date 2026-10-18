package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

// m4TextReply renders an Anthropic text reply (buffered JSON or SSE) whose
// text block is text.
func m4TextReply(text string, stream bool) string {
	q := strconv.Quote(text)
	if !stream {
		return `{"content":[{"type":"text","text":` + q + `}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}`
	}
	// Split the text across two deltas to exercise accumulation.
	h := len(text) / 2
	a, b := strconv.Quote(text[:h]), strconv.Quote(text[h:])
	return `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":` + a + `}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":` + b + `}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`
}

// m4Wire sends req through a real client against an httptest server and
// returns the captured request body plus the result.
func m4Wire(t *testing.T, req *ChatRequest, reply string, stream bool) (map[string]any, *ChatResult) {
	t.Helper()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = fmt.Fprint(w, reply)
	}))
	t.Cleanup(srv.Close)
	c := newTestClient(t, ProviderConfig{ID: "anthropic", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, _ := m2Run(t, c, req, stream)
	return body, res
}

func TestM4_NativeJSONWireShape(t *testing.T) {
	msgs := []Message{{Role: RoleUser, Content: "x"}}
	schemaRF := &ResponseFormat{Type: ResponseJSONSchema, Name: "answer", Schema: testSchema}
	cases := []struct {
		name       string
		req        *ChatRequest
		wantOC     string
		wantTools  int
		wantChoice string // "" = absent
		wantThink  string // "" = absent
	}{
		{"schema", &ChatRequest{Messages: msgs, ResponseFormat: schemaRF},
			`{"format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 0, "", ""},
		{"json_object", &ChatRequest{Messages: msgs, ResponseFormat: &ResponseFormat{Type: ResponseJSONObject}},
			`{"format":{"type":"json_schema","schema":{"type":"object"}}}`, 0, "", ""},
		{"schema without name", &ChatRequest{Messages: msgs, ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: testSchema}},
			`{"format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 0, "", ""},
		{"merged with effort", &ChatRequest{Messages: msgs, Thinking: "high", ResponseFormat: schemaRF},
			`{"effort":"high","format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 0, "", `{"type":"adaptive"}`},
		{"thinking enabled (no effort)", &ChatRequest{Messages: msgs, Thinking: "enabled", ResponseFormat: schemaRF},
			`{"format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 0, "", `{"type":"adaptive"}`},
		{"with tools", &ChatRequest{Messages: msgs, Tools: []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}}, ResponseFormat: schemaRF},
			`{"format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 1, "", ""},
		{"with auto ToolChoice", &ChatRequest{Messages: msgs, Tools: []ToolDef{{Name: "f"}}, ToolChoice: &ToolChoice{Mode: ToolChoiceAuto}, ResponseFormat: schemaRF},
			`{"format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 1, `{"type":"auto"}`, ""},
		{"with forced ToolChoice and adaptive thinking", &ChatRequest{Messages: msgs, Tools: []ToolDef{{Name: "f"}}, ToolChoice: &ToolChoice{Mode: ToolChoiceRequired}, Thinking: "low", ResponseFormat: schemaRF},
			`{"effort":"low","format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 1, `{"type":"any"}`, `{"type":"adaptive"}`},
		{"non-object schema passes through", &ChatRequest{Messages: msgs, ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: json.RawMessage(`{"type":"array"}`)}},
			`{"format":{"type":"json_schema","schema":{"type":"array"}}}`, 0, "", ""},
		{"legacy budget keeps JSON, no effort", &ChatRequest{Messages: msgs, Thinking: "high", ThinkingBudget: 2000, ResponseFormat: schemaRF},
			`{"format":{"type":"json_schema","schema":` + string(testSchema) + `}}`, 0, "", `{"type":"enabled","budget_tokens":2000}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := buildBody(t, FormatAnthropic, "claude", tc.req)
			jsonEq(t, "output_config", m["output_config"], tc.wantOC)
			tools, _ := m["tools"].([]any)
			if len(tools) != tc.wantTools {
				t.Errorf("tools = %v, want %d (no synthetic JSON tool)", tools, tc.wantTools)
			}
			for _, x := range tools {
				if n := x.(map[string]any)["name"]; n == "answer" || n == anthropicJSONToolDefault {
					t.Errorf("synthetic JSON tool leaked: %v", n)
				}
			}
			if tc.wantChoice == "" {
				if m["tool_choice"] != nil {
					t.Errorf("tool_choice = %v, want absent", m["tool_choice"])
				}
			} else {
				jsonEq(t, "tool_choice", m["tool_choice"], tc.wantChoice)
			}
			if tc.wantThink == "" {
				if m["thinking"] != nil {
					t.Errorf("thinking = %v, want absent", m["thinking"])
				}
			} else {
				jsonEq(t, "thinking", m["thinking"], tc.wantThink)
			}
		})
	}
}

// Over a real HTTP exchange, native JSON mode leaves the reply text
// untouched (no folding) with the normal finish mapping, buffered and
// streaming, and sends output_config.format on the wire.
func TestM4_NativeJSONReplyUnchanged(t *testing.T) {
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		Thinking:       "medium",
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Name: "answer", Schema: testSchema}}
	for _, stream := range []bool{false, true} {
		body, res := m4Wire(t, req, m4TextReply(`{"a":1}`, stream), stream)
		jsonEq(t, fmt.Sprintf("stream=%v output_config", stream), body["output_config"],
			`{"effort":"medium","format":{"type":"json_schema","schema":`+string(testSchema)+`}}`)
		if body["tools"] != nil || body["tool_choice"] != nil {
			t.Errorf("stream=%v: wire tools/tool_choice = %v/%v", stream, body["tools"], body["tool_choice"])
		}
		if res.Content != `{"a":1}` || len(res.ToolCalls) != 0 || res.FinishReason != FinishStop {
			t.Errorf("stream=%v: result = %+v", stream, res)
		}
	}
}

// A genuine tool call alongside native JSON mode stays a tool call, even if
// its name equals the schema name (no folding without the quirk).
func TestM4_NativeJSONNoFolding(t *testing.T) {
	reply := `{"content":[{"type":"tool_use","id":"t","name":"answer","input":{"a":1}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}}`
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		Tools:          []ToolDef{{Name: "answer", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Name: "answer", Schema: testSchema}}
	_, res := m4Wire(t, req, reply, false)
	if res.Content != "" || len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "answer" || res.FinishReason != FinishToolCalls {
		t.Errorf("result = %+v, want untouched tool call", res)
	}
}

// Extra["output_config"] replaces the SDK-built object (shallow top-level
// merge); other Extra keys still merge.
func TestM4_ExtraOverridesOutputConfig(t *testing.T) {
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		Thinking:       "high",
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: testSchema},
		Extra:          map[string]any{"output_config": map[string]any{"effort": "xhigh"}, "service_tier": "auto"}}
	for _, stream := range []bool{false, true} {
		body, _ := m4Wire(t, req, m4TextReply(`{"a":1}`, stream), stream)
		jsonEq(t, fmt.Sprintf("stream=%v output_config", stream), body["output_config"], `{"effort":"xhigh"}`)
		if body["service_tier"] != "auto" {
			t.Errorf("stream=%v: other Extra keys must merge: %v", stream, body)
		}
		jsonEq(t, "thinking untouched", body["thinking"], `{"type":"adaptive"}`)
	}
	// Extra can also supply output_config when the SDK builds none.
	plain := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		Extra: map[string]any{"output_config": map[string]any{"effort": "xhigh"}}}
	m, _ := buildBody(t, FormatAnthropic, "claude", plain)
	_ = m
	body, _ := m4Wire(t, plain, m4TextReply("ok", false), false)
	jsonEq(t, "extra only", body["output_config"], `{"effort":"xhigh"}`)
}

// The builtin anthropic provider resolves to the native path through the
// public entry point (no quirk consulted from the client for JSON mode).
func TestM4_NativeJSONViaPublicChat(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, m4TextReply(`{"a":1}`, false))
	}))
	t.Cleanup(srv.Close)
	cfg := ProviderConfig{ID: "anthropic", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k"}
	c := newTestClient(t, cfg, srv)
	res, err := c.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONObject}})
	if err != nil {
		t.Fatal(err)
	}
	jsonEq(t, "output_config", body["output_config"], `{"format":{"type":"json_schema","schema":{"type":"object"}}}`)
	if res.Content != `{"a":1}` {
		t.Errorf("Content = %q", res.Content)
	}
}
