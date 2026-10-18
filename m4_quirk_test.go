package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// m4QuirkGateway registers a custom Anthropic-format provider through the
// public options, with the AnthropicToolJSON quirk set as given, and returns
// a chat client plus a pointer to the last captured request body.
func m4QuirkGateway(t *testing.T, quirkOn bool, reply func(stream bool) string) (*ChatClient, *map[string]any) {
	t.Helper()
	body := map[string]any{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = map[string]any{}
		_ = json.Unmarshal(b, &body)
		stream, _ := body["stream"].(bool)
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = fmt.Fprint(w, reply(stream))
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("gw",
		WithFormat(FormatAnthropic), WithBaseURL(srv.URL), WithAPIKey("k"),
		WithQuirks(Quirks{AnthropicToolJSON: quirkOn})))
	c, err := sdk.Chat("gw", "claude-x")
	if err != nil {
		t.Fatal(err)
	}
	return c, &body
}

func m4ToolReply(stream bool) string {
	if !stream {
		return `{"content":[{"type":"tool_use","id":"t","name":"answer","input":{"a":1}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}}`
	}
	return strings.Join([]string{
		`event: message_start` + "\n" + `data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"answer"}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n") + "\n\n"
}

func m4JSONReq() *ChatRequest {
	return &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Name: "answer", Schema: testSchema}}
}

func TestM4Quirk_BuiltinAnthropicOff(t *testing.T) {
	p, err := New().Provider("anthropic")
	if err != nil {
		t.Fatal(err)
	}
	if p.Config().Quirks.AnthropicToolJSON {
		t.Error("built-in anthropic must have AnthropicToolJSON off")
	}
}

func TestM4Quirk_CustomProviderSetsFlag(t *testing.T) {
	sdk := New(WithProvider("gw", WithFormat(FormatAnthropic), WithBaseURL("http://x"), WithAPIKey("k"),
		WithQuirks(Quirks{AnthropicToolJSON: true})))
	p, err := sdk.Provider("gw")
	if err != nil {
		t.Fatal(err)
	}
	if !p.Config().Quirks.AnthropicToolJSON {
		t.Error("custom provider must carry AnthropicToolJSON through WithQuirks")
	}
}

// Quirk ON: forced synthetic tool, tool_choice, no output_config.format.
func TestM4Quirk_EmulationRequestBody(t *testing.T) {
	c, body := m4QuirkGateway(t, true, m4ToolReply)
	if _, err := c.Call(context.Background(), m4JSONReq()); err != nil {
		t.Fatal(err)
	}
	jsonEq(t, "tool_choice", (*body)["tool_choice"], `{"type":"tool","name":"answer"}`)
	tools, _ := (*body)["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", (*body)["tools"])
	}
	last := tools[len(tools)-1].(map[string]any)
	if last["name"] != "answer" {
		t.Fatalf("json tool = %v", last)
	}
	jsonEq(t, "json tool schema", last["input_schema"], string(testSchema))
	if oc, ok := (*body)["output_config"].(map[string]any); ok && oc["format"] != nil {
		t.Errorf("emulation must not send output_config.format: %v", oc)
	}

	// json_object: default tool name and {"type":"object"} schema.
	obj := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONObject}}
	if _, err := c.Call(context.Background(), obj); err != nil {
		t.Fatal(err)
	}
	tools, _ = (*body)["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["name"] != anthropicJSONToolDefault {
		t.Fatalf("default json tool = %v", (*body)["tools"])
	}
	jsonEq(t, "default schema", tools[0].(map[string]any)["input_schema"], `{"type":"object"}`)
}

// Quirk ON: the tool input is folded back into Content (buffered + stream).
func TestM4Quirk_EmulationResultFolding(t *testing.T) {
	c, _ := m4QuirkGateway(t, true, m4ToolReply)
	res, err := c.Call(context.Background(), m4JSONReq())
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != `{"a":1}` || len(res.ToolCalls) != 0 || res.FinishReason != FinishStop {
		t.Errorf("buffered = %+v", res)
	}
	var deltas []Delta
	res, err = c.CallStream(context.Background(), m4JSONReq(), func(d Delta) error { deltas = append(deltas, d); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != `{"a":1}` || len(res.ToolCalls) != 0 || res.FinishReason != FinishStop {
		t.Errorf("stream = %+v", res)
	}
	if len(deltas) == 0 {
		t.Error("expected streamed deltas")
	}
	for _, d := range deltas {
		if d.Kind != DeltaContent {
			t.Errorf("JSON mode must stream content deltas only, got %+v", d)
		}
	}
}

// Quirk OFF on the same wire reply: no folding, native path.
func TestM4Quirk_OffDoesNotFold(t *testing.T) {
	c, body := m4QuirkGateway(t, false, m4ToolReply)
	res, err := c.Call(context.Background(), m4JSONReq())
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "" || len(res.ToolCalls) != 1 {
		t.Errorf("quirk off must not fold: %+v", res)
	}
	if (*body)["tool_choice"] != nil {
		t.Errorf("quirk off must not force a tool: %v", (*body)["tool_choice"])
	}
}

// Quirk ON: the four pre-change ConfigErrors (plus name collision).
func TestM4Quirk_EmulationConfigErrors(t *testing.T) {
	rf := &ResponseFormat{Type: ResponseJSONSchema, Name: "f", Schema: testSchema}
	msgs := []Message{{Role: RoleUser, Content: "x"}}
	cases := []struct {
		name string
		req  *ChatRequest
		want string
	}{
		{"tool name collision", &ChatRequest{Messages: msgs, Tools: []ToolDef{{Name: "f"}}, ResponseFormat: rf}, "Tools"},
		{"ToolChoice", &ChatRequest{Messages: msgs, ToolChoice: &ToolChoice{Mode: ToolChoiceAuto}, ResponseFormat: rf}, "ToolChoice"},
		{"Tools", &ChatRequest{Messages: msgs, Tools: []ToolDef{{Name: "g"}}, ResponseFormat: rf}, "Tools"},
		{"thinking budget", &ChatRequest{Messages: msgs, Thinking: "low", ThinkingBudget: 2000,
			ResponseFormat: &ResponseFormat{Type: ResponseJSONObject}}, "extended thinking"},
		{"adaptive thinking", &ChatRequest{Messages: msgs, Thinking: "low",
			ResponseFormat: &ResponseFormat{Type: ResponseJSONObject}}, "extended thinking"},
		{"non-object schema", &ChatRequest{Messages: msgs,
			ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: json.RawMessage(`{"type":"array"}`)}}, "object"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pc := newProviderClient(ProviderConfig{ID: "gw", Format: FormatAnthropic, BaseURL: "https://x",
				Quirks: Quirks{AnthropicToolJSON: true}}, nil, nil)
			_, _, err := pc.buildChatRequest(tc.req, "claude", false)
			var ce *ConfigError
			if !errors.As(err, &ce) {
				t.Fatalf("err = %v, want ConfigError", err)
			}
			if !strings.Contains(ce.Error(), tc.want) {
				t.Errorf("err = %q, want mention of %q", ce.Error(), tc.want)
			}
		})
	}
}
