package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestResponsesFinishReason_Table(t *testing.T) {
	inc := func(reason string) *rsResponse {
		r := &rsResponse{Status: "incomplete"}
		if reason != "" {
			r.IncompleteDetails = &struct {
				Reason string `json:"reason"`
			}{Reason: reason}
		}
		return r
	}
	cases := []struct {
		name     string
		r        *rsResponse
		hasTools bool
		want     string
	}{
		{"completed", &rsResponse{Status: "completed"}, false, FinishStop},
		{"completed tools", &rsResponse{Status: "completed"}, true, FinishToolCalls},
		{"max output", inc("max_output_tokens"), true, FinishLength},
		{"content filter", inc("content_filter"), false, FinishContentFilter},
		{"content filter with tools", inc("content_filter"), true, FinishContentFilter},
		{"incomplete unknown, tools", inc("other"), true, FinishToolCalls},
		{"incomplete no details", inc(""), false, FinishLength},
		{"failed", &rsResponse{Status: "failed"}, true, ""},
		{"cancelled", &rsResponse{Status: "cancelled"}, false, ""},
	}
	for _, c := range cases {
		if got := responsesFinishReason(c.r, c.hasTools); got != c.want {
			t.Errorf("%s: %q, want %q", c.name, got, c.want)
		}
	}
}

func TestParseResponsesAPI_Errors(t *testing.T) {
	for body, want := range map[string]string{
		`{bad`:                          "parse responses",
		`{"error":{"message":"quota"}}`: "quota",
		`{"status":"failed"}`:           "responses failed",
		`{"status":"failed","error":{"message":""}}`: "responses failed",
	} {
		if _, err := parseResponsesAPI([]byte(body)); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err = %v, want %q", body, err, want)
		}
	}
	res, err := parseResponsesAPI([]byte(`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"},{"type":"refusal","text":"x"}]}],
		"usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":6},"output_tokens_details":{"reasoning_tokens":2}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "hi" || res.FinishReason != FinishStop {
		t.Errorf("result = %+v", res)
	}
	if u := res.Usage; u.PromptTokens != 4 || u.CacheReadTokens != 6 || u.ReasoningTokens != 2 || u.InputTokens() != 10 {
		t.Errorf("usage = %+v", u)
	}
	if u := usageFromResponses(nil); u != (Usage{}) {
		t.Errorf("nil usage = %+v", u)
	}
}

func TestMapResponsesStreamEvent_Edges(t *testing.T) {
	if _, _, err := mapResponsesStreamEvent([]byte(`{bad`), newStreamAccum()); err == nil {
		t.Error("malformed event must error")
	}
	// Terminal events without a response object fall back on accumulator state.
	cases := []struct {
		events []string
		want   string
	}{
		{[]string{`{"type":"response.completed"}`}, FinishStop},
		{[]string{`{"type":"response.incomplete"}`}, FinishLength},
		{[]string{`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c","name":"f"}}`, `{"type":"response.completed"}`}, FinishToolCalls},
	}
	for i, c := range cases {
		acc := newStreamAccum()
		var done bool
		for _, e := range c.events {
			_, d, err := mapResponsesStreamEvent([]byte(e), acc)
			if err != nil {
				t.Fatal(err)
			}
			done = d
		}
		if !done || acc.finishReason != c.want || !acc.sawFinish {
			t.Errorf("case %d: done=%v finish=%q", i, done, acc.finishReason)
		}
	}
	for ev, want := range map[string]string{
		`{"type":"error"}`:           "responses stream error",
		`{"type":"response.failed"}`: "responses failed",
		`{"type":"response.failed","response":{"error":{"message":"boom"}}}`: "boom",
	} {
		_, done, err := mapResponsesStreamEvent([]byte(ev), newStreamAccum())
		if !done || err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: done=%v err=%v", ev, done, err)
		}
	}
	// Reasoning summary and output text deltas accumulate; empty deltas emit nothing.
	acc := newStreamAccum()
	for _, e := range []string{
		`{"type":"response.reasoning_summary_text.delta","delta":"r"}`,
		`{"type":"response.reasoning_summary_text.delta","delta":""}`,
		`{"type":"response.output_text.delta","delta":"t"}`,
		`{"type":"response.output_text.delta","delta":""}`,
	} {
		_, _, _ = mapResponsesStreamEvent([]byte(e), acc)
	}
	if r := acc.result(); r.ReasoningContent != "r" || r.Content != "t" {
		t.Errorf("deltas = %+v", r)
	}
}

func TestBuildResponsesRequest_Fields(t *testing.T) {
	no := false
	req := &ChatRequest{
		System:            []SystemBlock{{Text: "base\n"}},
		Messages:          []Message{{Role: RoleSystem, Content: "inband"}, {Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: "a"}, {Role: RoleTool, ToolCallID: "c", Content: "r"}},
		Tools:             []ToolDef{{Name: "f"}},
		MaxTokens:         99,
		Temperature:       -1,
		TopP:              -1,
		Thinking:          "disabled",
		ToolChoice:        &ToolChoice{Mode: ToolChoiceAuto},
		ParallelToolCalls: &no,
	}
	b, _ := json.Marshal(buildResponsesRequest(req, "gpt-4.1", false))
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	if m["instructions"] != "base\n\ninband" || m["max_output_tokens"] != float64(99) || m["temperature"] != float64(0) || m["top_p"] != float64(0) {
		t.Errorf("request = %v", m)
	}
	r := m["reasoning"].(map[string]any)
	if r["effort"] != "none" || r["summary"] != nil {
		t.Errorf("disabled reasoning = %v", r)
	}
	// Reasoning-only models never get sampling fields.
	b, _ = json.Marshal(buildResponsesRequest(req, "gpt-5.6", false))
	m = nil
	_ = json.Unmarshal(b, &m)
	if m["temperature"] != nil || m["top_p"] != nil {
		t.Errorf("gpt-5.6 sampling = %v / %v", m["temperature"], m["top_p"])
	}
	for in, want := range map[string]string{"": "medium", "enabled": "medium", "low": "low", "max": "max", "disabled": "none"} {
		if got := responsesEffort(in); got != want {
			t.Errorf("effort(%q) = %q, want %q", in, got, want)
		}
	}
	// Empty conversation still sends an input array.
	if _, in := buildResponsesInput(&ChatRequest{}); in == nil || len(in) != 0 {
		t.Errorf("empty input = %#v", in)
	}
	// gpt-5.4 diverts only with explicit thinking; gpt-5.6 always with tools.
	tools := &ChatRequest{Tools: []ToolDef{{Name: "f"}}}
	if useResponsesAPI(nil, FormatOpenAI, "gpt-5.4", tools) {
		t.Error("gpt-5.4 without thinking must stay on chat completions")
	}
	tools.Thinking = "high"
	if !useResponsesAPI(nil, FormatOpenAI, "gpt-5.4", tools) || !useResponsesAPI(nil, FormatOpenAI, "gpt-6", tools) {
		t.Error("explicit thinking + tools must divert gpt-5.4 / gpt-6")
	}
}
