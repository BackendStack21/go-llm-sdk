package llm

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func decodeAnthropicBody(t *testing.T, req *ChatRequest) map[string]any {
	t.Helper()
	b, err := buildAnthropicRequest(req, "claude-test", false)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// Adaptive presets default max_tokens to 16384 (AC-3); the legacy explicit
// budget path keeps budget + visible-output room (AC-2).
func TestBuildAnthropicRequest_ThinkingDefaultMaxTokens(t *testing.T) {
	for _, level := range []string{"enabled", "low", "medium", "high", "max"} {
		m := decodeAnthropicBody(t, &ChatRequest{Thinking: level, Messages: []Message{{Role: RoleUser, Content: "x"}}})
		if maxTok := m["max_tokens"].(float64); maxTok != 16384 {
			t.Errorf("%s: max_tokens = %v, want 16384", level, maxTok)
		}
	}
	m := decodeAnthropicBody(t, &ChatRequest{Thinking: "high", ThinkingBudget: 3000, Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if maxTok := m["max_tokens"].(float64); maxTok != 3000+anthropicDefaultMaxTokens {
		t.Errorf("budget path: max_tokens = %v, want budget+%d", maxTok, anthropicDefaultMaxTokens)
	}
	m = decodeAnthropicBody(t, &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if maxTok := m["max_tokens"].(float64); maxTok != anthropicDefaultMaxTokens {
		t.Errorf("no thinking: max_tokens = %v, want %d", maxTok, anthropicDefaultMaxTokens)
	}
}

// Under adaptive thinking an explicit MaxTokens is sent as-is: no clamping,
// no ConfigError (AC-3). Replaces the preset-clamp expectations.
func TestBuildAnthropicRequest_AdaptiveMaxTokensAsIs(t *testing.T) {
	for _, mt := range []int{1024, 1500, 4000} {
		m := decodeAnthropicBody(t, &ChatRequest{Thinking: "high", MaxTokens: mt, Messages: []Message{{Role: RoleUser, Content: "x"}}})
		if m["max_tokens"].(float64) != float64(mt) {
			t.Errorf("max_tokens = %v, want %d as-is", m["max_tokens"], mt)
		}
		if th := m["thinking"].(map[string]any); th["type"] != "adaptive" {
			t.Errorf("thinking = %v, want adaptive", th)
		}
	}
}

// The explicit-budget path is unchanged: N >= 1024, max_tokens > N (AC-2).
func TestBuildAnthropicRequest_ExplicitBudgetPath(t *testing.T) {
	m := decodeAnthropicBody(t, &ChatRequest{Thinking: "medium", ThinkingBudget: 500, Messages: []Message{{Role: RoleUser, Content: "x"}}})
	th := m["thinking"].(map[string]any)
	if th["type"] != "enabled" || th["budget_tokens"].(float64) != 1024 {
		t.Errorf("thinking = %v, want enabled/1024 floor", th)
	}
	if _, ok := m["output_config"]; ok {
		t.Errorf("budget path must not emit output_config.effort: %v", m["output_config"])
	}
}

// Contradictory explicit settings fail fast instead of a guaranteed 400.
func TestBuildAnthropicRequest_ThinkingBudgetConflicts(t *testing.T) {
	cases := []ChatRequest{
		{Thinking: "enabled", ThinkingBudget: 8000, MaxTokens: 8000},
		{Thinking: "enabled", ThinkingBudget: 2000, MaxTokens: 1500},
		{Thinking: "low", ThinkingBudget: 1024, MaxTokens: 1024}, // explicit budget cannot fit under the cap
	}
	for i, req := range cases {
		req.Messages = []Message{{Role: RoleUser, Content: "x"}}
		_, err := buildAnthropicRequest(&req, "claude-test", false)
		var ce *ConfigError
		if !errors.As(err, &ce) {
			t.Errorf("case %d: err = %v, want ConfigError", i, err)
		}
	}
}

// Extended thinking rejects modified temperature / top_p.
func TestBuildAnthropicRequest_ThinkingOmitsSampling(t *testing.T) {
	for _, req := range []ChatRequest{{Thinking: "low"}, {Thinking: "low", ThinkingBudget: 2000}} { // adaptive and budget
		req.Temperature, req.TopP = 0.3, 0.5
		req.Messages = []Message{{Role: RoleUser, Content: "x"}}
		m := decodeAnthropicBody(t, &req)
		if _, ok := m["temperature"]; ok {
			t.Errorf("%+v: temperature must be omitted with thinking", req.Thinking)
		}
		if _, ok := m["top_p"]; ok {
			t.Error("top_p must be omitted with thinking")
		}
	}
	m := decodeAnthropicBody(t, &ChatRequest{Temperature: 0.3, Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if m["temperature"].(float64) != 0.3 {
		t.Error("temperature must survive without thinking")
	}
}

func TestParseAnthropicResponse_ThinkingBlocks(t *testing.T) {
	res, err := parseAnthropicResponse([]byte(`{"content":[
		{"type":"thinking","thinking":"a","signature":"S1"},
		{"type":"redacted_thinking","data":"ENC"},
		{"type":"thinking","thinking":"b","signature":"S2"},
		{"type":"tool_use","id":"t","name":"f","input":{}}],"stop_reason":"tool_use"}`))
	if err != nil {
		t.Fatal(err)
	}
	want := []ThinkingBlock{{Text: "a", Signature: "S1"}, {Redacted: "ENC"}, {Text: "b", Signature: "S2"}}
	if len(res.ThinkingBlocks) != len(want) {
		t.Fatalf("blocks = %+v", res.ThinkingBlocks)
	}
	for i := range want {
		if res.ThinkingBlocks[i] != want[i] {
			t.Errorf("block %d = %+v, want %+v", i, res.ThinkingBlocks[i], want[i])
		}
	}
	if res.ReasoningContent != "ab" || res.ThinkingSignature != "S2" {
		t.Errorf("legacy fields = %q / %q", res.ReasoningContent, res.ThinkingSignature)
	}
}

func TestMapAnthropicStreamEvent_ThinkingBlocks(t *testing.T) {
	acc := newStreamAccum()
	for _, e := range []string{
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"a"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"S1"}}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"ENC"}}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"b"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"S2"}}`,
	} {
		if _, _, err := mapAnthropicStreamEvent([]byte(e), acc); err != nil {
			t.Fatal(err)
		}
	}
	res := acc.result()
	want := []ThinkingBlock{{Text: "a", Signature: "S1"}, {Redacted: "ENC"}, {Text: "b", Signature: "S2"}}
	if len(res.ThinkingBlocks) != 3 {
		t.Fatalf("blocks = %+v", res.ThinkingBlocks)
	}
	for i := range want {
		if res.ThinkingBlocks[i] != want[i] {
			t.Errorf("block %d = %+v, want %+v", i, res.ThinkingBlocks[i], want[i])
		}
	}
	// Signatures from different blocks are never concatenated.
	if res.ThinkingSignature != "S2" {
		t.Errorf("ThinkingSignature = %q, want last block's S2", res.ThinkingSignature)
	}
}

func TestBuildAnthropicRequest_ReplaysThinkingBlocksInOrder(t *testing.T) {
	m := decodeAnthropicBody(t, &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, Content: "ok", ReasoningContent: "ignored", ThinkingSignature: "ignored",
			ThinkingBlocks: []ThinkingBlock{{Text: "a", Signature: "S1"}, {Redacted: "ENC"}},
			ToolCalls:      []ToolCall{{ID: "t", Name: "f", Arguments: `{}`}}},
		{Role: RoleTool, ToolCallID: "t", Content: "r"},
	}})
	blocks := m["messages"].([]any)[1].(map[string]any)["content"].([]any)
	types := []string{}
	for _, b := range blocks {
		types = append(types, b.(map[string]any)["type"].(string))
	}
	want := []string{"thinking", "redacted_thinking", "text", "tool_use"}
	if len(types) != len(want) {
		t.Fatalf("types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("types = %v, want %v", types, want)
		}
	}
	b0 := blocks[0].(map[string]any)
	if b0["thinking"] != "a" || b0["signature"] != "S1" {
		t.Errorf("thinking block = %v", b0)
	}
	if blocks[1].(map[string]any)["data"] != "ENC" {
		t.Errorf("redacted block = %v", blocks[1])
	}
}

func TestBuildAnthropicRequest_ThinkingBlockWithoutSignatureRejected(t *testing.T) {
	_, err := buildAnthropicRequest(&ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ThinkingBlocks: []ThinkingBlock{{Text: "a"}}},
	}}, "claude", false)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %v, want ConfigError", err)
	}
}

func TestChatResult_AssistantMessage(t *testing.T) {
	res := &ChatResult{
		Content: "c", ReasoningContent: "r", ThinkingSignature: "s",
		ThinkingBlocks: []ThinkingBlock{{Text: "r", Signature: "s"}},
		ToolCalls:      []ToolCall{{ID: "1", Name: "f", Arguments: "{}", Signature: "g"}},
	}
	m := res.AssistantMessage()
	if m.Role != RoleAssistant || m.Content != "c" || m.ReasoningContent != "r" || m.ThinkingSignature != "s" ||
		len(m.ThinkingBlocks) != 1 || len(m.ToolCalls) != 1 || m.ToolCalls[0].Signature != "g" {
		t.Fatalf("message = %+v", m)
	}
	// Copies, not aliases.
	m.ToolCalls[0].ID = "x"
	m.ThinkingBlocks[0].Text = "x"
	if res.ToolCalls[0].ID != "1" || res.ThinkingBlocks[0].Text != "r" {
		t.Error("AssistantMessage must copy slices")
	}
	var nilRes *ChatResult
	if got := nilRes.AssistantMessage(); got.Role != RoleAssistant {
		t.Errorf("nil result → %+v", got)
	}
}

// A signed thinking block with empty text (summarized/omitted display)
// still carries the required "thinking" key on replay.
func TestBuildAnthropicRequest_EmptyThinkingTextKeepsKey(t *testing.T) {
	for _, m := range []Message{
		{Role: RoleAssistant, ThinkingBlocks: []ThinkingBlock{{Signature: "sig"}}},
	} {
		b, err := buildAnthropicRequest(&ChatRequest{Messages: []Message{{Role: RoleUser, Content: "q"}, m}}, "claude", false)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `{"type":"thinking","thinking":"","signature":"sig"}`) {
			t.Errorf("body = %s", b)
		}
	}
}
