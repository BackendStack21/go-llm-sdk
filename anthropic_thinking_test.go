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

// Anthropic requires max_tokens > thinking.budget_tokens. An unset
// MaxTokens must leave room for the budget plus visible output.
func TestBuildAnthropicRequest_ThinkingDefaultMaxTokensExceedsBudget(t *testing.T) {
	for _, level := range []string{"enabled", "low", "medium", "high", "max"} {
		m := decodeAnthropicBody(t, &ChatRequest{Thinking: level, Messages: []Message{{Role: RoleUser, Content: "x"}}})
		budget := m["thinking"].(map[string]any)["budget_tokens"].(float64)
		maxTok := m["max_tokens"].(float64)
		if maxTok <= budget {
			t.Errorf("%s: max_tokens %v must exceed budget_tokens %v", level, maxTok, budget)
		}
		if maxTok != budget+anthropicDefaultMaxTokens {
			t.Errorf("%s: max_tokens = %v, want budget+%d", level, maxTok, anthropicDefaultMaxTokens)
		}
	}
}

// A preset level larger than an explicit MaxTokens is clamped below it.
func TestBuildAnthropicRequest_ThinkingPresetClampedBelowMaxTokens(t *testing.T) {
	m := decodeAnthropicBody(t, &ChatRequest{Thinking: "high", MaxTokens: 4000, Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if m["max_tokens"].(float64) != 4000 {
		t.Errorf("explicit max_tokens must be honored, got %v", m["max_tokens"])
	}
	// Half the cap stays for visible output (never a one-token answer).
	if b := m["thinking"].(map[string]any)["budget_tokens"].(float64); b != 2000 {
		t.Errorf("budget_tokens = %v, want 2000 (half of max_tokens)", b)
	}
	m = decodeAnthropicBody(t, &ChatRequest{Thinking: "high", MaxTokens: 1500, Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if b := m["thinking"].(map[string]any)["budget_tokens"].(float64); b != 1024 {
		t.Errorf("small cap: budget_tokens = %v, want the 1024 minimum", b)
	}
	m = decodeAnthropicBody(t, &ChatRequest{Thinking: "low", MaxTokens: 1500, Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if b := m["thinking"].(map[string]any)["budget_tokens"].(float64); b != 1024 {
		t.Errorf("preset below cap is kept: budget_tokens = %v", b)
	}
}

// Contradictory explicit settings fail fast instead of a guaranteed 400.
func TestBuildAnthropicRequest_ThinkingBudgetConflicts(t *testing.T) {
	cases := []ChatRequest{
		{Thinking: "enabled", ThinkingBudget: 8000, MaxTokens: 8000},
		{Thinking: "enabled", ThinkingBudget: 2000, MaxTokens: 1500},
		{Thinking: "low", MaxTokens: 1024}, // cannot clamp below the 1024 minimum
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
	m := decodeAnthropicBody(t, &ChatRequest{Thinking: "low", Temperature: 0.3, TopP: 0.5, Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if _, ok := m["temperature"]; ok {
		t.Error("temperature must be omitted with thinking")
	}
	if _, ok := m["top_p"]; ok {
		t.Error("top_p must be omitted with thinking")
	}
	m = decodeAnthropicBody(t, &ChatRequest{Temperature: 0.3, Messages: []Message{{Role: RoleUser, Content: "x"}}})
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
