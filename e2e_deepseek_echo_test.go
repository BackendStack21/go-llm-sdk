//go:build e2e

package llm

// Probe for the DeepSeek reasoning_content replay contract (PR #8).
//
// DeepSeek documents (guides/thinking_mode) that a request carrying tools must
// "fully pass back" reasoning_content for all previous turns, and that a
// violation returns 400 on every later request in the loop. Two questions that
// documentation does NOT settle, and which change what the right fix is:
//
//	A. Is a *present but empty* reasoning_content accepted?
//	B. Is an *omitted* reasoning_content accepted on an older assistant turn?
//
// The shipped behaviour echoes the key when empty for flagged providers, which
// is only correct if (A) holds. If (B) holds and (A) does not, the echo is the
// wrong layer and the caller must preserve real reasoning instead — the SDK
// cannot recover reasoning a caller discarded.
//
// Nothing is asserted: outcomes are reported with t.Logf so the probe is a
// measurement, never a flake. Run it deliberately, with credentials:
//
//	go test -tags e2e -run TestE2EDeepSeekEmptyReasoningEcho -timeout 5m -v .

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

// probeTool is the single tool both variants advertise, so the request shape
// differs only in the reasoning replay decision under test.
func probeTool() ToolDef {
	return ToolDef{
		Name:        "get_weather",
		Description: "Get the weather forecast for a location.",
		Parameters: json.RawMessage(
			`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
	}
}

// probeMessages is a tool loop whose SECOND assistant turn carries no reasoning
// text — exactly what DeepSeek elision produces, and what a caller that trims
// old reasoning produces. The first turn carries real reasoning, so the two
// turns differ only in the property under test.
func probeMessages() []Message {
	return []Message{
		{Role: RoleUser, Content: "Will it rain in Berlin tomorrow?"},
		{Role: RoleAssistant, Content: "Checking.", ReasoningContent: "I should look up the forecast.",
			ToolCalls: []ToolCall{{ID: "call_a", Name: "get_weather", Arguments: `{"location":"Berlin"}`}}},
		{Role: RoleTool, ToolCallID: "call_a", ToolName: "get_weather",
			Content: `{"condition":"rain","temp_c":11}`},
		// The elided / trimmed turn: no reasoning text available to replay.
		{Role: RoleAssistant, Content: "One more check.",
			ToolCalls: []ToolCall{{ID: "call_b", Name: "get_weather", Arguments: `{"location":"Berlin"}`}}},
		{Role: RoleTool, ToolCallID: "call_b", ToolName: "get_weather",
			Content: `{"condition":"rain","temp_c":11}`},
		{Role: RoleUser, Content: "So, rain or not?"},
	}
}

// probeEcho posts probeMessages with the echo quirk forced on or off and
// reports whether DeepSeek accepted the request.
func probeEcho(t *testing.T, echo bool, key, model string) error {
	t.Helper()
	sdk := New(WithProvider("deepseek",
		WithFormat(FormatOpenAI),
		WithBaseURL("https://api.deepseek.com"),
		WithAPIKey(key),
		WithQuirks(Quirks{ThinkingObject: true, EchoReasoningWithTools: echo}),
	))
	cc, err := sdk.Chat("deepseek", model)
	if err != nil {
		t.Fatalf("Chat(deepseek, %s): %v", model, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	res, err := cc.Call(ctx, &ChatRequest{
		Thinking: "enabled",
		Tools:    []ToolDef{probeTool()},
		Messages: probeMessages(),
	})
	if err != nil {
		t.Logf("EchoReasoningWithTools=%v → REJECTED: %v", echo, err)
		return err
	}
	t.Logf("EchoReasoningWithTools=%v → ACCEPTED (finish=%q, tools=%d, reasoning_chars=%d)",
		echo, res.FinishReason, len(res.ToolCalls), len(res.ReasoningContent))
	return nil
}

// TestE2EDeepSeekEmptyReasoningEcho answers questions A and B above and prints
// which fix layer the answers imply.
func TestE2EDeepSeekEmptyReasoningEcho(t *testing.T) {
	tg := e2eTargets[0] // deepseek
	key := e2eEnvKey(t, tg.keyEnv)
	model := tg.chatModel()

	emptyPresent := probeEcho(t, true, key, model) // A: key present, value ""
	omitted := probeEcho(t, false, key, model)     // B: key absent (pre-PR shape)

	switch {
	case emptyPresent == nil && omitted == nil:
		t.Log("VERDICT: DeepSeek accepts both a present-empty and an omitted key on older assistant turns. " +
			"The echo is harmless but not load-bearing here — reproduce the reported 400 with the caller's " +
			"real history before treating this as the fix.")
	case emptyPresent == nil && omitted != nil:
		t.Log("VERDICT: present-and-empty is accepted where omission is rejected — the shipped echo is the " +
			"correct fix. Ship it.")
	case emptyPresent != nil && omitted == nil:
		t.Log("VERDICT: present-and-empty is REJECTED where omission is accepted — the echo is the WRONG " +
			"layer. Keep the per-message rule (echo only real reasoning) and fix the caller instead: it must " +
			"preserve reasoning on every assistant turn of a tool loop (odek: keepRecentReasoning / " +
			"stripOldReasoning).")
	default:
		t.Log("VERDICT: both shapes rejected — the violation is elsewhere in the replay shape. Capture the " +
			"exact 400 body and the request history from the failing session.")
	}
}
