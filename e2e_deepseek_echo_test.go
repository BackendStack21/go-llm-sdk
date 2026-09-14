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
// Outcomes are reported with t.Logf, never asserted, so a provider hiccup is
// reported as INCONCLUSIVE rather than dressed up as a contract violation:
// only a 400 whose body names reasoning_content counts as REJECTED. The probe
// covers thinking enabled, disabled and unset, because the shipped echo does
// not depend on the thinking setting and the widest new wire state is the
// non-thinking one (an odek tool loop on a DeepSeek entry trims old reasoning
// and therefore sends empty values with or without thinking mode).
//
// Run deliberately, with credentials:
//
//	go test -tags e2e -run TestE2EDeepSeekEmptyReasoningEcho -timeout 10m -v .

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
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
//
// continuation selects the request's tail shape, which the live evidence shows
// matters: a request that ends with a TOOL RESULT continues the tool loop and is
// the shape odek actually sends when it 400s, while one that ends with a USER
// message is terminal. An earlier version of this probe only tested the terminal
// shape, where both the echoed and the omitted key were accepted — a false
// negative that this second dimension exists to close.
func probeMessages(continuation bool) []Message {
	msgs := []Message{
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
	}
	if continuation {
		return msgs // ends with a tool result: the loop continues
	}
	return append(msgs, Message{Role: RoleUser, Content: "So, rain or not?"})
}

// probeOutcome is one measurement: whether a shape was accepted, or why the
// question could not be answered.
type probeOutcome struct {
	accepted     bool
	inconclusive bool
	detail       string
}

// probeEcho posts probeMessages with the echo quirk forced on or off for one
// shape and classifies the result.
func probeEcho(t *testing.T, echo, continuation bool, key, model string) probeOutcome {
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
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	res, err := cc.Call(ctx, &ChatRequest{
		Thinking: "enabled",
		Tools:    []ToolDef{probeTool()},
		Messages: probeMessages(continuation),
	})
	label := fmt.Sprintf("echo=%-5v shape=%-12s", echo, shapeName(continuation))
	if err == nil {
		t.Logf("%s -> ACCEPTED (finish=%q, tools=%d, reasoning_chars=%d)",
			label, res.FinishReason, len(res.ToolCalls), len(res.ReasoningContent))
		return probeOutcome{accepted: true}
	}

	// Only a 400 that names reasoning_content is a verdict about the contract.
	// A rate limit, an outage, a timeout or a bad key must not be reported as
	// "DeepSeek rejected this shape".
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Logf("%s -> INCONCLUSIVE (transport/timeout, not a contract answer): %v", label, err)
		return probeOutcome{inconclusive: true, detail: err.Error()}
	}
	if apiErr.Status != 400 || !strings.Contains(apiErr.Message, "reasoning_content") {
		t.Logf("%s -> INCONCLUSIVE (HTTP %d, not a reasoning_content rejection): %v",
			label, apiErr.Status, apiErr.Message)
		return probeOutcome{inconclusive: true, detail: apiErr.Message}
	}
	t.Logf("%s -> REJECTED: %v", label, err)
	return probeOutcome{detail: err.Error()}
}

func shapeName(continuation bool) string {
	if continuation {
		return "continuation"
	}
	return "terminal"
}

// TestE2EDeepSeekEmptyReasoningEcho answers questions A and B and prints which
// fix layer the answers imply. It asserts nothing: it is a measurement.
func TestE2EDeepSeekEmptyReasoningEcho(t *testing.T) {
	tg := e2eTargets[0] // deepseek
	key := e2eEnvKey(t, tg.keyEnv)
	model := tg.chatModel()

	// A: key present with an empty value. B: key absent (the pre-PR shape).
	// Both in the tool-loop continuation shape odek actually sends when it 400s,
	// and in the terminal shape as a control.
	for _, continuation := range []bool{true, false} {
		t.Logf("--- shape=%s ---", shapeName(continuation))
		present := probeEcho(t, true, continuation, key, model)
		omitted := probeEcho(t, false, continuation, key, model)
		verdict(t, shapeName(continuation), present, omitted)
	}
}

// verdict reports which fix layer the two measurements imply.
func verdict(t *testing.T, shape string, present, omitted probeOutcome) {
	t.Helper()
	switch {
	case present.inconclusive || omitted.inconclusive:
		t.Logf("VERDICT [%s]: INCONCLUSIVE — a shape hit a non-contract error (rate limit, outage, "+
			"timeout). Re-run before drawing any conclusion.", shape)
	case present.accepted && omitted.accepted:
		t.Logf("VERDICT [%s]: DeepSeek accepts both a present-empty and an omitted key here. The echo is "+
			"harmless but not load-bearing for this shape — reproduce the reported 400 with the caller's "+
			"real history before treating the echo as the fix.", shape)
	case present.accepted && !omitted.accepted:
		t.Logf("VERDICT [%s]: present-and-empty is accepted where omission is rejected — the shipped "+
			"echo is the correct layer. Ship it.", shape)
	case !present.accepted && omitted.accepted:
		t.Logf("VERDICT [%s]: present-and-empty is REJECTED where omission is accepted — the echo is the "+
			"WRONG layer. Keep the per-message rule (echo only real reasoning) and fix the caller.", shape)
	default:
		t.Logf("VERDICT [%s]: both rejected — the violation is elsewhere in the replay shape. Capture the "+
			"exact 400 body and the request history from the failing session.", shape)
	}
}
