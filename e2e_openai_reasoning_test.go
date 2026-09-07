//go:build e2e

package llm

import (
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// OpenAI reasoning arms. OPENAI_API_KEY comes from env or the repo .env
// (never logged). Default model is a reasoning-capable one; override with
// OPENAI_E2E_MODEL. Per house rules: probe softly — assert only what the
// SDK guarantees (call success, parsing, canonical finish). Whether the
// provider returns reasoning *text* or reasoning *tokens* is server-side
// behavior, not an SDK contract.

// e2eOpenAIChat builds a chat client on the built-in openai registry entry
// (production discovery path: OPENAI_API_KEY via FromEnv).
func e2eOpenAIChat(t *testing.T) *ChatClient {
	t.Helper()
	e2eEnvKey(t, "OPENAI_API_KEY")
	sdk := New(FromEnv())
	cc, err := sdk.Chat("openai", openaiE2EModel())
	if err != nil {
		t.Fatalf("Chat(openai, %s): %v", openaiE2EModel(), err)
	}
	return cc
}

func openaiE2EModel() string {
	if v := strings.TrimSpace(os.Getenv("OPENAI_E2E_MODEL")); v != "" {
		return v
	}
	return "gpt-5-mini"
}

// OpenAI reasoning, buffered: Thinking=medium must produce a successful
// call with canonical finish; reasoning must be observable somewhere —
// reasoning_content text or Usage.ReasoningTokens.
func TestE2EOpenAIReasoningBuffered(t *testing.T) {
	cc := e2eOpenAIChat(t)
	res, err := cc.Call(e2eCtx(t, 180*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "A clock shows 3:15. What is the angle in degrees between the hour and minute hands? Work it out, then answer with the number only."}},
		Thinking:  "medium",
		MaxTokens: 2000,
	})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.FinishReason != FinishStop && res.FinishReason != FinishLength {
		t.Errorf("finish = %q, want stop or length", res.FinishReason)
	}
	if res.ReasoningContent == "" && res.Usage.ReasoningTokens == 0 {
		t.Errorf("no reasoning observable: ReasoningContent=%q ReasoningTokens=%d",
			res.ReasoningContent, res.Usage.ReasoningTokens)
	}
	if res.ReasoningContent != "" {
		t.Logf("reasoning text captured (%d chars)", len(res.ReasoningContent))
	}
	if res.Usage.ReasoningTokens > 0 {
		t.Logf("reasoning tokens: %d", res.Usage.ReasoningTokens)
	}
}

// OpenAI reasoning, streaming: reasoning deltas (DeltaReasoning) and/or the
// usage chunk's reasoning tokens must be observable; canonical finish.
func TestE2EOpenAIReasoningStreaming(t *testing.T) {
	cc := e2eOpenAIChat(t)
	var sawReasoningDeltas bool
	res, err := cc.CallStream(e2eCtx(t, 180*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "A water lily patch doubles in size every day. It covers the whole lake on day 48. On which day was it half covered? Answer with the day number only."}},
		Thinking:  "medium",
		MaxTokens: 2000,
	}, func(d Delta) error {
		if d.Kind == DeltaReasoning && d.Text != "" {
			sawReasoningDeltas = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.FinishReason != FinishStop && res.FinishReason != FinishLength {
		t.Errorf("finish = %q, want stop or length", res.FinishReason)
	}
	if res.Usage.ReasoningTokens == 0 && !sawReasoningDeltas {
		// Provider-side: gpt-5-mini sometimes skips reasoning entirely on
		// short prompts. Capture paths are unit-covered (openai_test.go:
		// stream usage details + reasoning deltas); live is a probe.
		t.Logf("no reasoning observable on stream (server-side elision): deltas=%v ReasoningTokens=%d",
			sawReasoningDeltas, res.Usage.ReasoningTokens)
	}
	if sawReasoningDeltas {
		t.Log("reasoning deltas captured")
	}
	t.Logf("usage: %+v content=%q finish=%q", res.Usage, res.Content, res.FinishReason)
	if res.Usage.ReasoningTokens > 0 {
		t.Logf("reasoning tokens: %d", res.Usage.ReasoningTokens)
	}
}

// Thinking=disabled on a reasoning model must still succeed with a clean
// call (reasoning_effort omitted / none path), proving the effort control
// round-trips both ways.
func TestE2EOpenAIReasoningDisabled(t *testing.T) {
	cc := e2eOpenAIChat(t)
	res, err := cc.Call(e2eCtx(t, 120*time.Second), &ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "Reply with exactly: OK"}},
		Thinking:  "disabled",
		MaxTokens: 500,
	})
	if err != nil {
		var ae *APIError
		if errors.As(err, &ae) && ae.Status == http.StatusBadRequest {
			t.Fatalf("disabled thinking rejected by provider: %v (SDK must omit effort on disabled)", err)
		}
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(strings.ToUpper(res.Content), "OK") {
		t.Errorf("content = %q, want it to contain OK", res.Content)
	}
}
