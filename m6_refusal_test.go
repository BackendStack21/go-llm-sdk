package llm

import (
	"strings"
	"testing"
)

func m6Sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

func m6AnthropicSSE(stopDetails string) string {
	return "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\",\"usage\":{\"input_tokens\":1}}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\"" + stopDetails + "},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

func TestM6_RefusalSurfacedOnEveryFormat(t *testing.T) {
	cases := []struct {
		name       string
		format     Format
		id         string
		model      string
		stream     bool
		body       string
		wantRef    string
		wantFinish string
		wantContnt string
		wantDeltas int // number of DeltaRefusal deltas, -1 = at least one
	}{
		{"openai buffered", FormatOpenAI, "openai", "", false,
			`{"choices":[{"message":{"role":"assistant","content":null,"refusal":"no way"},"finish_reason":"stop"}]}`,
			"no way", FinishContentFilter, "", 0},
		{"openai buffered empty finish", FormatOpenAI, "openai", "", false,
			`{"choices":[{"message":{"role":"assistant","refusal":"nope"},"finish_reason":""}]}`,
			"nope", FinishContentFilter, "", 0},
		{"openai buffered length kept", FormatOpenAI, "openai", "", false,
			`{"choices":[{"message":{"role":"assistant","content":"par","refusal":"nope"},"finish_reason":"length"}]}`,
			"nope", FinishLength, "par", 0},
		{"openai buffered normal", FormatOpenAI, "openai", "", false,
			`{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`,
			"", FinishStop, "hi", 0},
		{"openai stream", FormatOpenAI, "openai", "", true,
			m6Sse(`{"choices":[{"delta":{"refusal":"no "}}]}`, `{"choices":[{"delta":{"refusal":"way"}}]}`,
				`{"choices":[{"delta":{},"finish_reason":"stop"}]}`, "[DONE]"),
			"no way", FinishContentFilter, "", -1},
		{"openai stream normal", FormatOpenAI, "openai", "", true,
			m6Sse(`{"choices":[{"delta":{"content":"hi"}}]}`, `{"choices":[{"delta":{},"finish_reason":"stop"}]}`, "[DONE]"),
			"", FinishStop, "hi", 0},
		{"responses buffered", FormatOpenAI, "openai", "gpt-5.6-luna", false,
			`{"status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"no "},{"type":"refusal","refusal":"way"}]}]}`,
			"no way", FinishContentFilter, "", 0},
		{"responses buffered normal", FormatOpenAI, "openai", "gpt-5.6-luna", false,
			`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}`,
			"", FinishStop, "hi", 0},
		{"responses stream", FormatOpenAI, "openai", "gpt-5.6-luna", true,
			m6Sse(`{"type":"response.refusal.delta","delta":"no "}`, `{"type":"response.refusal.delta","delta":"way"}`,
				`{"type":"response.refusal.done","refusal":"no way"}`,
				`{"type":"response.completed","response":{"status":"completed"}}`),
			"no way", FinishContentFilter, "", -1},
		{"anthropic buffered explanation", FormatAnthropic, "anthropic", "", false,
			`{"content":[],"stop_reason":"refusal","stop_details":{"explanation":"policy"},"usage":{"input_tokens":1,"output_tokens":1}}`,
			"policy", FinishContentFilter, "", 0},
		{"anthropic buffered no explanation", FormatAnthropic, "anthropic", "", false,
			`{"content":[],"stop_reason":"refusal","usage":{"input_tokens":1,"output_tokens":1}}`,
			"", FinishContentFilter, "", 0},
		{"anthropic buffered normal", FormatAnthropic, "anthropic", "", false,
			`{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
			"", FinishStop, "hi", 0},
		{"anthropic stream explanation", FormatAnthropic, "anthropic", "", true,
			m6AnthropicSSE(`,"stop_details":{"explanation":"policy"}`),
			"policy", FinishContentFilter, "", 1},
		{"anthropic stream no explanation", FormatAnthropic, "anthropic", "", true,
			m6AnthropicSSE(``),
			"", FinishContentFilter, "", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cc := m2Client(t, c.format, c.id, c.body, c.stream)
			if c.model != "" {
				cc.model = c.model
			}
			req := m2Req()
			res, ds := m2Run(t, cc, req, c.stream)
			if res.Refusal != c.wantRef {
				t.Errorf("Refusal = %q, want %q", res.Refusal, c.wantRef)
			}
			if res.FinishReason != c.wantFinish {
				t.Errorf("FinishReason = %q, want %q", res.FinishReason, c.wantFinish)
			}
			if res.Content != c.wantContnt {
				t.Errorf("Content = %q, want %q", res.Content, c.wantContnt)
			}
			if !c.stream {
				return
			}
			var cat strings.Builder
			n := 0
			for _, d := range ds {
				if d.Kind == DeltaRefusal {
					n++
					cat.WriteString(d.Text)
				}
			}
			if cat.String() != c.wantRef {
				t.Errorf("concatenated DeltaRefusal = %q, want %q", cat.String(), c.wantRef)
			}
			if c.wantDeltas >= 0 && n != c.wantDeltas {
				t.Errorf("DeltaRefusal count = %d, want %d", n, c.wantDeltas)
			}
			if c.wantDeltas < 0 && n < 1 {
				t.Errorf("want at least one DeltaRefusal, got %d", n)
			}
		})
	}
}

// DeltaRefusal is appended after DeltaToolArgs so existing numbers hold.
func TestM6_DeltaRefusalAppended(t *testing.T) {
	if DeltaReasoning != 0 || DeltaContent != 1 || DeltaToolArgs != 2 || DeltaRefusal != 3 {
		t.Errorf("DeltaKind numbering: %d %d %d %d", DeltaReasoning, DeltaContent, DeltaToolArgs, DeltaRefusal)
	}
}

// The refusal never replays into the next turn.
func TestM6_AssistantMessageDoesNotReplayRefusal(t *testing.T) {
	m := (&ChatResult{Content: "c", Refusal: "no"}).AssistantMessage()
	if strings.Contains(m.Content, "no") || m.Content != "c" {
		t.Errorf("content = %q", m.Content)
	}
}
