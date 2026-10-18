package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

const m2AnthropicNoArgStream = `event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"%s","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`

func m2AnthropicStreamBody(name string) string {
	return strings.Replace(m2AnthropicNoArgStream, "%s", name, 1)
}

// Bug 5: a no-argument Anthropic tool streams input:{} then an empty
// partial_json; the result must carry "{}" (as buffered does), valid JSON.
func TestM2_AnthropicNoArgToolArgumentsNormalized(t *testing.T) {
	buffered := `{"content":[{"type":"tool_use","id":"tu1","name":"f","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":2}}`
	for _, stream := range []bool{false, true} {
		body := buffered
		if stream {
			body = m2AnthropicStreamBody("f")
		}
		c := m2Client(t, FormatAnthropic, "anthropic", body, stream)
		res, ds := m2Run(t, c, m2Req(), stream)
		if len(res.ToolCalls) != 1 {
			t.Fatalf("stream=%v: calls = %+v", stream, res.ToolCalls)
		}
		args := res.ToolCalls[0].Arguments
		if args != "{}" || !json.Valid([]byte(args)) {
			t.Errorf("stream=%v: Arguments = %q, want {}", stream, args)
		}
		_ = ds
	}
}

// Bug 5 (JSON mode, native since M4 SL02): the reply text is returned
// unchanged as Content, buffered and streaming alike. (The empty-input
// tool folding to "{}" returns in SL03 behind the AnthropicToolJSON quirk.)
func TestM2_AnthropicJSONModeEmptyInputContent(t *testing.T) {
	req := &ChatRequest{
		Messages:       []Message{{Role: RoleUser, Content: "hi"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONObject},
	}
	for _, stream := range []bool{false, true} {
		body := m4TextReply(`{}`, stream)
		c := m2Client(t, FormatAnthropic, "anthropic", body, stream)
		res, _ := m2Run(t, c, req, stream)
		if res.Content != "{}" {
			t.Errorf("stream=%v: Content = %q, want {}", stream, res.Content)
		}
		if res.FinishReason != FinishStop {
			t.Errorf("stream=%v: finish = %q, want stop", stream, res.FinishReason)
		}
	}
}

// Bug 6: Gemini image-safety finish reasons are content filters, like
// SAFETY; IMAGE_OTHER and unknown reasons stay unmapped.
func TestM2_GeminiImageSafetyFinishReasons(t *testing.T) {
	cases := map[string]string{
		"IMAGE_SAFETY":             FinishContentFilter,
		"IMAGE_PROHIBITED_CONTENT": FinishContentFilter,
		"IMAGE_RECITATION":         FinishContentFilter,
		"SAFETY":                   FinishContentFilter,
		"PROHIBITED_CONTENT":       FinishContentFilter,
		"SPII":                     FinishContentFilter,
		"BLOCKLIST":                FinishContentFilter,
		"RECITATION":               FinishContentFilter,
		"IMAGE_OTHER":              "",
		"OTHER":                    "",
	}
	for in, want := range cases {
		if got := mapGeminiFinishReason(in); got != want {
			t.Errorf("mapGeminiFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestM2_GeminiImageSafetyEndToEnd(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"IMAGE_SAFETY"}]}`
	for _, stream := range []bool{false, true} {
		b := body
		if stream {
			b = "data: " + body + "\n\n"
		}
		c := m2Client(t, FormatGemini, "gemini", b, stream)
		res, _ := m2Run(t, c, &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, stream)
		if res.FinishReason != FinishContentFilter {
			t.Errorf("stream=%v: finish = %q, want content_filter", stream, res.FinishReason)
		}
	}
}
