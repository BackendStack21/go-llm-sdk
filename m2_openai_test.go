package llm

import "testing"

// Bug 1: a tool-call turn must finish tool_calls even when an OpenAI-format
// provider reports finish_reason "stop" (named tool_choice, gateways).
func TestM2_OpenAIToolTurnWithStopFinishIsToolCalls(t *testing.T) {
	buffered := `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"stop"}]}`
	streamed := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{}"}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n" + "data: [DONE]\n\n"
	for _, stream := range []bool{false, true} {
		body := buffered
		if stream {
			body = streamed
		}
		c := m2Client(t, FormatOpenAI, "openai", body, stream)
		res, _ := m2Run(t, c, m2Req(), stream)
		if len(res.ToolCalls) != 1 {
			t.Fatalf("stream=%v: tool calls = %+v", stream, res.ToolCalls)
		}
		if res.FinishReason != FinishToolCalls {
			t.Errorf("stream=%v: finish = %q, want %q", stream, res.FinishReason, FinishToolCalls)
		}
	}
}

// Bug 1 control: a plain stop with no tool calls stays stop.
func TestM2_OpenAIPlainStopStaysStop(t *testing.T) {
	c := m2Client(t, FormatOpenAI, "openai", `{"choices":[{"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}]}`, false)
	res, _ := m2Run(t, c, m2Req(), false)
	if res.FinishReason != FinishStop {
		t.Errorf("finish = %q, want stop", res.FinishReason)
	}
}

// Bug 2: streamed tool-call deltas without an index but with distinct ids
// are separate calls; ToolIndex is the position in ChatResult.ToolCalls.
func TestM2_OpenAIStreamToolCallsWithoutIndex(t *testing.T) {
	body := `data: {"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"f","arguments":"{\"x\":1}"}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"tool_calls":[{"id":"b","function":{"name":"g","arguments":"{\"y\":2}"}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" + "data: [DONE]\n\n"
	c := m2Client(t, FormatOpenAI, "openai", body, true)
	res, ds := m2Run(t, c, m2Req(), true)
	want := []ToolCall{{ID: "a", Name: "f", Arguments: `{"x":1}`}, {ID: "b", Name: "g", Arguments: `{"y":2}`}}
	if len(res.ToolCalls) != len(want) {
		t.Fatalf("tool calls = %+v, want %+v", res.ToolCalls, want)
	}
	for i, w := range want {
		if res.ToolCalls[i] != w {
			t.Errorf("call %d = %+v, want %+v", i, res.ToolCalls[i], w)
		}
	}
	var idx []int
	for _, d := range ds {
		if d.Kind == DeltaToolArgs {
			idx = append(idx, d.ToolIndex)
		}
	}
	if len(idx) != 2 || idx[0] != 0 || idx[1] != 1 {
		t.Errorf("ToolIndex sequence = %v, want [0 1]", idx)
	}
}

// Bug 2 control: continuation chunks that omit both index and id (or repeat
// the same id) still extend the current call.
func TestM2_OpenAIStreamNoIndexContinuationStaysOneCall(t *testing.T) {
	body := `data: {"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"f","arguments":"{\"x\""}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":":1}"}}]}}]}` + "\n\n" +
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" + "data: [DONE]\n\n"
	c := m2Client(t, FormatOpenAI, "openai", body, true)
	res, _ := m2Run(t, c, m2Req(), true)
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != `{"x":1}` {
		t.Errorf("tool calls = %+v", res.ToolCalls)
	}
}
