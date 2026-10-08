package llm

import "testing"

// Delta.ToolIndex is the call's position in ChatResult.ToolCalls on every
// format, never a provider content-block or output-item index.
func TestDeltaToolIndex_IsToolCallPosition(t *testing.T) {
	type step struct {
		mapper streamEventMapper
		events []string
	}
	cases := map[string]step{
		"anthropic": {mapAnthropicStreamEvent, []string{
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"text"}}`,
			`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"a","name":"f"}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
			`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"b","name":"g"}}`,
			`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		}},
		"responses": {mapResponsesStreamEvent, []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning"}}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"a","name":"f"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{}"}`,
			`{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","call_id":"b","name":"g"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":2,"delta":"{}"}`,
		}},
		"openai": {mapOpenAIStreamEvent, []string{
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"f","arguments":"{}"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"g","arguments":"{}"}}]}}]}`,
		}},
	}
	for name, tc := range cases {
		acc := newStreamAccum()
		seen := map[string]int{}
		for _, e := range tc.events {
			ds, _, err := tc.mapper([]byte(e), acc)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			for _, d := range ds {
				if d.Kind == DeltaToolArgs {
					seen[d.ToolID] = d.ToolIndex
				}
			}
		}
		res := acc.result()
		if len(res.ToolCalls) != 2 {
			t.Fatalf("%s: calls = %+v", name, res.ToolCalls)
		}
		for pos, call := range res.ToolCalls {
			if got, ok := seen[call.ID]; !ok || got != pos {
				t.Errorf("%s: delta ToolIndex for %s = %d, want position %d", name, call.ID, got, pos)
			}
		}
	}
}
