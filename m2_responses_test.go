package llm

import "testing"

// Bug 3: an incomplete response whose reason is not max_output_tokens or
// content_filter is an unmapped stop reason: "" (invariant 1), never
// tool_calls or length.
func TestM2_ResponsesIncompleteUnknownReasonIsEmpty(t *testing.T) {
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
		{"unknown reason, tools", inc("something_new"), true, ""},
		{"unknown reason, no tools", inc("something_new"), false, ""},
		{"max output stays length", inc("max_output_tokens"), true, FinishLength},
		{"content filter stays", inc("content_filter"), false, FinishContentFilter},
	}
	for _, c := range cases {
		if got := responsesFinishReason(c.r, c.hasTools); got != c.want {
			t.Errorf("%s: %q, want %q", c.name, got, c.want)
		}
	}
}

// Bug 3 through the buffered parser and the stream mapper.
func TestM2_ResponsesIncompleteUnknownReasonParsers(t *testing.T) {
	res, err := parseResponsesAPI([]byte(`{"status":"incomplete","incomplete_details":{"reason":"something_new"},"output":[{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != "" {
		t.Errorf("buffered finish = %q, want \"\"", res.FinishReason)
	}
	acc := newStreamAccum()
	for _, e := range []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c1","name":"f"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{}"}`,
		`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"something_new"}}}`,
	} {
		if _, _, err := mapResponsesStreamEvent([]byte(e), acc); err != nil {
			t.Fatal(err)
		}
	}
	if got := acc.result().FinishReason; got != "" {
		t.Errorf("streaming finish = %q, want \"\"", got)
	}
}
