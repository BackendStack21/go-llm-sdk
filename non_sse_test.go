package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCallStream_NonSSEPreservesFirstGeneration(t *testing.T) {
	for _, responses := range []bool{false, true} {
		t.Run(fmt.Sprint(responses), func(t *testing.T) {
			requests := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests++
				var request map[string]any
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				stream, _ := request["stream"].(bool)
				if stream != (requests == 1) {
					t.Errorf("request %d stream=%v", requests, stream)
				}
				w.Header().Set("Content-Type", "application/json")
				if responses {
					fmt.Fprintf(w, `{"status":"completed","output":[{"type":"reasoning","encrypted_content":"signature","summary":[{"type":"summary_text","text":"Plan"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Generation %d"}]},{"type":"function_call","call_id":"call","name":"inspect","arguments":"{}"}],"usage":{"input_tokens":12,"output_tokens":3}}`, requests)
				} else {
					fmt.Fprintf(w, `{"choices":[{"message":{"content":"Generation %d","reasoning_content":"Plan","tool_calls":[{"id":"call","type":"function","function":{"name":"inspect","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":3}}`, requests)
				}
			}))
			defer srv.Close()
			model := "gpt-4o"
			if responses {
				model = "gpt-5.6-luna"
			}
			cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
			cc.model = model
			req := &ChatRequest{Model: model, Messages: []Message{{Role: RoleUser, Content: "hi"}}, Tools: []ToolDef{{Name: "inspect", Parameters: json.RawMessage(`{"type":"object"}`)}}}
			for i := 1; i <= 2; i++ {
				res, err := cc.CallStream(context.Background(), req, func(Delta) error { t.Error("buffered body emitted a delta"); return nil })
				if err != nil {
					t.Fatal(err)
				}
				if res.Content != fmt.Sprintf("Generation %d", i) || requests != i {
					t.Fatalf("first generation discarded: response=%+v requests=%d", res, requests)
				}
				if res.ReasoningContent != "Plan" || len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "inspect" || res.FinishReason != FinishToolCalls || res.Usage.PromptTokens != 12 || res.Usage.CompletionTokens != 3 {
					t.Fatalf("response metadata lost: %+v", res)
				}
				if responses && res.ThinkingSignature != "signature" {
					t.Errorf("signature lost: %+v", res)
				}
			}
		})
	}
}

type nonSSETransport struct{ body io.ReadCloser }

func (tr nonSSETransport) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: tr.body}, nil
}

type nonSSELargeReader struct{ bytes int }

func (r *nonSSELargeReader) Read(p []byte) (int, error) { r.bytes += len(p); return len(p), nil }

func TestAttemptStream_NonSSEBodyBounds(t *testing.T) {
	large := &nonSSELargeReader{}
	for _, tc := range []struct {
		name      string
		reader    io.Reader
		errorText string
	}{
		{"read_failure", errReader{}, "socket gone"},
		{"oversized", large, "response exceeds"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &http.Client{Transport: nonSSETransport{body: io.NopCloser(tc.reader)}}
			pc := newProviderClient(ProviderConfig{Format: FormatOpenAI}, client, client)
			out := pc.attemptStream(context.Background(), "http://provider.test/chat/completions", []byte(`{}`), mapOpenAIStreamEvent, func(Delta) error { t.Error("unexpected delta"); return nil }, false)
			if !out.bufferedResponse || out.result != nil || out.err == nil || !strings.Contains(out.err.Error(), tc.errorText) {
				t.Fatalf("outcome=%+v", out)
			}
			if !pc.learn.forceBuffered.Load() {
				t.Error("buffered mode not learned")
			}
		})
	}
	if large.bytes != maxResponseSize+1 {
		t.Errorf("read=%d, bound=%d", large.bytes, maxResponseSize+1)
	}
}

func TestCallStream_NonSSEMalformedBody(t *testing.T) {
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":`)
	}))
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "parse") || res != nil || requests != 1 {
		t.Fatalf("res=%+v err=%v requests=%d", res, err, requests)
	}
}
