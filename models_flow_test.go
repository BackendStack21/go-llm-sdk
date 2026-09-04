package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

// ── buffered dispatch per wire format (parseResponse arms) ───────────────

func TestCallBufferedPerFormatDispatch(t *testing.T) {
	cases := []struct {
		name   string
		format Format
		quirks Quirks
		body   string
		check  func(*testing.T, *ChatResult)
	}{
		{
			name:   "openai",
			format: FormatOpenAI,
			body:   `{"choices":[{"message":{"role":"assistant","content":"hey","reasoning_content":"thought"},"finish_reason":"tool_calls","tool_calls":[]}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`,
			check: func(t *testing.T, r *ChatResult) {
				if r.Content != "hey" || r.ReasoningContent != "thought" || r.FinishReason != FinishToolCalls {
					t.Fatalf("openai parse = %+v", r)
				}
			},
		},
		{
			name:   "anthropic",
			format: FormatAnthropic,
			body:   `{"content":[{"type":"text","text":"A"}],"stop_reason":"max_tokens","usage":{"input_tokens":4,"output_tokens":6}}`,
			check: func(t *testing.T, r *ChatResult) {
				if r.Content != "A" || r.FinishReason != FinishLength || r.Usage.CompletionTokens != 6 {
					t.Fatalf("anthropic parse = %+v", r)
				}
			},
		},
		{
			name:   "gemini",
			format: FormatGemini,
			body:   `{"candidates":[{"content":{"parts":[{"text":"G"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2}}`,
			check: func(t *testing.T, r *ChatResult) {
				if r.Content != "G" || r.FinishReason != FinishStop || r.Usage.PromptTokens != 1 {
					t.Fatalf("gemini parse = %+v", r)
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tc.body))
			})
			defer srv.Close()
			cc := newTestClient(t, ProviderConfig{ID: "x", Format: tc.format, BaseURL: srv.URL, APIKey: "k", Quirks: tc.quirks}, srv)
			res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
			if err != nil {
				t.Fatal(err)
			}
			tc.check(t, res)
		})
	}
}

// Parse failures must surface as descriptive errors, not empty results.
func TestBufferedParseErrorPaths(t *testing.T) {
	cases := []struct {
		name   string
		format Format
		body   string
		want   string
	}{
		{"openai provider error", FormatOpenAI, `{"error":{"message":"boom"}}`, "provider error"},
		{"openai no choices", FormatOpenAI, `{"choices":[]}`, "no choices"},
		{"gemini no candidates", FormatGemini, `{"candidates":[]}`, "no candidates"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tc.body))
			})
			defer srv.Close()
			cc := newTestClient(t, ProviderConfig{ID: "x", Format: tc.format, BaseURL: srv.URL, APIKey: "k"}, srv)
			_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want containing %q", err, tc.want)
			}
		})
	}
}

// ── retry state machine arms ─────────────────────────────────────────────

// A transport-level failure (dropped connection) retries and then succeeds.
func TestCallTransportErrorRetries(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n <= 2 {
			panic(http.ErrAbortHandler) // drop the connection
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.Content != "ok" || n != 3 {
		t.Fatalf("res=%q attempts=%d", res.Content, n)
	}
}

// A definitive 4xx after earlier 429s surfaces that 4xx (the request is
// definitively broken — retrying the rate limit was pointless).
func TestCallDefinitiveFailureAfter429(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(429)
			fmt.Fprint(w, `{"error":{"message":"slow"}}`)
			return
		}
		w.WriteHeader(401)
		fmt.Fprint(w, `{"error":{"message":"bad key"}}`)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	var ae *APIError
	if !errors.As(err, &ae) || ae.Status != 401 {
		t.Fatalf("err = %v, want the definitive 401", err)
	}
	if n != 2 {
		t.Errorf("attempts = %d, want 2", n)
	}
}

// A streamed request answered with a non-SSE body learns the buffered
// fallback mid-stream and completes on the buffered path.
func TestCallStreamLearnsBufferedFromNonSSEBody(t *testing.T) {
	var n int
	srv := httptestNewServer(func(w http.ResponseWriter, r *http.Request) {
		n++
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"buffered"},"finish_reason":"stop"}]}`)
	})
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil {
		t.Fatalf("CallStream: %v", err)
	}
	if res.Content != "buffered" {
		t.Errorf("res = %q", res.Content)
	}
	if n != 2 {
		t.Errorf("requests = %d, want 2 (learn, then buffered retry)", n)
	}
	// The learned fallback must short-circuit the NEXT CallStream of the
	// same client straight onto the buffered path.
	res2, err := cc.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error {
		return nil
	})
	if err != nil || res2.Content != "buffered" {
		t.Fatalf("second stream: res=%q err=%v", res2.Content, err)
	}
	if n != 3 {
		t.Errorf("requests = %d, want 3 (entry fast-path, no re-learn)", n)
	}
}

// ── provider error-body parsing per format ───────────────────────────────

func TestHTTPErrorPerFormat(t *testing.T) {
	cases := []struct {
		name    string
		format  Format
		body    string
		message string
		code    string
	}{
		{"anthropic", FormatAnthropic, `{"error":{"type":"invalid_request_error","message":"am"}}`, "am", "invalid_request_error"},
		{"gemini", FormatGemini, `{"error":{"message":"gm","status":"INVALID_ARGUMENT"}}`, "gm", "INVALID_ARGUMENT"},
		{"openai numeric code", FormatOpenAI, `{"error":{"message":"om","code":42}}`, "om", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pc := newProviderClient(ProviderConfig{ID: "p", Format: tc.format}, nil, nil)
			e := pc.httpError(400, []byte(tc.body))
			if e.Message != tc.message || e.Code != tc.code {
				t.Fatalf("APIError = %+v, want message %q code %q", e, tc.message, tc.code)
			}
		})
	}
}

// ── auth headers per format ──────────────────────────────────────────────

func TestSetAuthHeaders(t *testing.T) {
	anthropic := newProviderClient(ProviderConfig{ID: "a", Format: FormatAnthropic, APIKey: "ak"}, nil, nil)
	h := http.Header{}
	anthropic.setAuthHeaders(h)
	if h.Get("x-api-key") != "ak" || h.Get("anthropic-version") != "2023-06-01" {
		t.Errorf("anthropic headers = %v", h)
	}
	anthropicCustom := newProviderClient(ProviderConfig{ID: "a", Format: FormatAnthropic, APIKey: "ak", Quirks: Quirks{AnthropicVersion: "2044-01-01"}}, nil, nil)
	h = http.Header{}
	anthropicCustom.setAuthHeaders(h)
	if h.Get("anthropic-version") != "2044-01-01" {
		t.Errorf("custom version = %q", h.Get("anthropic-version"))
	}
	gemini := newProviderClient(ProviderConfig{ID: "g", Format: FormatGemini, APIKey: "gk"}, nil, nil)
	h = http.Header{}
	gemini.setAuthHeaders(h)
	if h.Get("x-goog-api-key") != "gk" {
		t.Errorf("gemini headers = %v", h)
	}
	bearer := newProviderClient(ProviderConfig{ID: "o", Format: FormatOpenAI, APIKey: "ok"}, nil, nil)
	h = http.Header{}
	bearer.setAuthHeaders(h)
	if h.Get("Authorization") != "Bearer ok" {
		t.Errorf("bearer = %q", h.Get("Authorization"))
	}
}

// ── canonical finish-reason vocabulary ───────────────────────────────────

func TestFinishReasonTables(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"stop", FinishStop}, {"length", FinishLength},
		{"tool_calls", FinishToolCalls}, {"function_call", FinishToolCalls},
		{"content_filter", FinishContentFilter}, {"", ""},
		{"mystery_reason", ""}, // non-canonical values never leak
	} {
		if got := mapOpenAIFinishReason(tc.in); got != tc.want {
			t.Errorf("mapOpenAIFinishReason(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, tc := range []struct{ in, want string }{
		{"end_turn", FinishStop}, {"stop_sequence", FinishStop},
		{"max_tokens", FinishLength}, {"tool_use", FinishToolCalls},
		{"refusal", FinishContentFilter}, {"", ""}, {"pause_turn", ""},
	} {
		if got := mapAnthropicStopReason(tc.in); got != tc.want {
			t.Errorf("mapAnthropicStopReason(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	for _, tc := range []struct{ in, want string }{
		{"STOP", FinishStop}, {"MAX_TOKENS", FinishLength},
		{"SAFETY", FinishContentFilter}, {"RECITATION", FinishContentFilter},
		{"BLOCKLIST", FinishContentFilter}, {"", ""}, {"OTHER", ""},
	} {
		if got := mapGeminiFinishReason(tc.in); got != tc.want {
			t.Errorf("mapGeminiFinishReason(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// ── SSE edges ────────────────────────────────────────────────────────────

// A single event assembled from multiple data lines may reach the 4 MiB
// event cap; oversized events are rejected with errSSEOversized.
func TestParseSSEStreamOversizedEvent(t *testing.T) {
	chunk := strings.Repeat("x", 700*1024) // 700 KiB per line, under the line cap
	var b strings.Builder
	for i := 0; i < 7; i++ { // 4.9 MiB accumulated into ONE event
		b.WriteString("data: " + chunk + "\n")
	}
	b.WriteString("\n")
	done := make(chan struct{})
	ch := make(chan sseItem, 2)
	go parseSSEStream(strings.NewReader(b.String()), ch, done)
	it := <-ch
	if it.kind != sseErr || !errors.Is(it.err, errSSEOversized) {
		t.Fatalf("oversized event: kind %v err %v", it.kind, it.err)
	}
}

// data payload whitespace: "data:x" keeps x, "data: x" strips one space.
func TestSSEPayloadTrimming(t *testing.T) {
	ch := make(chan sseItem, 4)
	done := make(chan struct{})
	go parseSSEStream(strings.NewReader("data:nospace\n\ndata: withspace\n\n"), ch, done)
	first := <-ch
	if string(first.data) != "nospace" {
		t.Fatalf("first = %q", first.data)
	}
	second := <-ch
	if string(second.data) != "withspace" {
		t.Fatalf("second = %q", second.data)
	}
}

// ── backoff bounds ───────────────────────────────────────────────────────

func TestBackoffDelayCap(t *testing.T) {
	old := backoffUnit
	backoffUnit = 10 * time.Second
	t.Cleanup(func() { backoffUnit = old })
	for attempt := 1; attempt <= 8; attempt++ {
		if d := backoffDelay(attempt); d > maxRetryBackoff {
			t.Fatalf("backoffDelay(%d) = %v exceeds cap", attempt, d)
		}
	}
	if d := backoffDelay(5); d != maxRetryBackoff {
		t.Errorf("backoffDelay(5) = %v, want capped %v", d, maxRetryBackoff)
	}
}
