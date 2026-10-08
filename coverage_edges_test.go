package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// errBody is a response body whose read fails mid-way.
type errBody struct{}

func (errBody) Read([]byte) (int, error) { return 0, errors.New("connection reset") }
func (errBody) Close() error             { return nil }

func bodyErrTransport() http.RoundTripper {
	return rtFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: errBody{}}, nil
	})
}

// A body that dies mid-read is a transport failure on every buffered path.
func TestBufferedBodyReadErrors(t *testing.T) {
	fastBackoff(t)
	sdk := New(WithTransport(bodyErrTransport()), WithRetryPolicy(RetryPolicy{MaxAttempts: 1}),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://x")))
	ctx := context.Background()
	c, _ := sdk.Chat("openai", "m")
	if _, err := c.Call(ctx, userMsg()); err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Errorf("Call: %v", err)
	}
	p, _ := sdk.Provider("openai")
	if _, err := p.ListModels(ctx); err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Errorf("ListModels: %v", err)
	}
	if _, err := sdk.Speak(ctx, "openai", "m", SpeakRequest{Text: "x", Voice: "v"}); err == nil {
		t.Error("Speak: want error")
	}
	if _, err := sdk.Transcribe(ctx, "openai", "m", TranscribeRequest{Audio: []byte("a")}); err == nil {
		t.Error("Transcribe: want error")
	}
}

// A learn-once trigger on the final allowed attempt surfaces its cause.
func TestLearnOnFinalAttempt(t *testing.T) {
	srv, n := countingServer(t, http.StatusBadRequest, nil)
	srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		w.WriteHeader(400)
		_, _ = fmt.Fprint(w, `{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-5.6 in /v1/chat/completions. Please use /v1/responses instead."}}`)
	})
	for _, stream := range []bool{false, true} {
		sdk := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 1}), WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
		c, _ := sdk.Chat("openai", "gpt-5.5")
		req := toolReq(nil)
		var err error
		if stream {
			_, err = c.CallStream(context.Background(), req, func(Delta) error { return nil })
		} else {
			_, err = c.Call(context.Background(), req)
		}
		var ae *APIError
		if !errors.As(err, &ae) || ae.Status != 400 {
			t.Errorf("stream=%v: err = %v", stream, err)
		}
	}
}

// withRetry: a definitive error after 429s is returned as is; a final
// retryable error after 429s keeps the rate-limit signal.
func TestWithRetry_AfterRateLimit(t *testing.T) {
	fastBackoff(t)
	seq := func(codes ...int) *httptest.Server {
		var i atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			k := int(i.Add(1)) - 1
			if k >= len(codes) {
				k = len(codes) - 1
			}
			w.WriteHeader(codes[k])
			_, _ = fmt.Fprint(w, `{"error":{"message":"x"}}`)
		}))
		t.Cleanup(srv.Close)
		return srv
	}
	run := func(srv *httptest.Server, attempts int) error {
		sdk := New(WithRetryPolicy(RetryPolicy{MaxAttempts: attempts}), WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
		c, _ := sdk.Chat("openai", "m")
		_, err := c.Call(context.Background(), userMsg())
		return err
	}
	var ae *APIError
	var rl *RateLimitError
	if err := run(seq(429, 400), 3); !errors.As(err, &ae) || ae.Status != 400 || errors.As(err, &rl) {
		t.Errorf("429 then 400: %v", err)
	}
	if err := run(seq(429, 500), 2); !errors.As(err, &rl) || rl.Attempts != 2 {
		t.Errorf("429 then final 500: %v", err)
	}
	if got := (&terminalError{err: errors.New("bad body")}).Error(); got != "bad body" {
		t.Errorf("terminalError = %q", got)
	}
	if terminal(nil) != nil {
		t.Error("terminal(nil) must be nil")
	}
}

// Streaming: a cancelled context ends before any attempt; transport
// failures exhaust into the wrapped error; persistent 429 is typed.
func TestCallStream_LadderEdges(t *testing.T) {
	fastBackoff(t)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1")), WithRetryPolicy(RetryPolicy{MaxAttempts: 2}))
	c, _ := sdk.Chat("openai", "m")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.CallStream(ctx, userMsg(), func(Delta) error { return nil }); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled: %v", err)
	}
	if _, err := c.CallStream(context.Background(), userMsg(), func(Delta) error { return nil }); err == nil {
		t.Error("unreachable host: want error")
	}
	srv, n := countingServer(t, http.StatusTooManyRequests, nil)
	s2 := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 2}), WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c2, _ := s2.Chat("openai", "m")
	var rl *RateLimitError
	if _, err := c2.CallStream(context.Background(), userMsg(), func(Delta) error { return nil }); !errors.As(err, &rl) || n.Load() != 2 {
		t.Errorf("429 ladder: %v (attempts %d)", err, n.Load())
	}
}

func TestBackoffDelayCapped_LongLadder(t *testing.T) {
	fastBackoff(t)
	for _, attempt := range []int{21, 40, 63, 1000} {
		if d := backoffDelayCapped(attempt, time.Hour); d <= 0 || d > time.Hour {
			t.Errorf("attempt %d → %v (must stay positive and capped)", attempt, d)
		}
	}
}

// Same-host redirects are still followed.
func TestSameHostRedirectFollowed(t *testing.T) {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/old/chat/completions" {
			http.Redirect(w, r, srv.URL+"/chat/completions", http.StatusTemporaryRedirect)
			return
		}
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL+"/old")))
	c, _ := sdk.Chat("openai", "m")
	if res, err := c.Call(context.Background(), userMsg()); err != nil || res.Content != "ok" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
}

func TestEngageLearnOnlyOnce(t *testing.T) {
	var n int
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI}, nil, nil)
	pc.opts.observer = func(LearnEvent) { n++ }
	pc.engageLearn(&pc.learn.forceBuffered, LearnBuffered, nil)
	pc.engageLearn(&pc.learn.forceBuffered, LearnBuffered, nil)
	if n != 1 {
		t.Errorf("observer fired %d times, want 1 (monotonic flag)", n)
	}
}

// A follower whose own context ends while waiting returns promptly.
func TestListModels_FollowerContextEnds(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
		_, _ = fmt.Fprint(w, `{"data":[]}`)
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	p, _ := sdk.Provider("openai")
	go func() { _, _ = p.ListModels(context.Background()) }()
	for {
		p.mu.Lock()
		inflight := p.flight != nil
		p.mu.Unlock()
		if inflight {
			break
		}
		time.Sleep(time.Millisecond)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := p.ListModels(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("follower err = %v", err)
	}
}

func TestSmallMappingEdges(t *testing.T) {
	// json_schema without Name uses the default schema name.
	m, _ := buildBody(t, FormatOpenAI, "gpt-4o", &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: testSchema}})
	if m["response_format"].(map[string]any)["json_schema"].(map[string]any)["name"] != defaultSchemaName {
		t.Errorf("default schema name: %v", m["response_format"])
	}
	// No JSON-tool call present: the result is untouched.
	res := &ChatResult{FinishReason: FinishToolCalls, ToolCalls: []ToolCall{{Name: "g"}}}
	foldAnthropicJSON(res, "answer")
	if len(res.ToolCalls) != 1 || res.FinishReason != FinishToolCalls {
		t.Errorf("unfolded = %+v", res)
	}
	// Object-shaped tool errors nest; text errors wrap.
	if got := string(wrapToolError(` {"code":7} `)); got != `{"error":{"code":7}}` {
		t.Errorf("object error = %s", got)
	}
	if u := usageFromOpenAI(nil); u != (Usage{}) {
		t.Errorf("nil usage = %+v", u)
	}
	// Anthropic: no content and no stop reason is a protocol error.
	if _, err := parseAnthropicResponse([]byte(`{"content":[]}`)); err == nil {
		t.Error("empty anthropic response must error")
	}
	// Gemini embeddings: malformed body.
	pc := newProviderClient(ProviderConfig{ID: "g", Format: FormatGemini}, nil, nil)
	if _, _, err := pc.parseEmbedResponse([]byte(`{bad`), 1); err == nil {
		t.Error("gemini embed parse must error")
	}
	// Tool images across a run of tool results land in one user message.
	req := &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "a", Name: "f"}, {ID: "b", Name: "f"}}},
		{Role: RoleTool, ToolCallID: "a", Parts: []ContentPart{TextPart("one"), ImagePart("image/png", []byte{1})}},
		{Role: RoleTool, ToolCallID: "b", Parts: []ContentPart{ImagePart("image/png", []byte{2})}},
	}}
	m, _ = buildBody(t, FormatOpenAI, "gpt-4o", req)
	msgs := m["messages"].([]any)
	if len(msgs) != 5 || msgs[4].(map[string]any)["role"] != "user" {
		t.Fatalf("messages = %v", msgs)
	}
	if parts := msgs[4].(map[string]any)["content"].([]any); len(parts) != 4 {
		t.Errorf("follow-up parts = %v", parts)
	}
	b, _ := json.Marshal(oaMessage{Role: "assistant", ContentParts: []oaContentPart{{Type: "text", Text: "x"}}})
	if !strings.Contains(string(b), `"content":[`) {
		t.Errorf("assistant parts = %s", b)
	}
}

func TestContentPartValidationEdges(t *testing.T) {
	big := make([]byte, MaxImageBytes+1)
	bad := []ContentPart{
		{Type: ContentPartText, Text: "x", MIMEType: "image/png"},
		{Type: ContentPartImage, Text: "x", MIMEType: "image/png", Image: []byte{1}},
		{Type: ContentPartImage, MIMEType: "image/png"},
		{Type: ContentPartImage, MIMEType: "image/png", Image: big},
	}
	for i, p := range bad {
		if err := validateMessageContent(Message{Role: RoleTool, Parts: []ContentPart{p}}, 0); err == nil {
			t.Errorf("part %d must be rejected", i)
		}
	}
}

func TestListModelsOpenAI_CreatedAt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[{"id":"m","created":1700000000,"max_input_tokens":42}]}`)
	}))
	t.Cleanup(srv.Close)
	ms, err := newListModels(srv.URL, FormatOpenAI)
	if err != nil || len(ms) != 1 || ms[0].CreatedAt.Unix() != 1700000000 || ms[0].ContextWindow != 42 {
		t.Fatalf("models=%+v err=%v", ms, err)
	}
}

// A malformed base URL is rejected at wiring time, and a request that
// cannot be built is never retried.
func TestBadBaseURL(t *testing.T) {
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://x\x7f")))
	var ce *ConfigError
	if _, err := sdk.Chat("openai", "m"); !errors.As(err, &ce) {
		t.Errorf("control char in base URL: %v", err)
	}
	fastBackoff(t)
	var n atomic.Int32
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "http://x"}, nil, nil)
	err := pc.withRetry(context.Background(), func() (time.Duration, error) {
		n.Add(1)
		return 0, &ConfigError{Msg: "build request: bad"}
	}, nil)
	if !errors.As(err, &ce) || n.Load() != 1 {
		t.Errorf("ConfigError retried: attempts=%d err=%v", n.Load(), err)
	}
}

func TestSpeakAndGeminiSpeechEdges(t *testing.T) {
	sdk := New(WithProvider("openai", WithAPIKey("k")))
	var ce *ConfigError
	if _, err := sdk.Speak(context.Background(), "openai", " ", SpeakRequest{Text: "x", Voice: "v"}); !errors.As(err, &ce) {
		t.Errorf("empty model: %v", err)
	}
	for _, body := range []string{`{bad`, `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/L16","data":"!!"}}]}}]}`} {
		if _, _, err := parseGeminiSpeakResponse([]byte(body)); err == nil {
			t.Errorf("%s: want error", body)
		}
	}
}

// Streaming: a non-retryable provider error ends the call at once; a 400
// that names no learnable constraint is returned as is on the buffered path.
func TestNonRetryableErrors(t *testing.T) {
	fastBackoff(t)
	srv, n := countingServer(t, http.StatusUnauthorized, nil)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c, _ := sdk.Chat("openai", "m")
	var ae *APIError
	if _, err := c.CallStream(context.Background(), userMsg(), func(Delta) error { return nil }); !errors.As(err, &ae) || ae.Status != 401 || n.Load() != 1 {
		t.Errorf("stream 401: %v (attempts %d)", err, n.Load())
	}
	srv2, n2 := countingServer(t, http.StatusBadRequest, nil)
	s2 := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv2.URL)))
	c2, _ := s2.Chat("openai", "m")
	if _, err := c2.Call(context.Background(), toolReq(nil)); !errors.As(err, &ae) || ae.Status != 400 || n2.Load() != 1 {
		t.Errorf("plain 400 with tools: %v (attempts %d)", err, n2.Load())
	}
	if !chatCompletionsRejectsExplicitReasoningWithTools("gpt-5.6") {
		t.Error("gpt-5.6 rejects explicit reasoning with tools")
	}
}

// Transport failures on the speech paths are retried, then surfaced.
func TestSpeechTransportErrors(t *testing.T) {
	sdk := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 1}), WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1")))
	if _, err := sdk.Speak(context.Background(), "openai", "m", SpeakRequest{Text: "x", Voice: "v"}); err == nil || !strings.Contains(err.Error(), "retry exhausted") {
		t.Errorf("Speak: %v", err)
	}
	if _, err := sdk.Transcribe(context.Background(), "openai", "m", TranscribeRequest{Audio: []byte("a")}); err == nil || !strings.Contains(err.Error(), "retry exhausted") {
		t.Errorf("Transcribe: %v", err)
	}
}
