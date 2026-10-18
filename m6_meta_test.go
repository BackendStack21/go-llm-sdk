package llm

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// metaSrv answers with body and hdr. When failFirst is set the first request
// is a 500 carrying poisoned headers that must never reach the result.
func metaSrv(t *testing.T, failFirst bool, ctype string, hdr map[string]string, body string) *httptest.Server {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if failFirst && hits.Add(1) == 1 {
			w.Header().Set("X-Request-Id", "req-poison")
			w.Header().Set("X-Ratelimit-Remaining-Requests", "1")
			w.Header().Set("X-Ratelimit-Remaining-Tokens", "2")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
			return
		}
		for k, v := range hdr {
			w.Header().Set(k, v)
		}
		w.Header().Set("Content-Type", ctype)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

var metaFullHdr = map[string]string{
	"X-Request-Id":                   "req-123",
	"X-Ratelimit-Remaining-Requests": "99",
	"X-Ratelimit-Remaining-Tokens":   "4000",
	"X-Ratelimit-Reset-Requests":     "1s",
	"X-Ratelimit-Reset-Tokens":       "6m0s",
}

var metaFullRL = RateLimit{Reported: true, RemainingRequests: 99, RemainingTokens: 4000,
	ResetRequests: time.Second, ResetTokens: 6 * time.Minute}

func metaSSE(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

func TestM6_MetaChatEveryFormat(t *testing.T) {
	anthStream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	cases := []struct {
		name      string
		format    Format
		id, model string
		stream    bool
		body      string
		wantID    string
		wantModel string
	}{
		{"openai buffered", FormatOpenAI, "openai", "", false,
			`{"id":"chatcmpl-1","model":"gpt-x","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`,
			"chatcmpl-1", "gpt-x"},
		{"openai stream", FormatOpenAI, "openai", "", true,
			metaSSE(`{"id":"chatcmpl-2","model":"gpt-y","choices":[{"delta":{"content":"hi"}}]}`,
				`{"id":"chatcmpl-2","model":"gpt-y","choices":[{"delta":{},"finish_reason":"stop"}]}`, "[DONE]"),
			"chatcmpl-2", "gpt-y"},
		{"responses buffered", FormatOpenAI, "openai", "gpt-5.6-luna", false,
			`{"id":"resp_1","model":"gpt-5.6-luna-2026","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}`,
			"resp_1", "gpt-5.6-luna-2026"},
		{"responses stream", FormatOpenAI, "openai", "gpt-5.6-luna", true,
			metaSSE(`{"type":"response.created","response":{"id":"resp_2","model":"gpt-5.6-luna-2026"}}`,
				`{"type":"response.output_text.delta","delta":"hi"}`,
				`{"type":"response.completed","response":{"id":"resp_2","model":"gpt-5.6-luna-2026","status":"completed"}}`),
			"resp_2", "gpt-5.6-luna-2026"},
		{"anthropic buffered", FormatAnthropic, "anthropic", "", false,
			`{"id":"msg_1","model":"claude-x","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
			"msg_1", "claude-x"},
		{"anthropic stream", FormatAnthropic, "anthropic", "", true, anthStream, "msg_1", "claude-x"},
		{"gemini buffered", FormatGemini, "gemini", "", false,
			`{"responseId":"gr-1","modelVersion":"gemini-2.5","candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`,
			"gr-1", "gemini-2.5"},
		{"gemini stream", FormatGemini, "gemini", "", true,
			metaSSE(`{"responseId":"gr-2","modelVersion":"gemini-2.5","candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`),
			"gr-2", "gemini-2.5"},
	}
	for _, c := range cases {
		for _, failFirst := range []bool{false, true} {
			t.Run(c.name+map[bool]string{false: "", true: " after failed attempt"}[failFirst], func(t *testing.T) {
				ctype := "application/json"
				if c.stream {
					ctype = "text/event-stream"
				}
				srv := metaSrv(t, failFirst, ctype, metaFullHdr, c.body)
				cc := newTestClient(t, ProviderConfig{ID: c.id, Format: c.format, BaseURL: srv.URL, APIKey: "k"}, srv)
				if c.model != "" {
					cc.model = c.model
				}
				res, _ := m2Run(t, cc, m2Req(), c.stream)
				if res.Meta.ID != c.wantID || res.Meta.Model != c.wantModel {
					t.Errorf("ID/Model = %q/%q, want %q/%q", res.Meta.ID, res.Meta.Model, c.wantID, c.wantModel)
				}
				if res.Meta.RequestID != "req-123" {
					t.Errorf("RequestID = %q, want req-123", res.Meta.RequestID)
				}
				if res.Meta.RateLimit != metaFullRL {
					t.Errorf("RateLimit = %+v, want %+v", res.Meta.RateLimit, metaFullRL)
				}
			})
		}
	}
}

// A failed attempt's headers never leak into a later success lacking them.
func TestM6_MetaFailedAttemptHeadersDoNotLeak(t *testing.T) {
	for _, stream := range []bool{false, true} {
		body := `{"id":"c1","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`
		ctype := "application/json"
		if stream {
			body = metaSSE(`{"id":"c1","choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`, "[DONE]")
			ctype = "text/event-stream"
		}
		srv := metaSrv(t, true, ctype, nil, body)
		cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
		res, _ := m2Run(t, cc, m2Req(), stream)
		if res.Meta.ID != "c1" {
			t.Errorf("stream=%v ID = %q, want c1", stream, res.Meta.ID)
		}
		if res.Meta.RequestID != "" || res.Meta.RateLimit != (RateLimit{}) {
			t.Errorf("stream=%v leaked failed-attempt headers: %+v", stream, res.Meta)
		}
	}
}

// Absent data stays zero on every format.
func TestM6_MetaAbsentIsZero(t *testing.T) {
	for _, c := range []struct {
		format Format
		id     string
		body   string
	}{
		{FormatOpenAI, "openai", `{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`},
		{FormatAnthropic, "anthropic", `{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`},
		{FormatGemini, "gemini", `{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`},
	} {
		srv := metaSrv(t, false, "application/json", nil, c.body)
		cc := newTestClient(t, ProviderConfig{ID: c.id, Format: c.format, BaseURL: srv.URL, APIKey: "k"}, srv)
		res, _ := m2Run(t, cc, m2Req(), false)
		if res.Meta != (ResponseMeta{}) {
			t.Errorf("%s: Meta = %+v, want zero", c.format, res.Meta)
		}
	}
}

// A successful JSON body to a streaming request (non-SSE fallback) fills Meta.
func TestM6_MetaNonSSEStreamFallback(t *testing.T) {
	srv := metaSrv(t, false, "application/json", metaFullHdr,
		`{"id":"chatcmpl-9","model":"gpt-z","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, _ := m2Run(t, cc, m2Req(), true)
	want := ResponseMeta{ID: "chatcmpl-9", Model: "gpt-z", RequestID: "req-123", RateLimit: metaFullRL}
	if res.Meta != want {
		t.Errorf("Meta = %+v, want %+v", res.Meta, want)
	}
}

func TestM6_ParseResponseMeta(t *testing.T) {
	future := time.Now().Add(90 * time.Second).UTC().Format(time.RFC3339)
	past := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	cases := []struct {
		name  string
		h     map[string]string
		check func(t *testing.T, m ResponseMeta)
	}{
		{"empty", nil, func(t *testing.T, m ResponseMeta) {
			if m != (ResponseMeta{}) {
				t.Errorf("got %+v", m)
			}
		}},
		{"openai form", metaFullHdr, func(t *testing.T, m ResponseMeta) {
			if m.RequestID != "req-123" || m.RateLimit != metaFullRL {
				t.Errorf("got %+v", m)
			}
		}},
		{"reset forms: ms and plain seconds", map[string]string{
			"x-ratelimit-reset-requests": "20ms", "x-ratelimit-reset-tokens": "30"}, func(t *testing.T, m ResponseMeta) {
			rl := m.RateLimit
			if !rl.Reported || rl.ResetRequests != 20*time.Millisecond || rl.ResetTokens != 30*time.Second {
				t.Errorf("got %+v", rl)
			}
		}},
		{"anthropic form", map[string]string{
			"anthropic-ratelimit-requests-remaining": "7", "anthropic-ratelimit-tokens-remaining": "800",
			"anthropic-ratelimit-requests-reset": future, "anthropic-ratelimit-tokens-reset": future,
			"request-id": "req-anth"}, func(t *testing.T, m ResponseMeta) {
			rl := m.RateLimit
			if m.RequestID != "req-anth" || !rl.Reported || rl.RemainingRequests != 7 || rl.RemainingTokens != 800 {
				t.Errorf("got %+v", m)
			}
			for _, d := range []time.Duration{rl.ResetRequests, rl.ResetTokens} {
				if d <= 60*time.Second || d > 91*time.Second {
					t.Errorf("reset = %v, want ~90s", d)
				}
			}
		}},
		{"anthropic reset in the past floors at zero but is reported", map[string]string{
			"anthropic-ratelimit-requests-reset": past}, func(t *testing.T, m ResponseMeta) {
			if !m.RateLimit.Reported || m.RateLimit.ResetRequests != 0 {
				t.Errorf("got %+v", m.RateLimit)
			}
		}},
		{"openai wins per field, anthropic fills the rest", map[string]string{
			"x-ratelimit-remaining-requests":         "5",
			"anthropic-ratelimit-requests-remaining": "50",
			"anthropic-ratelimit-tokens-remaining":   "60"}, func(t *testing.T, m ResponseMeta) {
			if m.RateLimit.RemainingRequests != 5 || m.RateLimit.RemainingTokens != 60 {
				t.Errorf("got %+v", m.RateLimit)
			}
		}},
		{"malformed field stays zero, others parse", map[string]string{
			"x-ratelimit-remaining-requests": "abc", "x-ratelimit-remaining-tokens": "12",
			"x-ratelimit-reset-requests": "soon"}, func(t *testing.T, m ResponseMeta) {
			rl := m.RateLimit
			if !rl.Reported || rl.RemainingRequests != 0 || rl.RemainingTokens != 12 || rl.ResetRequests != 0 {
				t.Errorf("got %+v", rl)
			}
		}},
		{"all malformed is not reported", map[string]string{
			"x-ratelimit-remaining-requests": "abc", "x-ratelimit-reset-tokens": "??"}, func(t *testing.T, m ResponseMeta) {
			if m.RateLimit != (RateLimit{}) {
				t.Errorf("got %+v", m.RateLimit)
			}
		}},
		{"request id precedence", map[string]string{
			"x-request-id": "a", "request-id": "b", "x-goog-request-id": "c", "x-amzn-requestid": "d"}, func(t *testing.T, m ResponseMeta) {
			if m.RequestID != "a" {
				t.Errorf("got %q", m.RequestID)
			}
		}},
		{"request-id second", map[string]string{"request-id": "b", "x-goog-request-id": "c"}, func(t *testing.T, m ResponseMeta) {
			if m.RequestID != "b" {
				t.Errorf("got %q", m.RequestID)
			}
		}},
		{"goog third", map[string]string{"x-goog-request-id": "c", "x-amzn-requestid": "d"}, func(t *testing.T, m ResponseMeta) {
			if m.RequestID != "c" {
				t.Errorf("got %q", m.RequestID)
			}
		}},
		{"amzn fourth, cf-ray ignored", map[string]string{"x-amzn-requestid": "d", "cf-ray": "ray"}, func(t *testing.T, m ResponseMeta) {
			if m.RequestID != "d" {
				t.Errorf("got %q", m.RequestID)
			}
		}},
		{"cf-ray alone ignored", map[string]string{"cf-ray": "ray"}, func(t *testing.T, m ResponseMeta) {
			if m.RequestID != "" {
				t.Errorf("got %q", m.RequestID)
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := http.Header{}
			for k, v := range c.h {
				h.Set(k, v)
			}
			c.check(t, parseResponseMeta(h))
		})
	}
}

// Embed / Speak / Transcribe carry RequestID + RateLimit; a failed first
// attempt's headers never appear.
func TestM6_MetaNonChatEntryPoints(t *testing.T) {
	embedOpenAI := `{"data":[{"index":0,"embedding":[0.5,1]}],"usage":{"prompt_tokens":2}}`
	embedGemini := `{"embeddings":[{"values":[0.5,1]}]}`
	geminiAudio := `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/L16;rate=24000","data":"AQID"}}]}}]}`
	for _, failFirst := range []bool{false, true} {
		suffix := map[bool]string{false: "", true: " after failed attempt"}[failFirst]
		runs := []struct {
			name   string
			format Format
			id     string
			ctype  string
			body   string
			call   func(t *testing.T, s *SDK, id string) ResponseMeta
		}{
			{"embed openai", FormatOpenAI, "openai", "application/json", embedOpenAI,
				func(t *testing.T, s *SDK, id string) ResponseMeta {
					r, err := s.Embed(t.Context(), id, "m", EmbedRequest{Inputs: []string{"a"}})
					if err != nil {
						t.Fatal(err)
					}
					return r.Meta
				}},
			{"embed gemini", FormatGemini, "gemini", "application/json", embedGemini,
				func(t *testing.T, s *SDK, id string) ResponseMeta {
					r, err := s.Embed(t.Context(), id, "m", EmbedRequest{Inputs: []string{"a"}})
					if err != nil {
						t.Fatal(err)
					}
					return r.Meta
				}},
			{"speak openai", FormatOpenAI, "openai", "audio/mpeg", "\x01\x02",
				func(t *testing.T, s *SDK, id string) ResponseMeta {
					r, err := s.Speak(t.Context(), id, "tts-1", SpeakRequest{Text: "hi", Voice: "alloy"})
					if err != nil {
						t.Fatal(err)
					}
					return r.Meta
				}},
			{"speak gemini", FormatGemini, "gemini", "application/json", geminiAudio,
				func(t *testing.T, s *SDK, id string) ResponseMeta {
					r, err := s.Speak(t.Context(), id, "gemini-tts", SpeakRequest{Text: "hi", Voice: "Kore"})
					if err != nil {
						t.Fatal(err)
					}
					return r.Meta
				}},
			{"transcribe openai", FormatOpenAI, "openai", "application/json", `{"text":"hi"}`,
				func(t *testing.T, s *SDK, id string) ResponseMeta {
					r, err := s.Transcribe(t.Context(), id, "whisper-1", TranscribeRequest{Audio: []byte{1, 2}})
					if err != nil {
						t.Fatal(err)
					}
					return r.Meta
				}},
		}
		for _, r := range runs {
			t.Run(r.name+suffix, func(t *testing.T) {
				srv := metaSrv(t, failFirst, r.ctype, metaFullHdr, r.body)
				s := newTestSDK(t, ProviderConfig{ID: r.id, Format: r.format, APIKey: "k"}, srv)
				m := r.call(t, s, r.id)
				if m.RequestID != "req-123" {
					t.Errorf("RequestID = %q, want req-123", m.RequestID)
				}
				if m.RateLimit != metaFullRL {
					t.Errorf("RateLimit = %+v, want %+v", m.RateLimit, metaFullRL)
				}
			})
		}
	}
}

func TestM6_MetaNonChatAbsentIsZero(t *testing.T) {
	srv := metaSrv(t, false, "application/json", nil, `{"text":"hi"}`)
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	r, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Meta != (ResponseMeta{}) {
		t.Errorf("Meta = %+v, want zero", r.Meta)
	}
}
