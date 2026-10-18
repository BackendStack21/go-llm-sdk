package llm

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// hdrRec records every request's headers; when failFirst is set the first
// request is a 500 so the per-attempt application of headers is observable.
type hdrRec struct {
	mu   sync.Mutex
	seen []http.Header
	hits atomic.Int32
}

func (r *hdrRec) all() []http.Header {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]http.Header(nil), r.seen...)
}

func hdrSrv(t *testing.T, failFirst bool, ctype, body string) (*httptest.Server, *hdrRec) {
	t.Helper()
	rec := &hdrRec{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.mu.Lock()
		rec.seen = append(rec.seen, r.Header.Clone())
		rec.mu.Unlock()
		if failFirst && rec.hits.Add(1) == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
			return
		}
		w.Header().Set("Content-Type", ctype)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

const (
	hdrOpenAIBody = `{"id":"c1","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`
	hdrRespBody   = `{"id":"resp_1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}`
	hdrAnthBody   = `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	hdrGemBody    = `{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`
)

func hdrSSE() string {
	return metaSSE(`{"id":"c1","choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`, "[DONE]")
}

// Per-request headers reach the server on every attempt, buffered, streaming
// and Responses-diverted, on every chat format.
func TestM6_HeadersChatEveryPathEveryAttempt(t *testing.T) {
	cases := []struct {
		name   string
		cfg    ProviderConfig
		model  string
		stream bool
		ctype  string
		body   string
	}{
		{"openai buffered", ProviderConfig{ID: "openai", Format: FormatOpenAI}, "", false, "application/json", hdrOpenAIBody},
		{"openai stream", ProviderConfig{ID: "openai", Format: FormatOpenAI}, "", true, "text/event-stream", hdrSSE()},
		{"responses diverted buffered", ProviderConfig{ID: "openai", Format: FormatOpenAI}, "gpt-5.6-luna", false, "application/json", hdrRespBody},
		{"anthropic buffered", ProviderConfig{ID: "anthropic", Format: FormatAnthropic}, "", false, "application/json", hdrAnthBody},
		{"gemini buffered", ProviderConfig{ID: "gemini", Format: FormatGemini}, "", false, "application/json", hdrGemBody},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv, rec := hdrSrv(t, true, c.ctype, c.body)
			cfg := c.cfg
			cfg.BaseURL, cfg.APIKey = srv.URL, "k"
			cc := newTestClient(t, cfg, srv)
			if c.model != "" {
				cc.model = c.model
			}
			req := m2Req()
			req.Headers = map[string]string{"X-Trace": "t-1", "anthropic-beta": "beta-x"}
			if _, _ = m2Run(t, cc, req, c.stream); false {
				return
			}
			got := rec.all()
			if len(got) < 2 {
				t.Fatalf("attempts = %d, want >= 2", len(got))
			}
			for i, h := range got {
				if h.Get("X-Trace") != "t-1" || h.Get("Anthropic-Beta") != "beta-x" {
					t.Errorf("attempt %d headers = %v, want X-Trace and anthropic-beta", i+1, h)
				}
			}
		})
	}
}

// Per-request headers win over provider headers; an empty value removes a
// non-reserved provider header; neither leaks into the next call.
func TestM6_HeadersMergeOrderAndNoLeak(t *testing.T) {
	srv, rec := hdrSrv(t, false, "application/json", hdrOpenAIBody)
	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k",
		Headers: map[string]string{"X-Over": "provider", "X-Drop": "provider", "X-Keep": "provider"}}
	cc := newTestClient(t, cfg, srv)

	req := m2Req()
	req.Headers = map[string]string{"X-Over": "request", "X-Drop": "", "X-New": "n"}
	m2Run(t, cc, req, false)
	m2Run(t, cc, m2Req(), false)

	got := rec.all()
	if len(got) != 2 {
		t.Fatalf("requests = %d, want 2", len(got))
	}
	h := got[0]
	if h.Get("X-Over") != "request" || h.Get("X-New") != "n" || h.Get("X-Keep") != "provider" {
		t.Errorf("first call headers = %v", h)
	}
	if _, ok := h["X-Drop"]; ok {
		t.Errorf("empty value did not remove X-Drop: %v", h)
	}
	h = got[1]
	if h.Get("X-Over") != "provider" || h.Get("X-Drop") != "provider" || h.Get("X-New") != "" {
		t.Errorf("second call leaked per-request headers: %v", h)
	}
}

// Auth stays intact when per-request headers are present.
func TestM6_HeadersAuthIntact(t *testing.T) {
	srv, rec := hdrSrv(t, false, "application/json", hdrOpenAIBody)
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "sekret"}, srv)
	req := m2Req()
	req.Headers = map[string]string{"X-A": "b"}
	m2Run(t, cc, req, false)
	got := rec.all()
	if len(got) != 1 || got[0].Get("X-A") != "b" || got[0].Get("Authorization") != "Bearer sekret" {
		t.Errorf("headers = %v", got)
	}
}

// Embed, Speak and Transcribe honour per-request headers on every attempt
// and do not leak them into the next call.
func TestM6_HeadersNonChatEntryPoints(t *testing.T) {
	embedOpenAI := `{"data":[{"index":0,"embedding":[0.5,1]}],"usage":{"prompt_tokens":2}}`
	embedGemini := `{"embeddings":[{"values":[0.5,1]}]}`
	geminiAudio := `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/L16;rate=24000","data":"AQID"}}]}}]}`
	hd := map[string]string{"X-Trace": "t-2"}
	runs := []struct {
		name   string
		format Format
		id     string
		ctype  string
		body   string
		call   func(s *SDK, id string, h map[string]string) error
	}{
		{"embed openai", FormatOpenAI, "openai", "application/json", embedOpenAI, func(s *SDK, id string, h map[string]string) error {
			_, err := s.Embed(t.Context(), id, "m", EmbedRequest{Inputs: []string{"a"}, Headers: h})
			return err
		}},
		{"embed gemini", FormatGemini, "gemini", "application/json", embedGemini, func(s *SDK, id string, h map[string]string) error {
			_, err := s.Embed(t.Context(), id, "m", EmbedRequest{Inputs: []string{"a"}, Headers: h})
			return err
		}},
		{"speak openai", FormatOpenAI, "openai", "audio/mpeg", "\x01\x02", func(s *SDK, id string, h map[string]string) error {
			_, err := s.Speak(t.Context(), id, "tts-1", SpeakRequest{Text: "hi", Voice: "alloy", Headers: h})
			return err
		}},
		{"speak gemini", FormatGemini, "gemini", "application/json", geminiAudio, func(s *SDK, id string, h map[string]string) error {
			_, err := s.Speak(t.Context(), id, "gemini-tts", SpeakRequest{Text: "hi", Voice: "Kore", Headers: h})
			return err
		}},
		{"transcribe openai", FormatOpenAI, "openai", "application/json", `{"text":"hi"}`, func(s *SDK, id string, h map[string]string) error {
			_, err := s.Transcribe(t.Context(), id, "whisper-1", TranscribeRequest{Audio: []byte{1, 2}, Headers: h})
			return err
		}},
	}
	for _, r := range runs {
		t.Run(r.name, func(t *testing.T) {
			srv, rec := hdrSrv(t, true, r.ctype, r.body)
			s := newTestSDK(t, ProviderConfig{ID: r.id, Format: r.format, APIKey: "k"}, srv)
			if err := r.call(s, r.id, hd); err != nil {
				t.Fatal(err)
			}
			got := rec.all()
			if len(got) < 2 {
				t.Fatalf("attempts = %d, want >= 2", len(got))
			}
			for i, h := range got {
				if h.Get("X-Trace") != "t-2" {
					t.Errorf("attempt %d missing X-Trace: %v", i+1, h)
				}
			}
			n := len(got)
			if err := r.call(s, r.id, nil); err != nil {
				t.Fatal(err)
			}
			if last := rec.all()[n]; last.Get("X-Trace") != "" {
				t.Errorf("per-request header leaked into next call: %v", last)
			}
		})
	}
}

// Reserved names (any case) and malformed names/values are rejected with a
// *ConfigError before any HTTP request; the error never carries the value
// or the API key.
func TestM6_HeadersRejectedBeforeNetwork(t *testing.T) {
	const secretVal = "SECRET-VALUE-9f3a"
	const key = "key-must-not-leak"
	reserved := []string{"Authorization", "authorization", "AUTHORIZATION", "Proxy-Authorization", "X-Api-Key", "x-api-key",
		"X-Goog-Api-Key", "x-goog-api-key", "Host", "Content-Length", "Content-Type", "content-type", "Transfer-Encoding"}
	type bad struct {
		name       string
		h          map[string]string
		namedInErr string // header name that must appear in the error ("" = not asserted)
	}
	var bads []bad
	for _, n := range reserved {
		bads = append(bads, bad{"reserved " + n, map[string]string{n: secretVal}, n})
	}
	bads = append(bads,
		bad{"reserved with empty value", map[string]string{"Authorization": ""}, "Authorization"},
		bad{"empty name", map[string]string{"": secretVal}, ""},
		bad{"space in name", map[string]string{"X Bad": secretVal}, ""},
		bad{"colon in name", map[string]string{"X:Bad": secretVal}, ""},
		bad{"CR in name", map[string]string{"X-Bad\r": secretVal}, ""},
		bad{"LF in name", map[string]string{"X-Bad\n": secretVal}, ""},
		bad{"NUL in name", map[string]string{"X-Bad\x00": secretVal}, ""},
		bad{"non-ASCII name", map[string]string{"X-Bäd": secretVal}, ""},
		bad{"CR in value", map[string]string{"X-Ok": secretVal + "\rX-Evil: 1"}, ""},
		bad{"LF in value", map[string]string{"X-Ok": secretVal + "\nX-Evil: 1"}, ""},
		bad{"CRLF in value", map[string]string{"X-Ok": secretVal + "\r\nX-Evil: 1"}, ""},
		bad{"NUL in value", map[string]string{"X-Ok": secretVal + "\x00"}, ""},
	)
	entries := []struct {
		name string
		call func(t *testing.T, s *SDK, cc *ChatClient, h map[string]string) error
	}{
		{"chat", func(t *testing.T, _ *SDK, cc *ChatClient, h map[string]string) error {
			req := m2Req()
			req.Headers = h
			_, err := cc.Call(t.Context(), req)
			return err
		}},
		{"stream", func(t *testing.T, _ *SDK, cc *ChatClient, h map[string]string) error {
			req := m2Req()
			req.Headers = h
			_, err := cc.CallStream(t.Context(), req, func(Delta) error { return nil })
			return err
		}},
		{"responses", func(t *testing.T, _ *SDK, cc *ChatClient, h map[string]string) error {
			cc.model = "gpt-5.6-luna"
			req := m2Req()
			req.Headers = h
			_, err := cc.Call(t.Context(), req)
			return err
		}},
		{"embed", func(t *testing.T, s *SDK, _ *ChatClient, h map[string]string) error {
			_, err := s.Embed(t.Context(), "openai", "m", EmbedRequest{Inputs: []string{"a"}, Headers: h})
			return err
		}},
		{"speak", func(t *testing.T, s *SDK, _ *ChatClient, h map[string]string) error {
			_, err := s.Speak(t.Context(), "openai", "tts-1", SpeakRequest{Text: "hi", Voice: "alloy", Headers: h})
			return err
		}},
		{"transcribe", func(t *testing.T, s *SDK, _ *ChatClient, h map[string]string) error {
			_, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}, Headers: h})
			return err
		}},
	}
	for _, e := range entries {
		for _, b := range bads {
			t.Run(e.name+"/"+b.name, func(t *testing.T) {
				srv, rec := hdrSrv(t, false, "application/json", hdrOpenAIBody)
				cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: key}
				s := newTestSDK(t, cfg, srv)
				cfg.BaseURL = srv.URL
				cc := newTestClient(t, cfg, srv)
				err := e.call(t, s, cc, b.h)
				var ce *ConfigError
				if !errors.As(err, &ce) {
					t.Fatalf("err = %v (%T), want *ConfigError", err, err)
				}
				if n := len(rec.all()); n != 0 {
					t.Errorf("%d HTTP requests made, want 0", n)
				}
				msg := err.Error()
				if strings.Contains(msg, secretVal) || strings.Contains(msg, key) {
					t.Errorf("error text leaks value or key: %q", msg)
				}
				if b.namedInErr != "" && !strings.Contains(strings.ToLower(msg), strings.ToLower(b.namedInErr)) {
					t.Errorf("error %q does not name header %q", msg, b.namedInErr)
				}
			})
		}
	}
}

// anthropic-version and anthropic-beta are not reserved; a provider header
// that is not reserved can be set by name in any case.
func TestM6_HeadersAnthropicVersionOverridable(t *testing.T) {
	srv, rec := hdrSrv(t, false, "application/json", hdrAnthBody)
	cc := newTestClient(t, ProviderConfig{ID: "anthropic", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k"}, srv)
	req := m2Req()
	req.Headers = map[string]string{"Anthropic-Version": "2099-01-01"}
	m2Run(t, cc, req, false)
	got := rec.all()
	if len(got) != 1 || got[0].Get("Anthropic-Version") != "2099-01-01" || got[0].Get("X-Api-Key") != "k" {
		t.Errorf("headers = %v", got)
	}
}

// Nil and empty maps change nothing (AC-5).
func TestM6_HeadersNilEmptyNoChange(t *testing.T) {
	for _, h := range []map[string]string{nil, {}} {
		srv, rec := hdrSrv(t, false, "application/json", hdrOpenAIBody)
		cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k",
			Headers: map[string]string{"X-P": "p"}}, srv)
		req := m2Req()
		req.Headers = h
		m2Run(t, cc, req, false)
		got := rec.all()
		if len(got) != 1 || got[0].Get("X-P") != "p" || got[0].Get("Authorization") != "Bearer k" {
			t.Errorf("headers = %v", got)
		}
	}
}
