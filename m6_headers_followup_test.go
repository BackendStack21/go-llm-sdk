package llm

import (
	"errors"
	"strings"
	"testing"
)

func hdrFollowEntries() []struct {
	name string
	call func(t *testing.T, s *SDK, cc *ChatClient, h map[string]string) error
} {
	return []struct {
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
}

// Two keys equal case-insensitively are ambiguous and rejected; so are
// values with ASCII control bytes other than HTAB (including DEL).
func TestM6_HeadersFollowupRejected(t *testing.T) {
	const secretVal = "SECRET-VALUE-77c1"
	const key = "key-must-not-leak"
	bads := []struct {
		name string
		h    map[string]string
	}{
		{"duplicate differing case", map[string]string{"X-Dup": secretVal, "x-dup": secretVal + "2"}},
		{"duplicate upper and lower", map[string]string{"X-TRACE": secretVal, "x-trace": ""}},
		{"SOH in value", map[string]string{"X-Ok": secretVal + "\x01"}},
		{"BEL in value", map[string]string{"X-Ok": secretVal + "\x07"}},
		{"US in value", map[string]string{"X-Ok": secretVal + "\x1f"}},
		{"DEL in value", map[string]string{"X-Ok": secretVal + "\x7f"}},
	}
	for _, e := range hdrFollowEntries() {
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
				if msg := err.Error(); strings.Contains(msg, secretVal) || strings.Contains(msg, key) {
					t.Errorf("error leaks value or key: %q", msg)
				}
			})
		}
	}
}

// HTAB and printable/obs-text bytes remain valid in header values.
func TestM6_HeadersFollowupTabAllowed(t *testing.T) {
	srv, rec := hdrSrv(t, false, "application/json", hdrOpenAIBody)
	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	req := m2Req()
	req.Headers = map[string]string{"X-Tab": "a\tb", "X-Other": "v"}
	m2Run(t, cc, req, false)
	got := rec.all()
	if len(got) != 1 || got[0].Get("X-Tab") != "a\tb" || got[0].Get("X-Other") != "v" {
		t.Errorf("headers = %v", got)
	}
}
