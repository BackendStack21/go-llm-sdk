package llm

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// newTestSDK builds an SDK with one provider pinned to an httptest server
// and the backoff unit shortened so retry paths are fast.
func newTestSDK(t *testing.T, cfg ProviderConfig, srv *httptest.Server) *SDK {
	t.Helper()
	old := backoffUnit
	backoffUnit = time.Millisecond
	t.Cleanup(func() { backoffUnit = old })
	s := New()
	if srv != nil {
		cfg.BaseURL = srv.URL
	}
	s.put(cfg)
	return s
}

func TestSpeak_OpenAI(t *testing.T) {
	cases := []struct {
		name        string
		contentType string // response Content-Type; "" = omit header
		wantMIME    string
	}{
		{"propagated content type", "audio/mpeg", "audio/mpeg"},
		{"wav content type", "audio/wav", "audio/wav"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath, gotAuth string
			var reqBody map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				gotAuth = r.Header.Get("Authorization")
				b, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(b, &reqBody)
				w.Header().Set("Content-Type", "application/json")
				if tc.contentType == "\x00unset\x00" {
					// Suppress Go's sniffing so the SDK sees no Content-Type.
					w.Header()["Content-Type"] = nil
				} else if tc.contentType != "" {
					w.Header().Set("Content-Type", tc.contentType)
				}
				_, _ = w.Write([]byte{0xff, 0xf3, 0x00, 0x01})
			}))
			defer srv.Close()

			s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k-secret"}, srv)
			res, err := s.Speak("openai", "tts-1", SpeakRequest{Text: "hello", Voice: "alloy"})
			if err != nil {
				t.Fatalf("Speak: %v", err)
			}
			if gotPath != "/audio/speech" {
				t.Errorf("path = %q, want /audio/speech", gotPath)
			}
			if gotAuth == "" || !containsNoSecret(gotAuth, "k-secret") {
				t.Errorf("bearer auth not sent correctly: %q", gotAuth)
			}
			if reqBody["model"] != "tts-1" || reqBody["input"] != "hello" || reqBody["voice"] != "alloy" {
				t.Errorf("request body = %v", reqBody)
			}
			if reqBody["response_format"] != "mp3" {
				t.Errorf("response_format = %v, want mp3", reqBody["response_format"])
			}
			if _, has := reqBody["speed"]; has {
				t.Errorf("speed should be omitted when zero, got %v", reqBody["speed"])
			}
			if string(res.Audio) != string([]byte{0xff, 0xf3, 0x00, 0x01}) {
				t.Errorf("audio bytes = %v", res.Audio)
			}
			if res.MIMEType != tc.wantMIME {
				t.Errorf("MIMEType = %q, want %q", res.MIMEType, tc.wantMIME)
			}
			if res.Model != "tts-1" {
				t.Errorf("Model = %q, want tts-1", res.Model)
			}
		})
	}
}

// containsNoSecret reports a header value that must contain the key but the
// test asserts presence, not leakage into errors.
func containsNoSecret(v, key string) bool { return v == "Bearer "+key }

func TestSpeak_SpeedIncluded(t *testing.T) {
	var reqBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &reqBody)
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte{1})
	}))
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	if _, err := s.Speak("openai", "tts-1", SpeakRequest{Text: "hi", Voice: "echo", Speed: 1.5}); err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if reqBody["speed"] != 1.5 {
		t.Errorf("speed = %v, want 1.5", reqBody["speed"])
	}
}

func TestSpeak_Gemini(t *testing.T) {
	pcm := []byte{0x01, 0x02, 0x03, 0x04}
	resp := map[string]any{
		"candidates": []any{map[string]any{
			"content": map[string]any{
				"parts": []any{map[string]any{
					"inlineData": map[string]any{
						"mimeType": "audio/L16;rate=24000",
						"data":     base64.StdEncoding.EncodeToString(pcm),
					},
				}},
			},
		}},
	}
	var reqBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models/gemini-tts:generateContent" {
			t.Errorf("path = %q", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &reqBody)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	s := newTestSDK(t, ProviderConfig{ID: "gemini", Format: FormatGemini, APIKey: "k"}, srv)
	res, err := s.Speak("gemini", "gemini-tts", SpeakRequest{Text: "hello", Voice: "Kore"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if string(res.Audio) != string(pcm) {
		t.Errorf("audio bytes = %v, want %v", res.Audio, pcm)
	}
	if res.MIMEType != "audio/L16;rate=24000" {
		t.Errorf("MIMEType = %q", res.MIMEType)
	}
	if res.Model != "gemini-tts" {
		t.Errorf("Model = %q", res.Model)
	}
	gc := reqBody["generationConfig"].(map[string]any)
	mods, _ := gc["responseModalities"].([]any)
	if len(mods) != 1 || mods[0] != "AUDIO" {
		t.Errorf("responseModalities = %v", mods)
	}
	sc := gc["speechConfig"].(map[string]any)
	vc := sc["voiceConfig"].(map[string]any)
	pv := vc["prebuiltVoiceConfig"].(map[string]any)
	if pv["voiceName"] != "Kore" {
		t.Errorf("voiceName = %v", pv["voiceName"])
	}
	if reqBody["contents"] == nil {
		t.Errorf("contents missing")
	}
}

func TestSpeak_ConfigErrors(t *testing.T) {
	cases := []struct {
		name    string
		cfg     ProviderConfig
		req     SpeakRequest
		wantCfg bool
	}{
		{"empty text", ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, SpeakRequest{Text: " ", Voice: "alloy"}, true},
		{"anthropic format unsupported", ProviderConfig{ID: "anthropic", Format: FormatAnthropic, APIKey: "k"}, SpeakRequest{Text: "hi", Voice: "v"}, true},
		{"unknown provider", ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, SpeakRequest{Text: "hi"}, false},
		{"unauthenticated provider", ProviderConfig{ID: "nokey", Format: FormatOpenAI}, SpeakRequest{Text: "hi"}, true},
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = w.Write([]byte(`{"error":{"message":"no"}}`))
	}))
	defer srv.Close()
	old := backoffUnit
	backoffUnit = time.Millisecond
	t.Cleanup(func() { backoffUnit = old })

	op := cases[0].cfg
	op.BaseURL = srv.URL
	anthropicCfg := cases[1].cfg
	anthropicCfg.BaseURL = srv.URL
	ghost := ProviderConfig{ID: "ghost", Format: FormatOpenAI, APIKey: "x", BaseURL: srv.URL}
	nokey := cases[3].cfg
	nokey.BaseURL = srv.URL
	s := New()
	s.put(op)
	s.put(anthropicCfg)
	s.put(ghost)
	s.put(nokey)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.name == "unknown provider" {
				_, err := s.Speak("nosuch-provider", "m", tc.req)
				var ce *ConfigError
				if !errors.As(err, &ce) {
					t.Fatalf("err = %T (%v), want *ConfigError", err, err)
				}
				return
			}
			_, err := s.Speak(tc.cfg.ID, "m", tc.req)
			if err == nil {
				t.Fatalf("expected error")
			}
			var ce *ConfigError
			if !errors.As(err, &ce) {
				t.Fatalf("err = %T (%v), want *ConfigError", err, err)
			}
			if tc.name == "unknown provider" {
				return // already a ConfigError from Provider()
			}
		})
	}
}

func TestSpeak_HTTP500_TypedError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"speak exploded","type":"server_error"}}`))
	}))
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	_, err := s.Speak("openai", "tts-1", SpeakRequest{Text: "hi", Voice: "alloy"})
	var ae *APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %T (%v), want *APIError", err, err)
	}
	if ae.Status != http.StatusInternalServerError || ae.Message != "speak exploded" {
		t.Errorf("APIError = %+v", ae)
	}
}

func TestSpeak_RetriesThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) <= 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(500)
			_, _ = w.Write([]byte(`{"error":{"message":"transient"}}`))
			return
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte{9, 9})
	}))
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	res, err := s.Speak("openai", "tts-1", SpeakRequest{Text: "hi", Voice: "alloy"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if len(res.Audio) != 2 || calls.Load() != 2 {
		t.Errorf("calls=%d audio=%v", calls.Load(), res.Audio)
	}
}

// TestSpeak_MissingContentType drives a raw TCP listener that emits a 200
// response with no Content-Type header, so the well-known MIME fallback is
// asserted deterministically (no reliance on Go's body sniffing).
func TestSpeak_MissingContentType(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		body := []byte{0xff, 0xf3, 0x00, 0x01}
		fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", len(body))
		_, _ = conn.Write(body)
	}()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k", BaseURL: "http://" + ln.Addr().String()}, nil)
	res, err := s.Speak("openai", "tts-1", SpeakRequest{Text: "hello", Voice: "alloy"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if res.MIMEType != "audio/mpeg" {
		t.Errorf("MIMEType = %q, want audio/mpeg (well-known fallback)", res.MIMEType)
	}
}

// TestSpeak_OpenAI2xxJSONRejected guards against OpenAI-compatible gateways
// answering 2xx with a JSON error envelope: the SDK must surface a typed
// error, never hand JSON bytes back as audio.
func TestSpeak_OpenAI2xxJSONRejected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"error":{"message":"voice not found"}}`))
	}))
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	res, err := s.Speak("openai", "tts-1", SpeakRequest{Text: "hi", Voice: "alloy"})
	if err == nil {
		t.Fatalf("expected typed error for 2xx JSON body, got result with %d bytes", len(res.Audio))
	}
	var ae *APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %T (%v), want *APIError", err, err)
	}
	if ae.Status != http.StatusOK {
		t.Errorf("Status = %d, want 200", ae.Status)
	}
}

// TestSpeak_EmptyVoiceRejected pins ConfigError symmetry: both wire formats
// require a voice server-side, so an empty/blank Voice fails fast locally.
func TestSpeak_EmptyVoiceRejected(t *testing.T) {
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, nil)
	_, err := s.Speak("openai", "tts-1", SpeakRequest{Text: "hi", Voice: "  "})
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("openai: err = %T (%v), want *ConfigError", err, err)
	}
	gs := newTestSDK(t, ProviderConfig{ID: "gemini", Format: FormatGemini, APIKey: "k"}, nil)
	_, err = gs.Speak("gemini", "gemini-tts", SpeakRequest{Text: "hi", Voice: ""})
	if !errors.As(err, &ce) {
		t.Fatalf("gemini: err = %T (%v), want *ConfigError", err, err)
	}
}

// TestSpeak_GeminiNoAudioPartsTypedError asserts a Gemini 2xx response with
// no audio parts surfaces as a typed *APIError (HTTP 200), not a plain
// fmt.Errorf outside the error taxonomy.
func TestSpeak_GeminiNoAudioPartsTypedError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[]}}]}`))
	}))
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "gemini", Format: FormatGemini, APIKey: "k"}, srv)
	_, err := s.Speak("gemini", "gemini-tts", SpeakRequest{Text: "hi", Voice: "Kore"})
	var ae *APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %T (%v), want *APIError", err, err)
	}
	if ae.Status != http.StatusOK {
		t.Errorf("Status = %d, want 200", ae.Status)
	}
}
