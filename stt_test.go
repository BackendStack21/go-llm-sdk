package llm

import (
	"bytes"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// sttRecord is what a test server captures from one Transcribe request.
type sttRecord struct {
	path     string
	auth     string
	ct       string
	body     []byte
	boundary string
}

// sttParse splits a captured multipart body into form values and the
// uploaded file part.
func sttParse(t *testing.T, rec sttRecord) (map[string]string, *multipart.FileHeader) {
	t.Helper()
	mr := multipart.NewReader(bytes.NewReader(rec.body), rec.boundary)
	form, err := mr.ReadForm(1 << 20)
	if err != nil {
		t.Fatalf("parse multipart body: %v", err)
	}
	vals := map[string]string{}
	for k, v := range form.Value {
		if len(v) > 0 {
			vals[k] = v[0]
		}
	}
	var filePart *multipart.FileHeader
	if fh := form.File["file"]; len(fh) > 0 {
		filePart = fh[0]
	}
	return vals, filePart
}

// newSTTServer records one request, then answers with the given status,
// content type, and body.
func newSTTServer(status int, ctype, body string) (*httptest.Server, *sttRecord) {
	rec := &sttRecord{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.path = r.URL.Path
		rec.auth = r.Header.Get("Authorization")
		rec.ct = r.Header.Get("Content-Type")
		if i := strings.Index(rec.ct, "boundary="); i >= 0 {
			rec.boundary = rec.ct[i+len("boundary="):]
		}
		rec.body, _ = io.ReadAll(r.Body)
		if ctype != "" {
			w.Header().Set("Content-Type", ctype)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	return srv, rec
}

// TestTranscribe_MIMETypeHonored asserts the audio part's Content-Type
// comes from TranscribeRequest.MIMEType when set (RED: field was dead —
// CreateFormFile always wrote application/octet-stream).
func TestTranscribe_MIMETypeHonored(t *testing.T) {
	srv, rec := newSTTServer(http.StatusOK, "application/json", `{"text":"hi"}`)
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	if _, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{
		Audio:    []byte{1, 2, 3},
		Filename: "probe.mp3",
		MIMEType: "audio/mpeg",
	}); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	_, fp := sttParse(t, *rec)
	if fp == nil {
		t.Fatal("no file part")
	}
	if got := fp.Header.Get("Content-Type"); got != "audio/mpeg" {
		t.Errorf("file part Content-Type = %q, want audio/mpeg", got)
	}
}

// TestTranscribe_MIMETypeDefault asserts the audio part falls back to
// application/octet-stream when MIMEType is unset.
func TestTranscribe_MIMETypeDefault(t *testing.T) {
	srv, rec := newSTTServer(http.StatusOK, "application/json", `{"text":"hi"}`)
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	if _, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{
		Audio:    []byte{1, 2, 3},
		Filename: "probe.mp3",
	}); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	_, fp := sttParse(t, *rec)
	if fp == nil {
		t.Fatal("no file part")
	}
	if got := fp.Header.Get("Content-Type"); got != "application/octet-stream" {
		t.Errorf("file part Content-Type = %q, want application/octet-stream", got)
	}
}

// TestTranscribe_AudioTooLarge asserts oversized audio is rejected before
// any network round trip (RED: no request-side cap existed).
func TestTranscribe_AudioTooLarge(t *testing.T) {
	srv, _ := newSTTServer(http.StatusOK, "application/json", `{"text":"hi"}`)
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	_, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{
		Audio:    make([]byte, maxTranscribeAudioBytes+1),
		Filename: "big.mp3",
	})
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %T (%v), want *ConfigError", err, err)
	}
}

func TestTranscribe_OpenAI(t *testing.T) {
	audio := []byte{0xff, 0xf3, 0x00, 0x01, 0x02}
	srv, rec := newSTTServer(http.StatusOK, "application/json", `{"text":"hello world"}`)
	defer srv.Close()

	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k-secret"}, srv)
	res, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{
		Audio:    audio,
		Filename: "probe.mp3",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if rec.path != "/audio/transcriptions" {
		t.Errorf("path = %q, want /audio/transcriptions", rec.path)
	}
	if rec.auth != "Bearer k-secret" {
		t.Errorf("bearer auth not sent correctly: %q", rec.auth)
	}
	if !strings.HasPrefix(rec.ct, "multipart/form-data") {
		t.Errorf("content type = %q, want multipart/form-data", rec.ct)
	}
	vals, fp := sttParse(t, *rec)
	if fp == nil {
		t.Fatal("no file part in request")
	}
	if fp.Filename != "probe.mp3" {
		t.Errorf("file filename = %q, want probe.mp3", fp.Filename)
	}
	f, err := fp.Open()
	if err != nil {
		t.Fatalf("open file part: %v", err)
	}
	defer f.Close()
	got, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("read file part: %v", err)
	}
	if !bytes.Equal(got, audio) {
		t.Errorf("file bytes = %v, want %v", got, audio)
	}
	if vals["model"] != "whisper-1" {
		t.Errorf("model = %q, want whisper-1", vals["model"])
	}
	for _, k := range []string{"language", "prompt", "response_format"} {
		if v, ok := vals[k]; ok {
			t.Errorf("field %q = %q, want omitted when empty", k, v)
		}
	}
	if res.Text != "hello world" {
		t.Errorf("Text = %q", res.Text)
	}
	if res.Model != "whisper-1" {
		t.Errorf("Model = %q, want whisper-1", res.Model)
	}
}

func TestTranscribe_OptionalFields(t *testing.T) {
	srv, rec := newSTTServer(http.StatusOK, "application/json",
		`{"text":"bonjour","language":"fr","duration":1.5}`)
	defer srv.Close()

	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	res, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{
		Audio:    []byte{1, 2, 3},
		Filename: "a.wav",
		Language: "fr",
		Prompt:   "greeting",
		Format:   "verbose_json",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	vals, _ := sttParse(t, *rec)
	if vals["language"] != "fr" || vals["prompt"] != "greeting" || vals["response_format"] != "verbose_json" {
		t.Errorf("optional fields = %v", vals)
	}
	if res.Text != "bonjour" || res.Language != "fr" || res.DurationSec != 1.5 {
		t.Errorf("result = %+v", res)
	}
}

func TestTranscribe_PlainJSONZeroValues(t *testing.T) {
	srv, _ := newSTTServer(http.StatusOK, "application/json", `{"text":"only text"}`)
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	res, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if res.Text != "only text" || res.Language != "" || res.DurationSec != 0 {
		t.Errorf("result = %+v, want zero values for absent fields", res)
	}
}

func TestTranscribe_HTTP500JSONEnvelope(t *testing.T) {
	srv, _ := newSTTServer(http.StatusInternalServerError, "application/json",
		`{"error":{"message":"boom","type":"server_error"}}`)
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	_, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}})
	var ae *APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %T (%v), want *APIError", err, err)
	}
	if ae.Status != http.StatusInternalServerError {
		t.Errorf("Status = %d, want 500", ae.Status)
	}
}

func TestTranscribe_2xxNonJSONBody(t *testing.T) {
	srv, _ := newSTTServer(http.StatusOK, "text/plain", "not json at all")
	defer srv.Close()
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	_, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}})
	var ae *APIError
	if !errors.As(err, &ae) {
		t.Fatalf("err = %T (%v), want *APIError", err, err)
	}
	if ae.Status != http.StatusOK {
		t.Errorf("Status = %d, want 200", ae.Status)
	}
}

func TestTranscribe_ConfigErrors(t *testing.T) {
	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k-secret"}
	cases := []struct {
		name   string
		sdkCfg ProviderConfig
		call   func(s *SDK) error
	}{
		{"empty audio", cfg, func(s *SDK) error {
			_, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Filename: "a.mp3"})
			return err
		}},
		{"empty model", cfg, func(s *SDK) error {
			_, err := s.Transcribe(t.Context(), "openai", "", TranscribeRequest{Audio: []byte{1}})
			return err
		}},
		{"unknown provider", cfg, func(s *SDK) error {
			_, err := s.Transcribe(t.Context(), "nope", "whisper-1", TranscribeRequest{Audio: []byte{1}})
			return err
		}},
		{"unauthenticated", ProviderConfig{ID: "openai", Format: FormatOpenAI}, func(s *SDK) error {
			_, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}})
			return err
		}},
		{"non-openai format", ProviderConfig{ID: "anthropic", Format: FormatAnthropic, APIKey: "k"}, func(s *SDK) error {
			_, err := s.Transcribe(t.Context(), "anthropic", "whisper-1", TranscribeRequest{Audio: []byte{1}})
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newTestSDK(t, tc.sdkCfg, nil)
			err := tc.call(s)
			var ce *ConfigError
			if !errors.As(err, &ce) {
				t.Fatalf("err = %T (%v), want *ConfigError", err, err)
			}
			if strings.Contains(err.Error(), "k-secret") {
				t.Errorf("error leaks api key: %v", err)
			}
		})
	}
}

func TestTranscribe_ResponseSizeCap(t *testing.T) {
	srv, _ := newSTTServer(http.StatusOK, "application/json", string(make([]byte, 100)))
	defer srv.Close()
	// Respond with more than the cap regardless of the cap value.
	srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(make([]byte, maxResponseSize+2))
	})
	s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
	_, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}})
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("err = %v, want size-cap error", err)
	}
}
