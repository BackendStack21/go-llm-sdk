package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ── TTS (text-to-speech) ─────────────────────────────────────────────────
//
// Speak turns text into audio bytes. Two wire formats are supported:
//   - OpenAI-compatible: POST {base}/audio/speech, binary audio response
//   - Gemini: generateContent with AUDIO response modality, base64 inlineData
//
// Any other format (anthropic, …) is a ConfigError. The SDK never
// transcodes: it returns the provider's bytes plus the MIME type the
// provider declared (or a well-known default when the provider omits it).
// API keys never appear in errors — the invariants of chat apply unchanged.

// SpeakRequest describes one text-to-speech conversion.
type SpeakRequest struct {
	Text   string  // required, non-empty
	Voice  string  // provider voice id (no fallback tables — invariant #5)
	Format string  // container for OpenAI-compat providers ("mp3" default)
	Speed  float64 // optional playback speed, OpenAI-compat only (0 = omit)
}

// SpeakResult carries the synthesized audio.
type SpeakResult struct {
	Audio    []byte // raw audio bytes as the provider returned them
	Model    string // model that produced the audio
	MIMEType string // provider-declared (or well-known default) MIME type
}

// Speak synthesizes speech with the named provider and model. Buffered
// only — no streaming in v1 — with the same retry ladder as chat.
func (s *SDK) Speak(providerID, model string, req SpeakRequest) (*SpeakResult, error) {
	if strings.TrimSpace(req.Text) == "" {
		return nil, &ConfigError{Msg: "speak request requires non-empty Text"}
	}
	if strings.TrimSpace(model) == "" {
		return nil, &ConfigError{Msg: "speak request requires a model"}
	}
	if strings.TrimSpace(req.Voice) == "" {
		return nil, &ConfigError{Msg: "speak request requires a Voice (both supported wire formats require one server-side)"}
	}
	p, err := s.Provider(providerID)
	if err != nil {
		return nil, err
	}
	if !p.Authenticated() {
		return nil, &ConfigError{Msg: providerID + " has no API key (set " + strings.ToUpper(providerID) + "_API_KEY or use WithAPIKey)"}
	}
	if p.invalid {
		return nil, &ConfigError{Msg: providerID + " has an invalid configuration"}
	}
	pc := newProviderClient(p.cfg, newBufferedHTTP(s.rt, s.timeout), nil)
	return pc.speak(context.Background(), model, req)
}

// buildSpeakRequest dispatches format-specific serialization. The third
// return is the well-known MIME type used when the provider omits
// Content-Type ("" for formats with no such default).
func (pc *providerClient) buildSpeakRequest(model string, req SpeakRequest) ([]byte, string, string, error) {
	switch pc.cfg.Format {
	case FormatGemini:
		return buildGeminiSpeakRequest(pc.base, model, req)
	case FormatOpenAI:
		format := req.Format
		if format == "" {
			format = "mp3"
		}
		body := map[string]any{
			"model":           model,
			"input":           req.Text,
			"voice":           req.Voice,
			"response_format": format,
		}
		if req.Speed != 0 {
			body["speed"] = req.Speed
		}
		b, err := json.Marshal(body)
		return b, pc.base + "/audio/speech", "audio/mpeg", err
	default:
		return nil, "", "", &ConfigError{Msg: "provider format " + string(pc.cfg.Format) + " does not support speech"}
	}
}

// speak runs the TTS request against one provider with retry semantics
// identical to the buffered chat path (binary body, no SSE).
func (pc *providerClient) speak(ctx context.Context, model string, req SpeakRequest) (*SpeakResult, error) {
	body, url, mimeHint, err := pc.buildSpeakRequest(model, req)
	if err != nil {
		return nil, err
	}
	gemini := pc.cfg.Format == FormatGemini

	var (
		lastErr error
		rateErr *APIError
		rateRA  time.Duration
	)
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		data, ctype, ra, err := pc.postAudio(ctx, url, body)
		if err != nil {
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				switch {
				case apiErr.Status == http.StatusTooManyRequests && billingExhausted(apiErr):
					return nil, apiErr
				case apiErr.Status == http.StatusTooManyRequests:
					rateErr, rateRA, lastErr = apiErr, ra, apiErr
					if attempt < maxRetries {
						if !retrySleep(ctx, retryDelay(ra, attempt)) {
							return nil, &RateLimitError{APIError: *rateErr, Attempts: attempt + 1, RetryAfter: rateRA}
						}
						continue
					}
				case apiErr.Retryable && attempt < maxRetries:
					lastErr = apiErr
					if !retrySleep(ctx, retryDelay(ra, attempt)) {
						return nil, ctx.Err()
					}
					continue
				}
				if rateErr != nil && !apiErr.Retryable && apiErr.Status != http.StatusTooManyRequests {
					return nil, apiErr
				}
				if rateErr != nil {
					return nil, &RateLimitError{APIError: *rateErr, Attempts: attempt + 1, RetryAfter: rateRA}
				}
				return nil, apiErr
			}
			// Transport error — retryable.
			lastErr = err
			if attempt < maxRetries {
				if !retrySleep(ctx, retryDelay(0, attempt)) {
					return nil, ctx.Err()
				}
				continue
			}
			return nil, fmt.Errorf("llm: retry exhausted (%d attempts): %w", maxRetries+1, err)
		}
		if gemini {
			audio, mime, perr := parseGeminiSpeakResponse(data)
			if perr != nil {
				// 2xx with no audio parts is a provider protocol
				// failure — surface it through the typed error
				// taxonomy (APIError at the actual HTTP status).
				return nil, &APIError{
					Provider: pc.cfg.ID,
					Status:   http.StatusOK,
					Message:  perr.Error(),
				}
			}
			data, ctype = audio, mime
		} else if strings.HasPrefix(strings.TrimSpace(ctype), "application/json") {
			// OpenAI-compatible gateways sometimes answer 2xx with a
			// JSON error envelope. Never hand JSON bytes back as
			// audio: parse the envelope through the shared httpError
			// path.
			return nil, pc.httpError(http.StatusOK, data)
		}
		return &SpeakResult{
			Audio:    data,
			Model:    model,
			MIMEType: resolveAudioMIME(ctype, mimeHint),
		}, nil
	}
	return nil, lastErr
}

// resolveAudioMIME prefers the provider's Content-Type; when absent it
// falls back to the wire format's well-known default (no guessing beyond
// that default — invariant #5).
func resolveAudioMIME(ctype, fallback string) string {
	ctype = strings.TrimSpace(ctype)
	if ctype != "" {
		return ctype
	}
	return fallback
}

// postAudio sends one TTS request and reads the full binary body (capped).
// Unlike post it captures the response Content-Type and never assumes JSON.
func (pc *providerClient) postAudio(ctx context.Context, url string, body []byte) (data []byte, ctype string, ra time.Duration, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, "", 0, &ConfigError{Msg: "build request: " + err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	pc.setAuthHeaders(req.Header)

	resp, err := pc.buffered().Do(req)
	if err != nil {
		return nil, "", 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	ctype = resp.Header.Get("Content-Type")
	data, err = io.ReadAll(io.LimitReader(resp.Body, maxResponseSize+1))
	if err != nil {
		return nil, "", 0, err
	}
	if len(data) > maxResponseSize {
		return nil, "", 0, fmt.Errorf("llm: audio response exceeds %d bytes", maxResponseSize)
	}
	ra = parseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", ra, pc.httpError(resp.StatusCode, data)
	}
	return data, ctype, ra, nil
}

// ── Gemini TTS wire types ────────────────────────────────────────────────

type geminiSpeakRequest struct {
	Contents         []gmContent           `json:"contents"`
	GenerationConfig geminiGenerationAudio `json:"generationConfig"`
}

type geminiGenerationAudio struct {
	ResponseModalities []string        `json:"responseModalities"`
	SpeechConfig       geminiSpeechCfg `json:"speechConfig"`
}

type geminiSpeechCfg struct {
	VoiceConfig geminiVoiceCfg `json:"voiceConfig"`
}

type geminiVoiceCfg struct {
	PrebuiltVoiceConfig geminiPrebuiltVoice `json:"prebuiltVoiceConfig"`
}

type geminiPrebuiltVoice struct {
	VoiceName string `json:"voiceName"`
}

type geminiSpeakResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				InlineData struct {
					MimeType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// buildGeminiSpeakRequest serializes a native Gemini TTS request via
// generateContent: AUDIO response modality + prebuilt voice config. The
// response MIME comes from inlineData.mimeType, so there is no
// well-known fallback.
func buildGeminiSpeakRequest(base, model string, req SpeakRequest) ([]byte, string, string, error) {
	greq := geminiSpeakRequest{
		Contents: []gmContent{{Role: "user", Parts: []gmPart{{Text: req.Text}}}},
		GenerationConfig: geminiGenerationAudio{
			ResponseModalities: []string{"AUDIO"},
			SpeechConfig:       geminiSpeechCfg{VoiceConfig: geminiVoiceCfg{PrebuiltVoiceConfig: geminiPrebuiltVoice{VoiceName: req.Voice}}},
		},
	}
	b, err := json.Marshal(greq)
	return b, fmt.Sprintf("%s/v1beta/models/%s:generateContent", base, model), "", err
}

// parseGeminiSpeakResponse decodes a buffered Gemini TTS response.
func parseGeminiSpeakResponse(data []byte) (audio []byte, mime string, err error) {
	var resp geminiSpeakResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, "", fmt.Errorf("llm: decode gemini speech response: %w", err)
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, "", fmt.Errorf("llm: gemini speech response contained no audio parts")
	}
	inline := resp.Candidates[0].Content.Parts[0].InlineData
	if inline.Data == "" {
		return nil, "", fmt.Errorf("llm: gemini speech response contained no audio data")
	}
	audio, err = base64.StdEncoding.DecodeString(inline.Data)
	if err != nil {
		return nil, "", fmt.Errorf("llm: decode gemini speech audio: %w", err)
	}
	return audio, inline.MimeType, nil
}
