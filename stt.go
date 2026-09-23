package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
	"time"
)

// ── STT (speech-to-text) ─────────────────────────────────────────────────
//
// Transcribe turns audio bytes into text. v1 supports the
// OpenAI-compatible wire format: POST {base}/audio/transcriptions,
// multipart/form-data, JSON response. Any other format (gemini,
// anthropic, …) is a ConfigError. The SDK never touches the
// filesystem: callers own the audio bytes. The chat invariants apply
// unchanged — canonical errors, keys never in error text, retry
// ladder identical to the buffered chat path.

// maxTranscribeAudioBytes bounds the audio payload accepted for one
// transcription request (matches OpenAI's documented 25MB upload limit).
// Oversized audio is rejected with a ConfigError before any network I/O.
const maxTranscribeAudioBytes = 25 << 20

// TranscribeRequest describes one speech-to-text conversion.
type TranscribeRequest struct {
	Audio    []byte // required, non-empty: raw audio bytes (caller-owned)
	Filename string // file part name for the audio (default "audio.wav")
	MIMEType string // optional content type for the audio part
	Language string // optional ISO-639-1 hint, OpenAI-compat only
	Prompt   string // optional conditioning text, OpenAI-compat only
	Format   string // optional response_format (provider default when empty)
}

// TranscribeResult carries the recognized text. Fields the provider
// omits stay zero — unknown data stays unknown (invariant #5).
type TranscribeResult struct {
	Text        string
	Model       string
	Language    string
	DurationSec float64
}

// Transcribe converts speech to text with the named provider and
// model. Buffered only, with the same retry ladder as chat.
func (s *SDK) Transcribe(ctx context.Context, providerID, model string, req TranscribeRequest) (*TranscribeResult, error) {
	if len(req.Audio) == 0 {
		return nil, &ConfigError{Msg: "transcribe request requires non-empty Audio"}
	}
	if len(req.Audio) > maxTranscribeAudioBytes {
		return nil, &ConfigError{Msg: "transcribe audio exceeds 25MB limit"}
	}
	if strings.TrimSpace(model) == "" {
		return nil, &ConfigError{Msg: "transcribe request requires a model"}
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
	return pc.transcribe(ctx, model, req)
}

// escapeQuotes escapes quotes and backslashes in a multipart filename,
// mirroring mime/multipart's internal escape.
func escapeQuotes(s string) string {
	r := strings.NewReplacer("\\", "\\\\", `"`, `\"`)
	return r.Replace(s)
}

// buildTranscribeRequest serializes the multipart body and target URL.
func (pc *providerClient) buildTranscribeRequest(model string, req TranscribeRequest) ([]byte, string, error) {
	if pc.cfg.Format != FormatOpenAI {
		return nil, "", &ConfigError{Msg: "provider format " + string(pc.cfg.Format) + " does not support transcription"}
	}
	filename := req.Filename
	if filename == "" {
		filename = "audio.wav"
	}
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mime := req.MIMEType
	if mime == "" {
		mime = "application/octet-stream"
	}
	hdr := textproto.MIMEHeader{}
	hdr.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, escapeQuotes(filename)))
	hdr.Set("Content-Type", mime)
	fh, err := mw.CreatePart(hdr)
	if err != nil {
		return nil, "", fmt.Errorf("llm: build transcription request: %w", err)
	}
	if _, err := fh.Write(req.Audio); err != nil {
		return nil, "", fmt.Errorf("llm: build transcription request: %w", err)
	}
	if err := mw.WriteField("model", model); err != nil {
		return nil, "", fmt.Errorf("llm: build transcription request: %w", err)
	}
	if req.Language != "" {
		if err := mw.WriteField("language", req.Language); err != nil {
			return nil, "", fmt.Errorf("llm: build transcription request: %w", err)
		}
	}
	if req.Prompt != "" {
		if err := mw.WriteField("prompt", req.Prompt); err != nil {
			return nil, "", fmt.Errorf("llm: build transcription request: %w", err)
		}
	}
	if req.Format != "" {
		if err := mw.WriteField("response_format", req.Format); err != nil {
			return nil, "", fmt.Errorf("llm: build transcription request: %w", err)
		}
	}
	if err := mw.Close(); err != nil {
		return nil, "", fmt.Errorf("llm: build transcription request: %w", err)
	}
	return buf.Bytes(), pc.base + "/audio/transcriptions", nil
}

// transcribe runs the STT request against one provider with retry
// semantics identical to the buffered chat path.
func (pc *providerClient) transcribe(ctx context.Context, model string, req TranscribeRequest) (*TranscribeResult, error) {
	body, url, err := pc.buildTranscribeRequest(model, req)
	if err != nil {
		return nil, err
	}
	ctype := "multipart/form-data; boundary=" + multipartBodyBoundary(body)

	var (
		lastErr error
		rateErr *APIError
		rateRA  time.Duration
	)
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		data, ra, err := pc.postMultipart(ctx, url, body, ctype)
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
		res, perr := parseTranscribeResponse(data)
		if perr != nil {
			// 2xx with an undecodable body is a provider protocol
			// failure — surface it through the typed error taxonomy
			// at the actual HTTP status (never a plain fmt.Errorf).
			return nil, &APIError{
				Provider: pc.cfg.ID,
				Status:   http.StatusOK,
				Message:  perr.Error(),
			}
		}
		res.Model = model
		return res, nil
	}
	return nil, lastErr
}

// multipartBodyBoundary extracts the boundary from a multipart writer's
// Content-Type header line.
func multipartBodyBoundary(body []byte) string {
	// The boundary is the last line of the leading preamble; cheaper and
	// stricter: parse it from the first line of the body.
	line := body
	if i := bytes.IndexByte(body, '\r'); i >= 0 {
		line = body[:i]
	} else if i := bytes.IndexByte(body, '\n'); i >= 0 {
		line = body[:i]
	}
	return strings.TrimPrefix(string(line), "--")
}

type transcribeResponse struct {
	Text     string  `json:"text"`
	Language string  `json:"language"`
	Duration float64 `json:"duration"`
}

// parseTranscribeResponse decodes a buffered transcription response.
// Both plain json ({text}) and verbose_json ({text,language,duration})
// decode through the same struct — fields the provider omits stay zero.
func parseTranscribeResponse(data []byte) (*TranscribeResult, error) {
	var resp transcribeResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("llm: decode transcription response: %w", err)
	}
	return &TranscribeResult{
		Text:        resp.Text,
		Language:    resp.Language,
		DurationSec: resp.Duration,
	}, nil
}

// postMultipart sends one transcription request and reads the full JSON
// body (capped). Mirrors postAudio but with the multipart content type.
func (pc *providerClient) postMultipart(ctx context.Context, url string, body []byte, ctype string) (data []byte, ra time.Duration, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, 0, &ConfigError{Msg: "build request: " + err.Error()}
	}
	req.Header.Set("Content-Type", ctype)
	pc.setAuthHeaders(req.Header)

	resp, err := pc.buffered().Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	data, err = io.ReadAll(io.LimitReader(resp.Body, maxResponseSize+1))
	if err != nil {
		return nil, 0, err
	}
	if len(data) > maxResponseSize {
		return nil, 0, fmt.Errorf("llm: response exceeds %d bytes", maxResponseSize)
	}
	ra = parseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, ra, pc.httpError(resp.StatusCode, data)
	}
	return data, ra, nil
}
