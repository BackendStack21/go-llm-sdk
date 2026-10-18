package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Dynamic model discovery. ListModels hits the provider's models endpoint
// and returns exactly what the account can access — the SDK never falls
// back to static model tables. Fields the provider does not report stay
// zero (unknown).

// defaultListAttempts is the attempt budget when no RetryPolicy is
// configured (model listings are less critical than chat).
const defaultListAttempts = 3

// listModels runs discovery for one provider through the shared retry
// ladder. WithRetryPolicy sets the attempt count and backoff cap (3 attempts
// when unconfigured); the request timeout is a whole-call budget; Retry-After
// is honored.
func (pc *providerClient) listModels(ctx context.Context) ([]Model, error) {
	ctx = pc.withAttempts(ctx, "models", "")
	ctx, cancel := context.WithTimeout(ctx, pc.requestTimeout())
	defer cancel()

	n := pc.opts.retry.MaxAttempts
	if n <= 0 {
		n = defaultListAttempts
	}
	var models []Model
	err := pc.withRetryN(ctx, n, func() (time.Duration, error) {
		var err error
		switch pc.cfg.Format {
		case FormatAnthropic:
			models, err = listModelsAnthropic(ctx, pc)
		case FormatGemini:
			models, err = listModelsGemini(ctx, pc)
		default:
			models, err = listModelsOpenAI(ctx, pc)
		}
		if err == nil {
			return 0, nil
		}
		var re *retryAfterError
		var ra time.Duration
		if errors.As(err, &re) {
			ra = re.ra
		}
		if errors.Is(err, ErrModelListTruncated) {
			err = terminal(err)
		}
		return ra, err
	}, nil)
	if err != nil {
		var apiErr *APIError
		var rl *RateLimitError
		if (errors.As(err, &apiErr) && apiErr.Retryable) || errors.As(err, &rl) {
			return nil, fmt.Errorf("llm: list models failed: %w", err) // exhausted retries
		}
		return nil, err
	}
	return models, nil
}

// retryAfterError carries a response's Retry-After hint alongside its error
// out of the per-format listing functions.
type retryAfterError struct {
	err error
	ra  time.Duration
}

func (e *retryAfterError) Error() string { return e.err.Error() }
func (e *retryAfterError) Unwrap() error { return e.err }

// cloneModels deep-copies a listing so callers never alias cached state.
func cloneModels(in []Model) []Model {
	out := make([]Model, len(in))
	copy(out, in)
	for i := range out {
		if out[i].Capabilities != nil {
			out[i].Capabilities = append([]string(nil), out[i].Capabilities...)
		}
	}
	return out
}

// get performs a GET with auth headers and a bounded body read.
func (pc *providerClient) get(ctx context.Context, url string) ([]byte, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, &ConfigError{Msg: "build request: " + err.Error()}
	}
	req.Header.Set("Accept", "application/json")
	pc.setHeaders(req.Header, nil)

	resp, err := doObserved(ctx, pc.buffered(), req)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxModelsResponseSize+1))
	if err != nil {
		return nil, 0, err
	}
	if len(data) > maxModelsResponseSize {
		return nil, 0, terminal(fmt.Errorf("llm: models response exceeds %d bytes", maxModelsResponseSize))
	}
	ra := parseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return data, ra, &retryAfterError{err: pc.httpError(resp.StatusCode, data), ra: ra}
	}
	return data, ra, nil
}

// ── OpenAI-format model listing ──────────────────────────────────────────

type oaModelEntry struct {
	ID            string `json:"id"`
	CreatedAt     int64  `json:"created"`          // unix seconds
	ContextLength int    `json:"context_length"`   // OpenRouter, Together, some providers
	MaxContext    int    `json:"max_context"`      // fallback field name
	MaxInput      int    `json:"max_input_tokens"` // common alternative
}

type oaModelsResponse struct {
	Data   []oaModelEntry `json:"data"`
	Models []oaModelEntry `json:"models"` // some providers use this wrapper
}

func listModelsOpenAI(ctx context.Context, pc *providerClient) ([]Model, error) {
	data, _, err := pc.get(ctx, pc.base+"/models")
	if err != nil {
		return nil, err
	}
	var r oaModelsResponse
	var entries []oaModelEntry
	if trimmed := bytes.TrimSpace(data); len(trimmed) > 0 && trimmed[0] == '[' {
		// Bare top-level array (Together AI).
		if err := json.Unmarshal(trimmed, &entries); err != nil {
			return nil, terminal(fmt.Errorf("llm: parse models response: %w", err))
		}
	} else {
		if err := json.Unmarshal(data, &r); err != nil {
			return nil, terminal(fmt.Errorf("llm: parse models response: %w", err))
		}
		entries = r.Data
		if len(entries) == 0 {
			entries = r.Models
		}
	}
	out := make([]Model, 0, len(entries))
	for _, m := range entries {
		mm := Model{ID: m.ID}
		if m.CreatedAt > 0 {
			mm.CreatedAt = time.Unix(m.CreatedAt, 0).UTC()
		}
		// Best-effort context window from whichever field the provider
		// names; zero (unknown) when none report it.
		switch {
		case m.ContextLength > 0:
			mm.ContextWindow = m.ContextLength
		case m.MaxContext > 0:
			mm.ContextWindow = m.MaxContext
		case m.MaxInput > 0:
			mm.ContextWindow = m.MaxInput
		}
		out = append(out, mm)
	}
	return out, nil
}

// maxModelPages bounds a paginated listing (100 models per page).
const maxModelPages = 100

// ErrModelListTruncated reports a model listing longer than the SDK's page
// cap; the SDK never returns a silently truncated listing.
var ErrModelListTruncated = errors.New("llm: model listing exceeds the page cap")
