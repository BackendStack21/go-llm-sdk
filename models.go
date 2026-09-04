package llm

import (
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

// listModels runs discovery for one provider with light retries (model
// listings are less critical than chat: 3 attempts).
func (pc *providerClient) listModels(ctx context.Context) ([]Model, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var (
			models []Model
			err    error
		)
		switch pc.cfg.Format {
		case FormatAnthropic:
			models, err = listModelsAnthropic(ctx, pc)
		case FormatGemini:
			models, err = listModelsGemini(ctx, pc)
		default:
			models, err = listModelsOpenAI(ctx, pc)
		}
		if err == nil {
			return models, nil
		}
		var apiErr *APIError
		if errors.As(err, &apiErr) && !apiErr.Retryable {
			return nil, err
		}
		lastErr = err
		if !retrySleep(ctx, backoffDelay(attempt+1)) {
			break
		}
	}
	return nil, fmt.Errorf("llm: list models failed: %w", lastErr)
}

// get performs a GET with auth headers and a bounded body read.
func (pc *providerClient) get(ctx context.Context, url string) ([]byte, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, &ConfigError{Msg: "build request: " + err.Error()}
	}
	req.Header.Set("Accept", "application/json")
	pc.setAuthHeaders(req.Header)

	resp, err := pc.buffered().Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxModelsResponseSize+1))
	if err != nil {
		return nil, 0, err
	}
	if len(data) > maxModelsResponseSize {
		return nil, 0, fmt.Errorf("llm: models response exceeds %d bytes", maxModelsResponseSize)
	}
	ra := parseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return data, ra, pc.httpError(resp.StatusCode, data)
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
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("llm: parse models response: %w", err)
	}
	entries := r.Data
	if len(entries) == 0 {
		entries = r.Models
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
