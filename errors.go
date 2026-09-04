package llm

import (
	"errors"
	"fmt"
	"time"
)

// ConfigError reports SDK misuse: unknown provider id, provider without an
// API key, or malformed options. Never retryable.
type ConfigError struct{ Msg string }

func (e *ConfigError) Error() string { return "llm: config: " + e.Msg }

// APIError is a non-2xx provider response. Status 0 with a non-nil
// underlying error form is not used; network failures surface as plain
// wrapped errors. Message is the provider's own error text when parseable.
// API keys are never included in any error text.
type APIError struct {
	Provider  string
	Status    int
	Code      string
	Message   string
	Retryable bool
}

func (e *APIError) Error() string {
	s := fmt.Sprintf("llm: %s: HTTP %d", e.Provider, e.Status)
	if e.Code != "" {
		s += " (" + e.Code + ")"
	}
	if e.Message != "" {
		s += ": " + e.Message
	}
	if e.Retryable {
		s += " [retryable]"
	}
	return s
}

// RateLimitError is the final failure after persistent 429 responses.
// RetryAfter carries the last observed Retry-After hint (0 if none).
type RateLimitError struct {
	APIError
	Attempts   int
	RetryAfter time.Duration
}

// Unwrap exposes the embedded APIError so errors.As(err, *APIError)
// reaches Status/Retryable without a type switch.
func (e *RateLimitError) Unwrap() error { return &e.APIError }

func (e *RateLimitError) Error() string {
	s := fmt.Sprintf("llm: %s: rate limited after %d attempts", e.Provider, e.Attempts)
	if e.RetryAfter > 0 {
		s += fmt.Sprintf(" (retry after %s)", e.RetryAfter)
	}
	if e.Message != "" {
		s += ": " + e.Message
	}
	return s
}

// StreamAbortedError is returned by CallStream when the delta handler
// aborted generation. CallStream also returns the partial ChatResult
// assembled so far alongside this error.
type StreamAbortedError struct{ Reason error }

func (e *StreamAbortedError) Error() string {
	return fmt.Sprintf("llm: stream aborted by consumer: %v", e.Reason)
}

func (e *StreamAbortedError) Unwrap() error { return e.Reason }

// ErrIdleTimeout is returned (with the partial result) when a stream goes
// silent longer than the idle watchdog after the first delta has been
// emitted. Before the first delta, idle timeouts are retried instead.
var ErrIdleTimeout = errors.New("llm: stream idle timeout")
