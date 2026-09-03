package llm

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strconv"
	"time"
)

// Retry policy, shared by every format. Eight attempts total (an initial
// attempt plus seven retries) with exponential backoff capped at 30s and
// ±20% jitter. Retry-After (seconds or HTTP-date) overrides backoff.
// Context cancellation is honored between attempts.
const (
	maxRetries        = 7
	maxRetryBackoff   = 30 * time.Second
	retryJitterFactor = 0.2
)

// retryableStatus reports whether an HTTP status should be retried.
// 529 is Anthropic's "overloaded" status.
func retryableStatus(code int) bool {
	switch code {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout, 529:
		return true
	}
	return false
}

// parseRetryAfter parses a Retry-After header value: either delay-seconds
// or an HTTP-date. Returns 0 when unparseable.
func parseRetryAfter(v string, now time.Time) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// backoffUnit is the exponential base; a package var so tests can shrink
// retries to microseconds.
var backoffUnit = time.Second

// backoffDelay returns the capped exponential backoff for attempt n
// (1-based retry number) with ±20% jitter. The cap applies after jitter,
// so the result never exceeds maxRetryBackoff.
func backoffDelay(attempt int) time.Duration {
	d := time.Duration(1<<uint(attempt)) * backoffUnit
	jitter := time.Duration(float64(d) * retryJitterFactor)
	if jitter > 0 {
		d += time.Duration(rand.Int64N(int64(2*jitter))) - jitter
	}
	if d < 0 {
		d = 0
	}
	if d > maxRetryBackoff {
		d = maxRetryBackoff
	}
	return d
}

// retrySleep waits d (or ctx cancellation) and reports whether the caller
// should keep going.
func retrySleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
