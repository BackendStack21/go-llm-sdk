package llm

import (
	"context"
	"errors"
	"fmt"
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
	// maxRetryAfter caps how long a server's Retry-After is honored. A
	// pathological or hostile value (e.g. "Retry-After: 86400") must not
	// wedge a call for hours; context cancellation can still break the
	// wait sooner.
	maxRetryAfter = 120 * time.Second
)

// retryableStatus reports whether an HTTP status should be retried.
// 529 is Anthropic's "overloaded" status. 520–524 are Cloudflare-origin
// incidents (CF-fronted providers emit these during origin hiccups).
func retryableStatus(code int) bool {
	switch code {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout, 529:
		return true
	}
	return code >= 520 && code <= 524
}

// parseRetryAfter parses a Retry-After header value: either delay-seconds
// or an HTTP-date. Returns 0 when unparseable. The result is capped at
// maxRetryAfter.
func parseRetryAfter(v string, now time.Time) time.Duration {
	if v == "" {
		return 0
	}
	var d time.Duration
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0
		}
		d = time.Duration(secs) * time.Second
	} else if t, err := http.ParseTime(v); err == nil {
		if delta := t.Sub(now); delta > 0 {
			d = delta
		}
	} else {
		return 0
	}
	if d > maxRetryAfter {
		d = maxRetryAfter
	}
	return d
}

// backoffUnit is the exponential base; a package var so tests can shrink
// retries to microseconds.
var backoffUnit = time.Second

// backoffDelay returns the capped exponential backoff for attempt n
// (1-based retry number) with ±20% jitter. The cap applies after jitter,
// so the result never exceeds maxRetryBackoff.
func backoffDelay(attempt int) time.Duration {
	return backoffDelayCapped(attempt, maxRetryBackoff)
}

// backoffDelayCapped is backoffDelay with an explicit cap.
func backoffDelayCapped(attempt int, maxBackoff time.Duration) time.Duration {
	if attempt > 20 {
		// 2^20 units already dwarfs any sane cap; a larger shift would
		// overflow time.Duration (a long MaxAttempts ladder must never
		// wrap to a zero or negative backoff).
		attempt = 20
	}
	if attempt < 0 {
		return 0
	}
	d := time.Duration(1<<uint(attempt)) * backoffUnit
	jitter := time.Duration(float64(d) * retryJitterFactor)
	if jitter > 0 {
		d += time.Duration(rand.Int64N(int64(2*jitter))) - jitter
	}
	if d < 0 {
		d = 0
	}
	if d > maxBackoff {
		d = maxBackoff
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

// RetryPolicy configures the retry ladder shared by chat (buffered and
// streaming), speech, transcription and embeddings. Zero fields keep the
// defaults.
type RetryPolicy struct {
	// MaxAttempts is the total number of attempts including the first
	// (default 8). 1 disables retries.
	MaxAttempts int
	// MaxBackoff caps one exponential backoff step (default 30s).
	// Retry-After from the server is honored independently (capped at
	// 120s).
	MaxBackoff time.Duration
}

// attempts resolves MaxAttempts (non-positive → default).
func (p RetryPolicy) attempts() int {
	if p.MaxAttempts > 0 {
		return p.MaxAttempts
	}
	return maxRetries + 1
}

// backoffCap resolves MaxBackoff (non-positive → default).
func (p RetryPolicy) backoffCap() time.Duration {
	if p.MaxBackoff > 0 {
		return p.MaxBackoff
	}
	return maxRetryBackoff
}

// terminalError marks an attempt failure that must never be retried
// (request assembly or response parsing): withRetry unwraps and returns it.
type terminalError struct{ err error }

func (e *terminalError) Error() string { return e.err.Error() }

// terminal wraps err as non-retryable (nil stays nil).
func terminal(err error) error {
	if err == nil {
		return nil
	}
	return &terminalError{err: err}
}

// withRetry runs op under the client's retry policy — the single buffered
// ladder shared by chat, speech, transcription and embeddings. op returns
// the Retry-After hint alongside its error. learn, when non-nil, is offered
// every non-retryable *APIError and reports whether a learn-once fallback
// was engaged, in which case the request is retried immediately.
//
// Classification: 429 billing exhaustion fails fast; other 429s back off
// and surface as *RateLimitError when the ladder or the context ends;
// retryable statuses and transport errors back off; anything else (and any
// terminal error) is returned as is.
func (pc *providerClient) withRetry(ctx context.Context, op func() (time.Duration, error), learn func(*APIError) bool) error {
	n := pc.opts.retry.attempts()
	var (
		rateErr *APIError // last 429
		rateRA  time.Duration
	)
	for attempt := 0; attempt < n; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		ra, err := op()
		if err == nil {
			return nil
		}
		var te *terminalError
		if errors.As(err, &te) {
			return te.err
		}
		var ce *ConfigError
		if errors.As(err, &ce) {
			return err // the request cannot be built: retrying cannot help
		}
		last := attempt == n-1
		var apiErr *APIError
		if !errors.As(err, &apiErr) {
			// Transport error — retryable.
			if cerr := ctx.Err(); cerr != nil {
				// Interrupted by deadline/cancel, not exhaustion.
				return cerr
			}
			if last {
				return fmt.Errorf("llm: retry exhausted (%d attempts): %w", n, err)
			}
			if !retrySleep(ctx, pc.retryDelay(0, attempt)) {
				return ctx.Err()
			}
			continue
		}
		switch {
		case apiErr.Status == http.StatusTooManyRequests:
			if billingExhausted(apiErr) {
				// Permanent: only a recharge fixes this.
				return apiErr
			}
			rateErr, rateRA = apiErr, ra
			if !last {
				if !retrySleep(ctx, pc.retryDelay(ra, attempt)) {
					// The sleep died with the context (deadline or cancel);
					// still surface the 429 — the caller needs
					// Status/RetryAfter to plan the retry.
					return &RateLimitError{APIError: *rateErr, Attempts: attempt + 1, RetryAfter: rateRA}
				}
				continue
			}
		case apiErr.Retryable && !last:
			if !retrySleep(ctx, pc.retryDelay(ra, attempt)) {
				return ctx.Err()
			}
			continue
		case learn != nil && learn(apiErr):
			// Learn-once fallback engaged: retry immediately.
			if !last {
				continue
			}
			return apiErr
		}
		if rateErr != nil && !apiErr.Retryable && apiErr.Status != http.StatusTooManyRequests {
			// A definitive failure after earlier 429s.
			return apiErr
		}
		if rateErr != nil {
			return &RateLimitError{APIError: *rateErr, Attempts: attempt + 1, RetryAfter: rateRA}
		}
		return apiErr
	}
	return ctx.Err() // unreachable: attempts() >= 1
}
