package llm

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ResponseMeta is provider-reported response metadata. Every field is
// provider-reported or zero — never guessed.
type ResponseMeta struct {
	ID        string // provider response/message id; "" when not reported
	Model     string // model reported by the provider; "" when not reported
	RequestID string // from the x-request-id family of response headers
	RateLimit RateLimit
}

// RateLimit is the provider-reported rate-limit headroom.
type RateLimit struct {
	Reported          bool // true when at least one field below was parsed
	RemainingRequests int
	RemainingTokens   int
	ResetRequests     time.Duration // time until the window resets; 0 = unknown
	ResetTokens       time.Duration
}

var requestIDHeaders = []string{"X-Request-Id", "Request-Id", "X-Goog-Request-Id", "X-Amzn-Requestid"}

// parseResponseMeta extracts RequestID and RateLimit from response headers
// (ID and Model come from the body).
func parseResponseMeta(h http.Header) ResponseMeta {
	var m ResponseMeta
	for _, k := range requestIDHeaders {
		if v := strings.TrimSpace(h.Get(k)); v != "" {
			m.RequestID = v
			break
		}
	}
	rl := &m.RateLimit
	rl.RemainingRequests = headerInt(h, rl, "X-Ratelimit-Remaining-Requests", "Anthropic-Ratelimit-Requests-Remaining")
	rl.RemainingTokens = headerInt(h, rl, "X-Ratelimit-Remaining-Tokens", "Anthropic-Ratelimit-Tokens-Remaining")
	rl.ResetRequests = headerReset(h, rl, "X-Ratelimit-Reset-Requests", "Anthropic-Ratelimit-Requests-Reset")
	rl.ResetTokens = headerReset(h, rl, "X-Ratelimit-Reset-Tokens", "Anthropic-Ratelimit-Tokens-Reset")
	return m
}

// headerInt parses the first parseable of the named headers.
func headerInt(h http.Header, rl *RateLimit, names ...string) int {
	for _, n := range names {
		if v, err := strconv.Atoi(strings.TrimSpace(h.Get(n))); err == nil && v >= 0 {
			rl.Reported = true
			return v
		}
	}
	return 0
}

// headerReset parses a Go-style duration, plain seconds, or an RFC3339
// timestamp (duration from now, floored at zero).
func headerReset(h http.Header, rl *RateLimit, names ...string) time.Duration {
	for _, n := range names {
		v := strings.TrimSpace(h.Get(n))
		if v == "" {
			continue
		}
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			rl.Reported = true
			return d
		}
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f < 1e9 {
			rl.Reported = true
			return time.Duration(f * float64(time.Second))
		}
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			rl.Reported = true
			return max(time.Until(t), 0)
		}
	}
	return 0
}

// withHeaderMeta overlays header-derived metadata onto r (nil-safe).
func (r *ChatResult) withHeaderMeta(h http.Header) {
	if r == nil {
		return
	}
	hm := parseResponseMeta(h)
	r.Meta.RequestID, r.Meta.RateLimit = hm.RequestID, hm.RateLimit
}
