package llm

import (
	"context"
	"net/http"
	"strings"
	"time"
)

// AttemptEvent describes one HTTP attempt of an SDK call. See
// WithAttemptObserver.
type AttemptEvent struct {
	Provider   string
	Operation  string        // "chat" | "stream" | "embed" | "speak" | "transcribe" | "models"
	Model      string        // "" for models
	Attempt    int           // 1-based, per entry-point call
	Status     int           // HTTP status; 0 on transport error
	Err        error         // nil on success; API keys are never included
	Latency    time.Duration // request start -> response headers received
	RetryDelay time.Duration // delay before the next attempt; 0 when final
	RequestID  string        // provider request id header, when reported
}

// WithAttemptObserver registers fn to be called once per HTTP attempt on
// every entry point (chat, stream, embed, speak, transcribe, models),
// including retries and learn-once immediate retries. fn runs on the
// request goroutine: keep it fast and never call back into the SDK. fn must
// not panic, but a panic is recovered and discarded. A nil fn disables
// observation.
func WithAttemptObserver(fn func(AttemptEvent)) Option {
	return func(s *SDK) { s.attemptObs = fn }
}

type attemptKey struct{}

// attemptRec is the per-call scratch the HTTP helpers fill and the retry
// ladders drain into AttemptEvents. It is only touched on the request
// goroutine.
type attemptRec struct {
	op, model string
	n         int
	sent      bool // an HTTP attempt happened since the last emit
	status    int
	latency   time.Duration
	reqID     string
}

func attemptRecFrom(ctx context.Context) *attemptRec {
	r, _ := ctx.Value(attemptKey{}).(*attemptRec)
	return r
}

// withAttempts attaches an attempt record to ctx when an observer is set.
// An existing record is kept so a stream downgraded to buffered mid-call
// keeps one numbering.
func (pc *providerClient) withAttempts(ctx context.Context, op, model string) context.Context {
	if pc.opts.attempt == nil || attemptRecFrom(ctx) != nil {
		return ctx
	}
	return context.WithValue(ctx, attemptKey{}, &attemptRec{op: op, model: model})
}

// doObserved is client.Do that records status, latency and request id for
// the attempt observer.
func doObserved(ctx context.Context, c *http.Client, req *http.Request) (*http.Response, error) {
	r := attemptRecFrom(ctx)
	if r == nil {
		return c.Do(req)
	}
	t0 := time.Now()
	resp, err := c.Do(req)
	r.sent = true
	r.latency = max(time.Since(t0), 1)
	r.status, r.reqID = 0, ""
	if resp != nil {
		r.status = resp.StatusCode
		r.reqID = parseResponseMeta(resp.Header).RequestID
	}
	return resp, err
}

// emitAttempt delivers the event for the attempt just finished (if any)
// with its final error and the delay before the next attempt.
func (pc *providerClient) emitAttempt(ctx context.Context, err error, delay time.Duration) {
	r := attemptRecFrom(ctx)
	if r == nil || !r.sent {
		return
	}
	r.sent = false
	r.n++
	ev := AttemptEvent{
		Provider: pc.cfg.ID, Operation: r.op, Model: r.model, Attempt: r.n,
		Status: r.status, Err: err, Latency: r.latency, RetryDelay: delay, RequestID: r.reqID,
	}
	ev.Err = pc.redactErr(err)
	defer func() { _ = recover() }()
	pc.opts.attempt(ev)
}

// redactedError carries err's text with the API key masked; the original
// stays reachable through errors.Is/As.
type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string { return e.msg }
func (e *redactedError) Unwrap() error { return e.err }

// redactErr masks the provider API key if a transport error echoed it.
func (pc *providerClient) redactErr(err error) error {
	k := pc.cfg.APIKey
	if err == nil || k == "" {
		return err
	}
	if m := err.Error(); strings.Contains(m, k) {
		return &redactedError{msg: strings.ReplaceAll(m, k, "[redacted]"), err: err}
	}
	return err
}
