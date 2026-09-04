package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// providerClient executes requests against one configured endpoint. The
// three wire formats share this single retry/stream orchestration; format
// differences live in the build/parse/map functions (openai.go,
// anthropic.go, gemini.go).
//
// Streaming contract (ported from odek's battle-tested client):
//   - Hard wall-clock deadline over the whole stream (context) plus an idle
//     watchdog between SSE events; keepalives reset the watchdog.
//   - Retries happen only before the first delta is emitted, so partial
//     output is never duplicated by a silent full retry.
//   - Consumer abort (delta handler error) returns the partial result
//     alongside *StreamAbortedError.
//   - Learn-once fallbacks: drop stream_options (field level), fall back to
//     the buffered path (provider rejects streaming), pin reasoning_effort
//     "none" (provider rejects effort combined with tools).
//
// learnOnce holds the learn-once fallback flags. They live on the Provider
// (shared across every ChatClient minted from it) so a constraint the
// provider teaches one client is honored by all of them.
type learnOnce struct {
	dropStreamOptions atomic.Bool
	forceBuffered     atomic.Bool
	forceNoneEffort   atomic.Bool
}

type providerClient struct {
	cfg        ProviderConfig
	bufPtr     atomic.Pointer[http.Client] // buffered client; SetRequestTimeout swaps it atomically
	streamHTTP *http.Client                // no deadline; SSE body reads
	base       string                      // trimmed base URL
	learn      *learnOnce                  // shared learn-once fallback state
}

// buffered returns the current buffered-path HTTP client.
func (pc *providerClient) buffered() *http.Client { return pc.bufPtr.Load() }

func newProviderClient(cfg ProviderConfig, buffered, stream *http.Client) *providerClient {
	return newProviderClientWithLearn(cfg, buffered, stream, &learnOnce{})
}

// newProviderClientWithLearn builds a client that shares learn-once state.
func newProviderClientWithLearn(cfg ProviderConfig, buffered, stream *http.Client, learn *learnOnce) *providerClient {
	if learn == nil {
		learn = &learnOnce{}
	}
	if buffered == nil {
		buffered = newBufferedHTTP(nil, 0)
	}
	pc := &providerClient{
		cfg:        cfg,
		streamHTTP: stream,
		base:       strings.TrimRight(cfg.BaseURL, "/"),
		learn:      learn,
	}
	pc.bufPtr.Store(buffered)
	return pc
}

// Response body read caps (DoS/OOM bound).
const (
	maxResponseSize       = 50 << 20 // 50 MB chat responses
	maxModelsResponseSize = 8 << 20  // 8 MB model listings
	maxErrorBodyPreview   = 512      // bytes of error body kept in APIError
)

// streamIdleTimeout bounds the silence between SSE events. Thinking models
// can legitimately spend minutes before their first event, so the default
// is generous. Package var so tests can shorten it; operators override
// via SetStreamIdleTimeout.
var streamIdleTimeout = 120 * time.Second

// SetStreamIdleTimeout overrides the SSE idle watchdog. Call at startup,
// before the first request; non-positive values are ignored.
func SetStreamIdleTimeout(d time.Duration) {
	if d > 0 {
		streamIdleTimeout = d
	}
}

// StreamIdleTimeout reports the active idle watchdog (introspection/tests).
func StreamIdleTimeout() time.Duration {
	return streamIdleTimeout
}

// errStreamStop is the internal sentinel for a clean stream end.
var errStreamStop = errors.New("llm: stream complete")

// errPrematureClose marks a 200+SSE stream the provider closed before its
// completion signal; retryable only before the first delta.
var errPrematureClose = errors.New("llm: provider closed the stream before completion")

// errNonSSE marks a streamed request answered with a regular body.
var errNonSSE = errors.New("llm: provider answered a streamed request with a non-event-stream body")

// consumerAbort wraps a delta-handler error through pumpSSE.
type consumerAbort struct{ err error }

func (c *consumerAbort) Error() string { return c.err.Error() }
func (c *consumerAbort) Unwrap() error { return c.err }

// requestTimeout is the per-request wall-clock budget; streaming uses it
// as the hard overall deadline.
func (pc *providerClient) requestTimeout() time.Duration {
	if t := pc.buffered().Timeout; t > 0 {
		return t
	}
	return DefaultTimeout
}

// ── request assembly ─────────────────────────────────────────────────────

// buildChatRequest dispatches format-specific serialization. stream=false
// yields the buffered request; stream=true the SSE request.
func (pc *providerClient) buildChatRequest(req *ChatRequest, model string, stream bool) ([]byte, string, error) {
	// Reject unknown roles loudly: OpenAI would silently send them as user
	// messages and Anthropic/Gemini would silently drop them.
	for i, m := range req.Messages {
		switch m.Role {
		case RoleUser, RoleAssistant, RoleSystem, RoleTool:
		default:
			return nil, "", &ConfigError{Msg: fmt.Sprintf("message %d: unknown role %q", i, string(m.Role))}
		}
	}
	if model == "" {
		model = req.Model
	}
	if model == "" {
		return nil, "", &ConfigError{Msg: "no model set (ChatRequest.Model empty and no ChatClient model)"}
	}
	switch pc.cfg.Format {
	case FormatAnthropic:
		body, err := buildAnthropicRequest(req, model, stream)
		return body, pc.base + "/v1/messages", err
	case FormatGemini:
		body, err := buildGeminiRequest(req, model, stream)
		if stream {
			return body, fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse", pc.base, model), err
		}
		return body, fmt.Sprintf("%s/v1beta/models/%s:generateContent", pc.base, model), err
	default: // FormatOpenAI
		oa := buildOpenAIRequest(pc.cfg, req, model, stream, !pc.learn.dropStreamOptions.Load())
		if pc.learn.forceNoneEffort.Load() && len(req.Tools) > 0 {
			oa = reasoningEffortNonePatched(oa)
		}
		body, err := json.Marshal(oa)
		return body, pc.base + "/chat/completions", err
	}
}

// setAuthHeaders applies format-specific authentication headers. Keys are
// never logged.
func (pc *providerClient) setAuthHeaders(h http.Header) {
	switch pc.cfg.Format {
	case FormatAnthropic:
		h.Set("x-api-key", pc.cfg.APIKey)
		v := pc.cfg.Quirks.AnthropicVersion
		if v == "" {
			v = "2023-06-01"
		}
		h.Set("anthropic-version", v)
	case FormatGemini:
		h.Set("x-goog-api-key", pc.cfg.APIKey)
	default:
		h.Set("Authorization", "Bearer "+pc.cfg.APIKey)
	}
}

// httpError renders a non-2xx response as *APIError with the provider's
// own error text when parseable.
func (pc *providerClient) httpError(status int, body []byte) *APIError {
	e := &APIError{
		Provider:  pc.cfg.ID,
		Status:    status,
		Retryable: retryableStatus(status),
	}
	var msg, code string
	switch pc.cfg.Format {
	case FormatAnthropic:
		var eb struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &eb) == nil && eb.Error.Message != "" {
			msg, code = eb.Error.Message, eb.Error.Type
		}
	case FormatGemini:
		var eb struct {
			Error struct {
				Message string `json:"message"`
				Status  string `json:"status"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &eb) == nil && eb.Error.Message != "" {
			msg, code = eb.Error.Message, eb.Error.Status
		}
	default:
		// OpenAI's canonical envelope nests the error object; some
		// gateways send the fields top-level. Try nested first.
		var nested struct {
			Error struct {
				Message string `json:"message"`
				Code    any    `json:"code"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &nested) == nil && nested.Error.Message != "" {
			msg = nested.Error.Message
			if s, ok := nested.Error.Code.(string); ok {
				code = s
			}
		} else {
			var eb oaErrorBody
			if json.Unmarshal(body, &eb) == nil && eb.Message != "" {
				msg = eb.Message
				if s, ok := eb.Code.(string); ok {
					code = s
				}
			}
		}
	}
	if msg == "" && len(body) > 0 {
		b := body
		if len(b) > maxErrorBodyPreview {
			b = b[:maxErrorBodyPreview]
		}
		msg = string(b)
	}
	e.Message, e.Code = msg, code
	if e.Status == http.StatusTooManyRequests && billingExhausted(e) {
		// Permanent billing/resource exhaustion: never retryable — fail
		// fast instead of burning the backoff ladder.
		e.Retryable = false
	}
	return e
}

// post sends one HTTP request and reads the full body (capped). Non-2xx
// responses return an *APIError with the Retry-After hint alongside.
func (pc *providerClient) post(ctx context.Context, client *http.Client, url string, body []byte) (data []byte, ra time.Duration, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, 0, &ConfigError{Msg: "build request: " + err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	pc.setAuthHeaders(req.Header)

	resp, err := client.Do(req)
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
		return data, ra, pc.httpError(resp.StatusCode, data)
	}
	return data, ra, nil
}

// ── error classification helpers ─────────────────────────────────────────

// reasoningEffortRejected reports whether err is a 400 whose provider
// response names reasoning_effort as the offending parameter.
func reasoningEffortRejected(err error) bool {
	var e *APIError
	if !errors.As(err, &e) || e.Status != http.StatusBadRequest {
		return false
	}
	return strings.Contains(e.Message, "reasoning_effort")
}

// streamOptionsRejected classifies a 400 naming stream_options.
func streamOptionsRejected(e *APIError) bool {
	return e != nil && e.Status == http.StatusBadRequest &&
		strings.Contains(e.Message, "stream_options")
}

// streamRejected classifies a 400 rejecting streaming itself. Deliberately
// narrow: the message must pair "stream" with an explicit rejection
// phrase, so unrelated 400s that merely mention streaming (context-length
// errors, parameter limits) never trigger the permanent buffered downgrade.
func streamRejected(e *APIError) bool {
	if e == nil || e.Status != http.StatusBadRequest {
		return false
	}
	m := strings.ToLower(e.Message)
	if !strings.Contains(m, "stream") || strings.Contains(m, "stream_options") {
		return false
	}
	for _, phrase := range []string{"not support", "unsupported", "does not support", "not allowed", "disabled", "reject"} {
		if strings.Contains(m, phrase) {
			return true
		}
	}
	return false
}

// billingExhausted reports whether a 429 is really a permanent
// billing/resource failure (e.g. Z.ai "Insufficient balance or no resource
// package", OpenAI "insufficient_quota"). Only a recharge fixes it, so
// running the full backoff ladder is wasted time.
func billingExhausted(e *APIError) bool {
	if e == nil || e.Status != http.StatusTooManyRequests {
		return false
	}
	m := strings.ToLower(e.Message)
	return strings.Contains(m, "insufficient balance") ||
		strings.Contains(m, "insufficient_quota") ||
		strings.Contains(m, "no resource package") ||
		strings.Contains(m, "exceeded your current quota")
}

// retryDelay picks Retry-After when present, else exponential backoff.
func retryDelay(ra time.Duration, attempt int) time.Duration {
	if ra > 0 {
		return ra
	}
	return backoffDelay(attempt + 1)
}

// ── buffered chat ────────────────────────────────────────────────────────

// call runs a buffered chat completion with retries.
func (pc *providerClient) call(ctx context.Context, req *ChatRequest, model string) (*ChatResult, error) {
	var (
		lastErr error
		rateErr *APIError // last 429
		rateRA  time.Duration
	)
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		body, url, err := pc.buildChatRequest(req, model, false)
		if err != nil {
			return nil, err
		}
		data, ra, err := pc.post(ctx, pc.buffered(), url, body)
		if err != nil {
			var apiErr *APIError
			if errors.As(err, &apiErr) {
				switch {
				case apiErr.Status == http.StatusTooManyRequests:
					if billingExhausted(apiErr) {
						// Permanent: only a recharge fixes this.
						return nil, apiErr
					}
					rateErr, rateRA, lastErr = apiErr, ra, apiErr
					if attempt < maxRetries {
						if !retrySleep(ctx, retryDelay(ra, attempt)) {
							// The sleep died with the context (deadline or
							// cancel); still surface the 429 — the caller
							// needs Status/RetryAfter to plan the retry.
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
				case apiErr.Status == http.StatusBadRequest && len(req.Tools) > 0 &&
					!pc.learn.forceNoneEffort.Load() && reasoningEffortRejected(apiErr):
					// Learn the constraint once; retry immediately with
					// effort pinned to "none".
					pc.learn.forceNoneEffort.Store(true)
					lastErr = apiErr
					if attempt < maxRetries {
						continue
					}
					return nil, apiErr
				}
				if rateErr != nil && !apiErr.Retryable && apiErr.Status != http.StatusTooManyRequests {
					// A definitive failure after earlier 429s.
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
		return pc.parseResponse(data)
	}
	if rateErr != nil {
		return nil, &RateLimitError{APIError: *rateErr, Attempts: maxRetries + 1, RetryAfter: rateRA}
	}
	return nil, fmt.Errorf("llm: retry exhausted (%d attempts): %w", maxRetries+1, lastErr)
}

// parseResponse dispatches format-specific buffered parsing.
func (pc *providerClient) parseResponse(data []byte) (*ChatResult, error) {
	switch pc.cfg.Format {
	case FormatAnthropic:
		return parseAnthropicResponse(data)
	case FormatGemini:
		return parseGeminiResponse(data)
	default:
		return parseOpenAIResponse(data)
	}
}

// ── streaming chat ───────────────────────────────────────────────────────

// streamEventMapper folds one SSE payload into acc and reports completion.
type streamEventMapper func(data []byte, acc *streamAccum) ([]Delta, bool, error)

// mapper dispatches the format's SSE mapping.
func (pc *providerClient) mapper() streamEventMapper {
	switch pc.cfg.Format {
	case FormatAnthropic:
		return mapAnthropicStreamEvent
	case FormatGemini:
		return mapGeminiStreamEvent
	default:
		return mapOpenAIStreamEvent
	}
}

// callStream runs a streaming chat completion. onDelta receives canonical
// fragments; returning an error from it aborts with the partial result and
// *StreamAbortedError.
func (pc *providerClient) callStream(ctx context.Context, req *ChatRequest, model string, onDelta func(Delta) error) (*ChatResult, error) {
	if pc.learn.forceBuffered.Load() {
		return pc.call(ctx, req, model)
	}
	deadlineCtx, cancel := context.WithTimeout(ctx, pc.requestTimeout())
	defer cancel()

	mapper := pc.mapper()
	var (
		lastErr error
		rateErr *APIError
		rateRA  time.Duration
	)
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := deadlineCtx.Err(); err != nil {
			break
		}
		if pc.learn.forceBuffered.Load() {
			cancel()
			return pc.call(ctx, req, model)
		}
		body, url, err := pc.buildChatRequest(req, model, true)
		if err != nil {
			return nil, err
		}

		out := pc.attemptStream(deadlineCtx, url, body, mapper, onDelta, len(req.Tools) > 0)
		switch {
		case out.success():
			return out.result, nil
		case out.abort != nil:
			return out.result, out.abort
		case out.err != nil && out.result != nil:
			// Partial output was delivered: never retry, surface with the
			// partial result (a retry would duplicate user-visible output).
			return out.result, out.err
		case out.learnRetry: // learn-once applied; retry without backoff
			if out.apiErr != nil {
				lastErr = out.apiErr
			} else if out.err != nil {
				lastErr = out.err
			}
			if attempt < maxRetries {
				continue
			}
			// Learn trigger fired on the final attempt: surface the cause
			// instead of falling off the loop with a nil error.
			return nil, lastErr
		case out.apiErr != nil && out.apiErr.Retryable:
			if out.apiErr.Status == http.StatusTooManyRequests {
				rateErr, rateRA = out.apiErr, out.retryAfter
			}
			lastErr = out.apiErr
			if attempt < maxRetries && retrySleep(deadlineCtx, retryDelay(out.retryAfter, attempt)) {
				continue
			}
			if rateErr != nil {
				return nil, &RateLimitError{APIError: *rateErr, Attempts: attempt + 1, RetryAfter: rateRA}
			}
			return nil, out.apiErr
		case out.apiErr != nil:
			return nil, out.apiErr
		case out.err != nil && attempt < maxRetries:
			lastErr = out.err
			if retrySleep(deadlineCtx, retryDelay(0, attempt)) {
				continue
			}
			if err := deadlineCtx.Err(); err != nil {
				return nil, err // interrupted by deadline/cancel, not exhaustion
			}
			return nil, fmt.Errorf("llm: retry exhausted (%d attempts): %w", attempt+1, lastErr)
		default:
			return nil, out.err
		}
	}
	if rateErr != nil {
		return nil, &RateLimitError{APIError: *rateErr, Attempts: maxRetries + 1, RetryAfter: rateRA}
	}
	if lastErr != nil {
		return nil, fmt.Errorf("llm: retry exhausted (%d attempts): %w", maxRetries+1, lastErr)
	}
	return nil, deadlineCtx.Err()
}

// streamOutcome is one streaming attempt's result. Exactly one of the
// terminal fields is meaningful; see the cases in callStream.
type streamOutcome struct {
	result     *ChatResult         // final (success) or partial (abort/fail)
	abort      *StreamAbortedError // consumer aborted; result is partial
	apiErr     *APIError           // provider 4xx/5xx (retryable flag inside)
	err        error               // transport / watchdog / parse error
	retryAfter time.Duration       // Retry-After hint when apiErr is a 429
	learnRetry bool                // learn-once flag set; retry immediately
}

func (o *streamOutcome) success() bool {
	return o.err == nil && o.abort == nil && o.apiErr == nil && !o.learnRetry
}

// attemptStream performs one streaming attempt. learnEffort enables the
// reasoning_effort learn-once trigger (set when the request carries tools).
func (pc *providerClient) attemptStream(ctx context.Context, url string, body []byte, mapper streamEventMapper, onDelta func(Delta) error, learnEffort bool) streamOutcome {
	req, rerr := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if rerr != nil {
		return streamOutcome{err: &ConfigError{Msg: "build request: " + rerr.Error()}}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	pc.setAuthHeaders(req.Header)

	resp, derr := pc.streamHTTP.Do(req)
	if derr != nil {
		return streamOutcome{err: derr} // transport error: retryable
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		ra := parseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
		e := pc.httpError(resp.StatusCode, data)
		switch {
		case streamOptionsRejected(e):
			pc.learn.dropStreamOptions.Store(true)
			return streamOutcome{learnRetry: true, apiErr: e}
		case learnEffort && !pc.learn.forceNoneEffort.Load() && reasoningEffortRejected(e):
			pc.learn.forceNoneEffort.Store(true)
			return streamOutcome{learnRetry: true, apiErr: e}
		case streamRejected(e):
			pc.learn.forceBuffered.Store(true)
			return streamOutcome{learnRetry: true, apiErr: e}
		default:
			return streamOutcome{apiErr: e, retryAfter: ra}
		}
	}

	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		// Provider answered a streamed request with a regular body.
		pc.learn.forceBuffered.Store(true)
		return streamOutcome{learnRetry: true, err: errNonSSE}
	}

	acc := newStreamAccum()
	perr := pumpSSE(ctx, resp.Body, streamIdleTimeout, func(data []byte) error {
		if string(bytes.TrimSpace(data)) == "[DONE]" {
			return errStreamStop
		}
		deltas, done, merr := mapper(data, acc)
		if merr != nil {
			return merr
		}
		for _, d := range deltas {
			acc.emitted = true
			if herr := onDelta(d); herr != nil {
				return &consumerAbort{err: herr}
			}
		}
		if done {
			return errStreamStop
		}
		return nil
	})

	switch {
	case perr == nil, errors.Is(perr, errStreamStop):
		if perr == nil && pc.cfg.Format != FormatGemini && acc.finishReason == "" {
			// Gemini completes at EOF; every other format has an explicit
			// completion signal that never arrived — the provider dropped
			// the stream. Never surface this as an empty success.
			if acc.emitted {
				return streamOutcome{
					result: acc.result(),
					err:    fmt.Errorf("llm: stream closed before completion: %w", errPrematureClose),
				}
			}
			return streamOutcome{err: errPrematureClose}
		}
		return streamOutcome{result: acc.result()}
	default:
		var ca *consumerAbort
		if errors.As(perr, &ca) {
			return streamOutcome{result: acc.result(), abort: &StreamAbortedError{Reason: ca.err}}
		}
		if !acc.emitted {
			// No output delivered yet: a full retry duplicates nothing.
			// Idle timeout, deadline, transport and parse errors are all
			// retryable at this stage.
			return streamOutcome{err: perr}
		}
		// Partial output was delivered: never retry, surface with partial.
		return streamOutcome{
			result: acc.result(),
			err:    fmt.Errorf("llm: stream failed after partial output: %w", perr),
		}
	}
}

// ── stream accumulator ───────────────────────────────────────────────────

// toolCallAccum assembles one tool call from streaming fragments: the first
// fragment for an index carries id/name; argument fragments concatenate.
type toolCallAccum struct {
	id   string
	name string
	args strings.Builder
}

// streamAccum assembles a ChatResult from SSE chunks across formats.
type streamAccum struct {
	content           strings.Builder
	reasoning         strings.Builder
	calls             []*toolCallAccum
	callIndex         map[int]*toolCallAccum
	finishReason      string
	usage             Usage
	emitted           bool   // any delta delivered to the consumer
	thinkingSignature string // anthropic signature_delta capture
}

func newStreamAccum() *streamAccum {
	return &streamAccum{callIndex: make(map[int]*toolCallAccum)}
}

// call returns the accumulator for tool-call index, creating it in order.
func (a *streamAccum) call(idx int) *toolCallAccum {
	if c, ok := a.callIndex[idx]; ok {
		return c
	}
	c := &toolCallAccum{}
	a.callIndex[idx] = c
	a.calls = append(a.calls, c)
	return c
}

func (a *streamAccum) result() *ChatResult {
	res := &ChatResult{
		Content:           a.content.String(),
		ReasoningContent:  a.reasoning.String(),
		ThinkingSignature: a.thinkingSignature,
		FinishReason:      a.finishReason,
		Usage:             a.usage,
	}
	for _, c := range a.calls {
		res.ToolCalls = append(res.ToolCalls, ToolCall{ID: c.id, Name: c.name, Arguments: c.args.String()})
	}
	return res
}
