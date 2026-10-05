package llm

import (
	"sync"
	"sync/atomic"
)

// LearnKind identifies which learn-once fallback flag engaged.
type LearnKind string

const (
	// LearnBuffered: streaming was rejected by the provider (explicit
	// "streaming not supported" 400) or a 2xx response arrived without an
	// SSE content type. Every later generation on this provider is buffered.
	LearnBuffered LearnKind = "buffered"
	// LearnResponses: the provider requires /v1/responses for
	// reasoning_effort combined with function tools.
	LearnResponses LearnKind = "responses"
	// LearnNoneEffort: the provider rejected reasoning_effort with tools;
	// effort is pinned to "none" for later requests.
	LearnNoneEffort LearnKind = "none_effort"
	// LearnDropStreamOpts: the provider rejected stream_options; the field
	// is omitted from later requests.
	LearnDropStreamOpts LearnKind = "drop_stream_options"
)

// LearnEvent reports one learn-once constraint engaging. Status is the HTTP
// status that triggered the fallback (0 when none applied, e.g. a 2xx
// response that arrived without SSE framing); Message carries the provider
// error text when one exists.
type LearnEvent struct {
	Kind     LearnKind
	Provider string
	Status   int
	Message  string
}

var (
	learnObserverMu sync.RWMutex
	learnObserver   func(LearnEvent)
)

// SetLearnObserver registers a process-wide callback fired once per engaged
// learn flag, at the moment the fallback decision is made. Nil (the default)
// disables observation entirely. Delivery is at-most-once: a callback
// registered after a flag engaged (or swapped during the engaging call) does
// not retroactively receive that engagement. The callback runs on the request
// goroutine: keep it fast and non-blocking; it must never call back into the
// SDK and must not panic.
func SetLearnObserver(fn func(LearnEvent)) {
	learnObserverMu.Lock()
	learnObserver = fn
	learnObserverMu.Unlock()
}

func notifyLearn(ev LearnEvent) {
	learnObserverMu.RLock()
	fn := learnObserver
	learnObserverMu.RUnlock()
	if fn != nil {
		fn(ev)
	}
}

// engageLearn flips a learn-once flag exactly once (CAS) and fires the
// observer on the engaging call only. apiErr may be nil when no HTTP error
// triggered the fallback. Behavior is otherwise identical to Store(true):
// the caller's case guards remain the routing decision.
func (pc *providerClient) engageLearn(flag *atomic.Bool, kind LearnKind, apiErr *APIError) {
	if !flag.CompareAndSwap(false, true) {
		return
	}
	ev := LearnEvent{Kind: kind, Provider: pc.cfg.ID}
	if apiErr != nil {
		ev.Status = apiErr.Status
		ev.Message = apiErr.Message
	}
	notifyLearn(ev)
}
