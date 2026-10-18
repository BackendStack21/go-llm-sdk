package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

const obsKey = "sk-observer-secret-key"

type obsCollector struct {
	mu sync.Mutex
	ev []AttemptEvent
}

func (c *obsCollector) fn(e AttemptEvent) {
	c.mu.Lock()
	c.ev = append(c.ev, e)
	c.mu.Unlock()
}

func (c *obsCollector) all() []AttemptEvent {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]AttemptEvent(nil), c.ev...)
}

// obsStep is one scripted server response.
type obsStep struct {
	status int
	header map[string]string
	ctype  string
	body   string
}

func obsServer(t *testing.T, steps ...obsStep) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		i := int(n.Add(1)) - 1
		if i >= len(steps) {
			i = len(steps) - 1
		}
		s := steps[i]
		for k, v := range s.header {
			w.Header().Set(k, v)
		}
		if s.ctype != "" {
			w.Header().Set("Content-Type", s.ctype)
		}
		w.WriteHeader(s.status)
		_, _ = w.Write([]byte(s.body))
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

type obsEntry struct {
	name  string
	op    string
	model string
	ctype string
	ok    string
	call  func(s *SDK) error
}

func obsEntries() []obsEntry {
	ctx := context.Background()
	chatOK := `{"id":"c1","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`
	return []obsEntry{
		{"chat", "chat", "m", "application/json", chatOK, func(s *SDK) error {
			c, err := s.Chat("openai", "m")
			if err != nil {
				return err
			}
			_, err = c.Call(ctx, userMsg())
			return err
		}},
		{"stream", "stream", "m", "text/event-stream", metaSSE(`{"id":"c1","choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`, "[DONE]"), func(s *SDK) error {
			c, err := s.Chat("openai", "m")
			if err != nil {
				return err
			}
			_, err = c.CallStream(ctx, userMsg(), func(Delta) error { return nil })
			return err
		}},
		{"embed", "embed", "m", "application/json", `{"data":[{"index":0,"embedding":[0.5,1]}],"usage":{"prompt_tokens":2}}`, func(s *SDK) error {
			_, err := s.Embed(ctx, "openai", "m", EmbedRequest{Inputs: []string{"a"}})
			return err
		}},
		{"speak", "speak", "tts-1", "audio/mpeg", "\x01\x02", func(s *SDK) error {
			_, err := s.Speak(ctx, "openai", "tts-1", SpeakRequest{Text: "hi", Voice: "alloy"})
			return err
		}},
		{"transcribe", "transcribe", "whisper-1", "application/json", `{"text":"hi"}`, func(s *SDK) error {
			_, err := s.Transcribe(ctx, "openai", "whisper-1", TranscribeRequest{Audio: []byte{1, 2}})
			return err
		}},
		{"models", "models", "", "application/json", `{"data":[{"id":"m1"}]}`, func(s *SDK) error {
			p, err := s.Provider("openai")
			if err != nil {
				return err
			}
			_, err = p.ListModels(ctx, ForceRefresh())
			return err
		}},
	}
}

func obsSDK(t *testing.T, srv *httptest.Server, col *obsCollector, extra ...Option) *SDK {
	t.Helper()
	fastBackoff(t)
	opts := append([]Option{
		WithAttemptObserver(col.fn),
		WithProvider("openai", WithAPIKey(obsKey), WithBaseURL(srv.URL)),
	}, extra...)
	return New(opts...)
}

func obsNoKey(t *testing.T, ev AttemptEvent) {
	t.Helper()
	if ev.Err != nil && strings.Contains(ev.Err.Error(), obsKey) {
		t.Errorf("event Err leaks API key: %v", ev.Err)
	}
}

// A first-try success yields exactly one fully populated event.
func TestM6_ObserverSuccessFirstTry(t *testing.T) {
	for _, e := range obsEntries() {
		t.Run(e.name, func(t *testing.T) {
			srv, _ := obsServer(t, obsStep{200, map[string]string{"X-Request-Id": "req-1"}, e.ctype, e.ok})
			col := &obsCollector{}
			if err := e.call(obsSDK(t, srv, col)); err != nil {
				t.Fatal(err)
			}
			evs := col.all()
			if len(evs) != 1 {
				t.Fatalf("events = %d, want 1: %+v", len(evs), evs)
			}
			ev := evs[0]
			if ev.Provider != "openai" || ev.Operation != e.op || ev.Model != e.model {
				t.Errorf("identity = %q/%q/%q, want openai/%s/%q", ev.Provider, ev.Operation, ev.Model, e.op, e.model)
			}
			if ev.Attempt != 1 || ev.Status != 200 || ev.Err != nil {
				t.Errorf("attempt/status/err = %d/%d/%v", ev.Attempt, ev.Status, ev.Err)
			}
			if ev.Latency <= 0 {
				t.Errorf("Latency = %v, want > 0", ev.Latency)
			}
			if ev.RetryDelay != 0 {
				t.Errorf("RetryDelay = %v, want 0", ev.RetryDelay)
			}
			if ev.RequestID != "req-1" {
				t.Errorf("RequestID = %q, want req-1", ev.RequestID)
			}
		})
	}
}

// A retried call produces one event per HTTP attempt; only non-final
// retryable attempts carry a RetryDelay.
func TestM6_ObserverRetryProducesNEvents(t *testing.T) {
	for _, e := range obsEntries() {
		t.Run(e.name, func(t *testing.T) {
			srv, n := obsServer(t,
				obsStep{500, map[string]string{"X-Request-Id": "bad-1"}, "application/json", `{"error":{"message":"boom"}}`},
				obsStep{429, map[string]string{"X-Request-Id": "bad-2"}, "application/json", `{"error":{"message":"slow"}}`},
				obsStep{200, map[string]string{"X-Request-Id": "good"}, e.ctype, e.ok})
			col := &obsCollector{}
			if err := e.call(obsSDK(t, srv, col)); err != nil {
				t.Fatal(err)
			}
			evs := col.all()
			if int(n.Load()) != 3 || len(evs) != 3 {
				t.Fatalf("requests = %d, events = %d, want 3/3", n.Load(), len(evs))
			}
			wantStatus := []int{500, 429, 200}
			wantID := []string{"bad-1", "bad-2", "good"}
			for i, ev := range evs {
				if ev.Attempt != i+1 || ev.Status != wantStatus[i] || ev.RequestID != wantID[i] {
					t.Errorf("event %d = attempt %d status %d id %q", i, ev.Attempt, ev.Status, ev.RequestID)
				}
				if ev.Latency <= 0 {
					t.Errorf("event %d Latency = %v", i, ev.Latency)
				}
				if i < 2 {
					var ae *APIError
					if !errors.As(ev.Err, &ae) || ae.Status != wantStatus[i] {
						t.Errorf("event %d Err = %v, want *APIError %d", i, ev.Err, wantStatus[i])
					}
					if ev.RetryDelay <= 0 {
						t.Errorf("event %d RetryDelay = %v, want > 0", i, ev.RetryDelay)
					}
				} else if ev.Err != nil || ev.RetryDelay != 0 {
					t.Errorf("final event err=%v delay=%v, want nil/0", ev.Err, ev.RetryDelay)
				}
				obsNoKey(t, ev)
			}
		})
	}
}

// A terminal (non-retryable) failure fires one event with the error and no
// RetryDelay.
func TestM6_ObserverTerminalFailure(t *testing.T) {
	for _, e := range obsEntries() {
		t.Run(e.name, func(t *testing.T) {
			srv, n := obsServer(t, obsStep{401, nil, "application/json", `{"error":{"message":"nope"}}`})
			col := &obsCollector{}
			if err := e.call(obsSDK(t, srv, col)); err == nil {
				t.Fatal("want error")
			}
			evs := col.all()
			if len(evs) != int(n.Load()) || len(evs) != 1 {
				t.Fatalf("events = %d, requests = %d, want 1/1", len(evs), n.Load())
			}
			var ae *APIError
			if evs[0].Status != 401 || !errors.As(evs[0].Err, &ae) || evs[0].RetryDelay != 0 {
				t.Errorf("event = %+v", evs[0])
			}
			obsNoKey(t, evs[0])
		})
	}
}

// A Retry-After hint is reflected in RetryDelay.
func TestM6_ObserverRetryAfterHonoured(t *testing.T) {
	e := obsEntries()[0]
	srv, _ := obsServer(t,
		obsStep{429, map[string]string{"Retry-After": "1"}, "application/json", `{"error":{"message":"slow"}}`},
		obsStep{200, nil, e.ctype, e.ok})
	col := &obsCollector{}
	if err := e.call(obsSDK(t, srv, col)); err != nil {
		t.Fatal(err)
	}
	evs := col.all()
	if len(evs) != 2 || evs[0].RetryDelay < 900*1e6 {
		t.Fatalf("events = %+v, want 2 with first RetryDelay >= ~1s", evs)
	}
}

type obsFailRT struct{ n atomic.Int32 }

func (f *obsFailRT) RoundTrip(*http.Request) (*http.Response, error) {
	f.n.Add(1)
	return nil, errors.New("dial failed using " + obsKey)
}

// A transport error reports Status 0, retries, and never leaks the key.
func TestM6_ObserverTransportError(t *testing.T) {
	for _, e := range obsEntries() {
		t.Run(e.name, func(t *testing.T) {
			fastBackoff(t)
			rt := &obsFailRT{}
			col := &obsCollector{}
			s := New(WithAttemptObserver(col.fn), WithTransport(rt), WithRetryPolicy(RetryPolicy{MaxAttempts: 2}),
				WithProvider("openai", WithAPIKey(obsKey), WithBaseURL("http://127.0.0.1:1")))
			err := e.call(s)
			if err == nil {
				t.Fatal("want error")
			}
			evs := col.all()
			if len(evs) != 2 || int(rt.n.Load()) != 2 {
				t.Fatalf("events = %d, round trips = %d, want 2/2", len(evs), rt.n.Load())
			}
			for i, ev := range evs {
				if ev.Status != 0 || ev.Err == nil || ev.Attempt != i+1 {
					t.Errorf("event %d = %+v", i, ev)
				}
				obsNoKey(t, ev)
			}
			if evs[0].RetryDelay <= 0 || evs[1].RetryDelay != 0 {
				t.Errorf("RetryDelay = %v then %v, want >0 then 0", evs[0].RetryDelay, evs[1].RetryDelay)
			}
		})
	}
}

// The learn-once immediate retry counts as an attempt (RetryDelay 0 though
// not final).
func TestM6_ObserverLearnOnceRetryCounts(t *testing.T) {
	e := obsEntries()[1]
	srv, _ := obsServer(t,
		obsStep{400, nil, "application/json", `{"error":{"message":"Unknown parameter: stream_options"}}`},
		obsStep{200, nil, e.ctype, e.ok})
	col := &obsCollector{}
	if err := e.call(obsSDK(t, srv, col)); err != nil {
		t.Fatal(err)
	}
	evs := col.all()
	if len(evs) != 2 {
		t.Fatalf("events = %d, want 2: %+v", len(evs), evs)
	}
	if evs[0].Status != 400 || evs[0].Attempt != 1 || evs[0].Err == nil || evs[0].RetryDelay != 0 {
		t.Errorf("learn attempt = %+v", evs[0])
	}
	if evs[1].Status != 200 || evs[1].Attempt != 2 || evs[1].Err != nil {
		t.Errorf("retry attempt = %+v", evs[1])
	}
}

// A stream that fails after headers reports Status 200 with the error on
// that attempt's single event.
func TestM6_ObserverStreamMidFailure(t *testing.T) {
	srv, _ := obsServer(t, obsStep{200, nil, "text/event-stream",
		metaSSE(`{"id":"c1","choices":[{"delta":{"content":"par"}}]}`)})
	col := &obsCollector{}
	s := obsSDK(t, srv, col)
	c, _ := s.Chat("openai", "m")
	if _, err := c.CallStream(context.Background(), userMsg(), func(Delta) error { return nil }); err == nil {
		t.Fatal("want premature-close error")
	}
	evs := col.all()
	if len(evs) != 1 || evs[0].Status != 200 || evs[0].Err == nil || evs[0].Operation != "stream" {
		t.Fatalf("events = %+v", evs)
	}
}

// A panicking observer is recovered: the call still succeeds on every entry
// point, streams complete, and the observer is still invoked per attempt.
func TestM6_ObserverPanicRecovered(t *testing.T) {
	for _, e := range obsEntries() {
		t.Run(e.name, func(t *testing.T) {
			fastBackoff(t)
			srv, _ := obsServer(t,
				obsStep{500, nil, "application/json", `{"error":{"message":"boom"}}`},
				obsStep{200, nil, e.ctype, e.ok})
			var calls atomic.Int32
			s := New(WithAttemptObserver(func(AttemptEvent) { calls.Add(1); panic("observer boom") }),
				WithProvider("openai", WithAPIKey(obsKey), WithBaseURL(srv.URL)))
			if err := e.call(s); err != nil {
				t.Fatalf("call failed despite observer panic: %v", err)
			}
			if calls.Load() != 2 {
				t.Errorf("observer calls = %d, want 2", calls.Load())
			}
		})
	}
}

// Without an observer nothing changes (AC-5).
func TestM6_ObserverNilIsNoop(t *testing.T) {
	for _, e := range obsEntries() {
		t.Run(e.name, func(t *testing.T) {
			fastBackoff(t)
			srv, _ := obsServer(t, obsStep{200, nil, e.ctype, e.ok})
			s := New(WithAttemptObserver(nil), WithProvider("openai", WithAPIKey(obsKey), WithBaseURL(srv.URL)))
			if err := e.call(s); err != nil {
				t.Fatal(err)
			}
		})
	}
}
