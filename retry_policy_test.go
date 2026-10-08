package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// countingServer answers every request with status and counts hits.
func countingServer(t *testing.T, status int, header map[string]string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		for k, v := range header {
			w.Header().Set(k, v)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &n
}

func fastBackoff(t *testing.T) {
	t.Helper()
	old := backoffUnit
	backoffUnit = time.Millisecond
	t.Cleanup(func() { backoffUnit = old })
}

func userMsg() *ChatRequest {
	return &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}
}

// WithRetryPolicy bounds the attempt count on every retrying entry point.
func TestWithRetryPolicy_MaxAttempts(t *testing.T) {
	fastBackoff(t)
	srv, n := countingServer(t, http.StatusInternalServerError, nil)
	sdk := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 2}),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c, err := sdk.Chat("openai", "m")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	check := func(name string, want int32) {
		t.Helper()
		if got := n.Swap(0); got != want {
			t.Errorf("%s: attempts = %d, want %d", name, got, want)
		}
	}
	if _, err := c.Call(ctx, userMsg()); err == nil {
		t.Error("Call: want error")
	}
	check("Call", 2)
	if _, err := c.CallStream(ctx, userMsg(), func(Delta) error { return nil }); err == nil {
		t.Error("CallStream: want error")
	}
	check("CallStream", 2)
	if _, err := sdk.Speak(ctx, "openai", "tts", SpeakRequest{Text: "hi", Voice: "v"}); err == nil {
		t.Error("Speak: want error")
	}
	check("Speak", 2)
	if _, err := sdk.Transcribe(ctx, "openai", "stt", TranscribeRequest{Audio: []byte("a")}); err == nil {
		t.Error("Transcribe: want error")
	}
	check("Transcribe", 2)

	one := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 1}),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c1, _ := one.Chat("openai", "m")
	_, _ = c1.Call(ctx, userMsg())
	check("MaxAttempts=1", 1)
}

// The default ladder is unchanged: 8 attempts.
func TestRetryPolicy_Defaults(t *testing.T) {
	var p RetryPolicy
	if p.attempts() != maxRetries+1 || p.backoffCap() != maxRetryBackoff {
		t.Fatalf("defaults = %d / %v", p.attempts(), p.backoffCap())
	}
	p = RetryPolicy{MaxAttempts: -3, MaxBackoff: -time.Second}
	if p.attempts() != maxRetries+1 || p.backoffCap() != maxRetryBackoff {
		t.Fatalf("negative values must fall back to defaults: %d / %v", p.attempts(), p.backoffCap())
	}
}

func TestRetryPolicy_MaxBackoffCapsDelay(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI}, nil, nil)
	pc.opts.retry = RetryPolicy{MaxBackoff: 5 * time.Millisecond}
	for attempt := 0; attempt < 10; attempt++ {
		if d := pc.retryDelay(0, attempt); d > 5*time.Millisecond {
			t.Fatalf("attempt %d delay %v exceeds cap", attempt, d)
		}
	}
	// Retry-After is a server instruction and is honored over the cap.
	if d := pc.retryDelay(time.Second, 0); d != time.Second {
		t.Errorf("Retry-After delay = %v", d)
	}
}

// A buffered call's retries share one wall-clock budget (the request
// timeout), exactly like streaming — never timeout × attempts.
func TestCall_WholeCallBudget(t *testing.T) {
	srv, _ := countingServer(t, http.StatusServiceUnavailable, map[string]string{"Retry-After": "1"})
	sdk := New(WithRequestTimeout(300*time.Millisecond),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c, _ := sdk.Chat("openai", "m")
	start := time.Now()
	_, err := c.Call(context.Background(), userMsg())
	if el := time.Since(start); el > 900*time.Millisecond {
		t.Fatalf("buffered call ran %v, want ≤ the 300ms budget (+slack)", el)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want deadline exceeded", err)
	}
}

// WithStreamIdleTimeout is per SDK and wins over the process default.
func TestWithStreamIdleTimeout(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		<-release
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	sdk := New(WithStreamIdleTimeout(50*time.Millisecond), WithRetryPolicy(RetryPolicy{MaxAttempts: 1}),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c, _ := sdk.Chat("openai", "m")
	start := time.Now()
	_, err := c.CallStream(context.Background(), userMsg(), func(Delta) error { return nil })
	if !errors.Is(err, ErrIdleTimeout) {
		t.Fatalf("err = %v, want ErrIdleTimeout", err)
	}
	if el := time.Since(start); el > 2*time.Second {
		t.Errorf("idle watchdog fired after %v, want ~50ms", el)
	}
	if New(WithStreamIdleTimeout(-1)).idle != 0 {
		t.Error("non-positive idle timeout must be ignored")
	}
}

// WithLearnObserver is per SDK and supersedes the process-wide observer.
func TestWithLearnObserver(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)
	var global atomic.Int32
	SetLearnObserver(func(LearnEvent) { global.Add(1) })
	t.Cleanup(func() { SetLearnObserver(nil) })

	var mu sync.Mutex
	var got []LearnEvent
	sdk := New(WithLearnObserver(func(ev LearnEvent) { mu.Lock(); got = append(got, ev); mu.Unlock() }),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	c, _ := sdk.Chat("openai", "m")
	if _, err := c.CallStream(context.Background(), userMsg(), func(Delta) error { return nil }); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || got[0].Kind != LearnBuffered || got[0].Provider != "openai" {
		t.Errorf("SDK observer events = %+v", got)
	}
	if global.Load() != 0 {
		t.Error("process-wide observer must not fire when the SDK has its own")
	}
}

// SetStreamIdleTimeout is safe to call while streams read the default.
func TestSetStreamIdleTimeout_RaceFree(t *testing.T) {
	orig := StreamIdleTimeout()
	t.Cleanup(func() { SetStreamIdleTimeout(orig) })
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(2)
		go func() { defer wg.Done(); SetStreamIdleTimeout(time.Second) }()
		go func() { defer wg.Done(); _ = StreamIdleTimeout() }()
	}
	wg.Wait()
}

// setIdleForTest pins the process-wide idle watchdog for one test.
func setIdleForTest(t *testing.T, d time.Duration) {
	t.Helper()
	old := StreamIdleTimeout()
	SetStreamIdleTimeout(d)
	t.Cleanup(func() { SetStreamIdleTimeout(old) })
}
