package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// A malformed models body is a terminal protocol failure: retrying the
// same endpoint cannot fix it.
func TestListModels_ParseErrorNotRetried(t *testing.T) {
	for _, f := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
		var n atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			n.Add(1)
			_, _ = fmt.Fprint(w, `{not json`)
		}))
		_, err := newListModels(srv.URL, f)
		srv.Close()
		if err == nil {
			t.Errorf("%s: want parse error", f)
		}
		if got := n.Load(); got != 1 {
			t.Errorf("%s: requests = %d, want 1 (no retry)", f, got)
		}
	}
}

// A listing longer than the page cap is an error, never a silent
// truncation.
func TestListModels_PageCapIsAnError(t *testing.T) {
	cases := map[Format]string{
		FormatAnthropic: `{"data":[{"id":"x"}],"has_more":true,"last_id":"x"}`,
		FormatGemini:    `{"models":[{"name":"models/x"}],"nextPageToken":"P"}`,
	}
	for f, body := range cases {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = fmt.Fprint(w, body)
		}))
		models, err := newListModels(srv.URL, f)
		srv.Close()
		if err == nil || !errors.Is(err, ErrModelListTruncated) {
			t.Errorf("%s: err = %v (models %d), want ErrModelListTruncated", f, err, len(models))
		}
	}
}

// Concurrent cache misses share one upstream fetch.
func TestListModels_SingleFlight(t *testing.T) {
	var n atomic.Int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		<-release
		_, _ = fmt.Fprint(w, `{"data":[{"id":"m"}]}`)
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	p, _ := sdk.Provider("openai")
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ms, err := p.ListModels(context.Background())
			if err == nil && (len(ms) != 1 || ms[0].ID != "m") {
				err = fmt.Errorf("models = %+v", ms)
			}
			errs <- err
		}()
	}
	deadline := time.Now().Add(2 * time.Second)
	for n.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	time.Sleep(50 * time.Millisecond) // let the other callers queue up
	close(release)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Error(err)
		}
	}
	if got := n.Load(); got != 1 {
		t.Errorf("upstream fetches = %d, want 1", got)
	}
}

// A follower whose leader's context died fetches for itself instead of
// inheriting the leader's cancellation.
func TestListModels_SingleFlightLeaderCancelled(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n.Add(1) == 1 {
			<-r.Context().Done() // leader hangs until cancelled
			return
		}
		_, _ = fmt.Fprint(w, `{"data":[{"id":"m"}]}`)
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	p, _ := sdk.Provider("openai")
	leaderCtx, cancel := context.WithCancel(context.Background())
	leaderErr := make(chan error, 1)
	go func() { _, err := p.ListModels(leaderCtx); leaderErr <- err }()
	for n.Load() == 0 {
		time.Sleep(2 * time.Millisecond)
	}
	followerRes := make(chan []Model, 1)
	followerErr := make(chan error, 1)
	go func() {
		ms, err := p.ListModels(context.Background())
		followerRes <- ms
		followerErr <- err
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()
	if err := <-leaderErr; !errors.Is(err, context.Canceled) {
		t.Errorf("leader err = %v", err)
	}
	if err := <-followerErr; err != nil {
		t.Fatalf("follower err = %v, want its own successful fetch", err)
	}
	if ms := <-followerRes; len(ms) != 1 {
		t.Errorf("follower models = %+v", ms)
	}
}

type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// A panic inside the leader's fetch (e.g. a user RoundTripper) must never
// leave the flight in place: later calls fetch again.
func TestListModels_LeaderPanicDoesNotWedge(t *testing.T) {
	var calls atomic.Int32
	rt := rtFunc(func(r *http.Request) (*http.Response, error) {
		if calls.Add(1) == 1 {
			panic("transport exploded")
		}
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"data":[{"id":"m"}]}`))}, nil
	})
	sdk := New(WithTransport(rt), WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://x")))
	p, _ := sdk.Provider("openai")
	func() {
		defer func() { _ = recover() }()
		_, _ = p.ListModels(context.Background())
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	ms, err := p.ListModels(ctx)
	if err != nil || len(ms) != 1 {
		t.Fatalf("after leader panic: models=%v err=%v", ms, err)
	}
}

// An upstream timeout inside the leader's fetch is the shared result, not a
// cancellation: followers must not re-lead one after another.
func TestListModels_UpstreamTimeoutShared(t *testing.T) {
	fastBackoff(t)
	var calls atomic.Int32
	gate := make(chan struct{})
	rt := rtFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		<-gate
		return nil, fmt.Errorf("dial: %w", context.DeadlineExceeded)
	})
	sdk := New(WithTransport(rt), WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://x")))
	p, _ := sdk.Provider("openai")
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = p.ListModels(context.Background()) }()
	}
	time.Sleep(50 * time.Millisecond)
	close(gate)
	wg.Wait()
	if got := calls.Load(); got != 3 { // one flight = 3 attempts
		t.Errorf("upstream attempts = %d, want 3 (one shared flight)", got)
	}
}
