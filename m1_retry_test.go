package llm

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"
)

// Bug 4: a huge Retry-After must clamp, never overflow negative.
func TestM1_ParseRetryAfterOverflowClamps(t *testing.T) {
	for _, v := range []string{"9223372037", "99999999999", "9223372036854775807"} {
		got := parseRetryAfter(v, time.Now())
		if got != maxRetryAfter {
			t.Errorf("parseRetryAfter(%q) = %v, want %v", v, got, maxRetryAfter)
		}
	}
}

// Bug 5: oversized responses are terminal; they must not be retried (each
// retry is a new billed generation).
func TestM1_OversizedResponseNotRetried(t *testing.T) {
	cases := []struct {
		name string
		size int
		call func(s *SDK) error
	}{
		{"chat", maxResponseSize + 1, func(s *SDK) error {
			cc, err := s.Chat("openai", "m")
			if err != nil {
				return err
			}
			_, err = cc.Call(context.Background(), m1Req())
			return err
		}},
		{"speak", maxResponseSize + 1, func(s *SDK) error {
			_, err := s.Speak(context.Background(), "openai", "tts-1", SpeakRequest{Text: "hi", Voice: "alloy"})
			return err
		}},
		{"transcribe", maxResponseSize + 1, func(s *SDK) error {
			_, err := s.Transcribe(context.Background(), "openai", "whisper-1", TranscribeRequest{Audio: []byte("abc")})
			return err
		}},
		{"models", maxModelsResponseSize + 1, func(s *SDK) error {
			p, err := s.Provider("openai")
			if err != nil {
				return err
			}
			_, err = p.ListModels(context.Background())
			return err
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fastBackoff(t)
			var requests atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				buf := make([]byte, 1<<20)
				for n := 0; n < tc.size; n += len(buf) {
					if _, err := w.Write(buf); err != nil {
						return
					}
				}
			}))
			defer srv.Close()
			s := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 3}))
			s.put(ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"})
			if err := tc.call(s); err == nil {
				t.Fatal("expected an oversize error")
			}
			if n := requests.Load(); n != 1 {
				t.Fatalf("oversized response retried: %d requests, want 1", n)
			}
		})
	}
}

// Bug 6: when the wait would outlive the context deadline, return the last
// provider/transport error promptly instead of sleeping to the deadline and
// returning a bare context error.
func TestM1_BackoffBeyondDeadlineKeepsLastError(t *testing.T) {
	cases := []struct {
		name    string
		handler http.HandlerFunc
		check   func(t *testing.T, err error)
	}{
		{"503 with Retry-After", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Retry-After", "60")
			w.WriteHeader(http.StatusServiceUnavailable)
		}, func(t *testing.T, err error) {
			var ae *APIError
			if !errors.As(err, &ae) || ae.Status != http.StatusServiceUnavailable {
				t.Fatalf("last APIError lost: %v", err)
			}
		}},
		{"500 with long backoff", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}, func(t *testing.T, err error) {
			var ae *APIError
			if !errors.As(err, &ae) || ae.Status != http.StatusInternalServerError {
				t.Fatalf("last APIError lost: %v", err)
			}
		}},
		{"transport error", func(w http.ResponseWriter, r *http.Request) {
			c, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				_ = c.Close()
			}
		}, func(t *testing.T, err error) {
			var ue *url.Error
			var ne net.Error
			if !errors.As(err, &ue) && !errors.As(err, &ne) {
				t.Fatalf("last transport error lost: %v", err)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			old := backoffUnit
			backoffUnit = time.Minute
			t.Cleanup(func() { backoffUnit = old })
			srv := httptest.NewServer(tc.handler)
			defer srv.Close()
			cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
			backoffUnit = time.Minute // newTestClient shrinks it; restore the long unit

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			start := time.Now()
			_, err := cc.Call(ctx, m1Req())
			elapsed := time.Since(start)
			if err == nil {
				t.Fatal("expected an error")
			}
			if elapsed > time.Second {
				t.Errorf("slept past the budget: returned after %v (deadline 2s)", elapsed)
			}
			tc.check(t, err)
		})
	}
}
