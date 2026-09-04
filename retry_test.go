package llm

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestRetryableStatus(t *testing.T) {
	for _, s := range []int{408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529} {
		if !retryableStatus(s) {
			t.Errorf("retryableStatus(%d) = false, want true", s)
		}
	}
	for _, s := range []int{400, 401, 403, 404, 422, 200, 525, 530} {
		if retryableStatus(s) {
			t.Errorf("retryableStatus(%d) = true, want false", s)
		}
	}
}

func TestParseRetryAfter_Seconds(t *testing.T) {
	now := time.Now()
	if d := parseRetryAfter("3", now); d != 3*time.Second {
		t.Fatalf("parseRetryAfter(\"3\") = %v, want 3s", d)
	}
	if d := parseRetryAfter("0", now); d != 0 {
		t.Fatalf("parseRetryAfter(\"0\") = %v, want 0", d)
	}
	if d := parseRetryAfter("-5", now); d != 0 {
		t.Fatalf("parseRetryAfter(\"-5\") = %v, want 0", d)
	}
}

func TestParseRetryAfter_HTTPDate(t *testing.T) {
	now := time.Now().UTC()
	future := now.Add(90 * time.Second)
	v := future.Format(http.TimeFormat)
	d := parseRetryAfter(v, now)
	if d <= 80*time.Second || d > 95*time.Second {
		t.Fatalf("parseRetryAfter(HTTP-date) = %v, want ~90s", d)
	}
	// A date in the past clamps to 0.
	past := now.Add(-time.Hour).Format(http.TimeFormat)
	if d := parseRetryAfter(past, now); d != 0 {
		t.Fatalf("parseRetryAfter(past) = %v, want 0", d)
	}
}

func TestParseRetryAfter_Garbage(t *testing.T) {
	if d := parseRetryAfter("soon", time.Now()); d != 0 {
		t.Fatalf("parseRetryAfter(\"soon\") = %v, want 0", d)
	}
	if d := parseRetryAfter("", time.Now()); d != 0 {
		t.Fatalf("parseRetryAfter(\"\") = %v, want 0", d)
	}
}

func TestParseRetryAfter_CappedAt120s(t *testing.T) {
	now := time.Now()
	if d := parseRetryAfter("100000", now); d != maxRetryAfter {
		t.Fatalf("parseRetryAfter(\"100000\") = %v, want cap %v", d, maxRetryAfter)
	}
	if d := parseRetryAfter("120", now); d != 120*time.Second {
		t.Fatalf("parseRetryAfter(\"120\") = %v, want 120s (exactly the cap)", d)
	}
	future := now.UTC().Add(time.Hour).Format(http.TimeFormat)
	if d := parseRetryAfter(future, now); d != maxRetryAfter {
		t.Fatalf("parseRetryAfter(HTTP-date +1h) = %v, want cap %v", d, maxRetryAfter)
	}
}

func TestBackoffDelay_CappedAndNonNegative(t *testing.T) {
	for attempt := 1; attempt <= 10; attempt++ {
		d := backoffDelay(attempt)
		if d < 0 {
			t.Fatalf("backoffDelay(%d) negative: %v", attempt, d)
		}
		if d > maxRetryBackoff+maxRetryBackoff/5 {
			t.Fatalf("backoffDelay(%d) = %v exceeds cap+jitter", attempt, d)
		}
	}
	if d := backoffDelay(9); d > maxRetryBackoff {
		t.Fatalf("backoffDelay(9) = %v, want capped at %v", d, maxRetryBackoff)
	}
}

func TestRetrySleep_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if retrySleep(ctx, time.Hour) {
		t.Fatal("cancelled context must not sleep through")
	}
	if !retrySleep(context.Background(), 0) {
		t.Fatal("zero delay on live context must return true")
	}
}

// ── SSE ──────────────────────────────────────────────────────────────────

func TestPumpSSE_MultilineAndDone(t *testing.T) {
	events := []string{}
	body := strings.NewReader(
		"event: message\n" +
			"data: {\"a\":\n" +
			"data: 1}\n\n" +
			": keepalive\n\n" +
			"data: [DONE]\n\n")
	err := pumpSSE(context.Background(), io.NopCloser(body), time.Minute, func(d []byte) error {
		events = append(events, string(d))
		return nil
	})
	if err != nil {
		t.Fatalf("pumpSSE: %v", err)
	}
	want := []string{"{\"a\":\n1}", "[DONE]"}
	if len(events) != len(want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
	for i := range want {
		if events[i] != want[i] {
			t.Errorf("events[%d] = %q, want %q", i, events[i], want[i])
		}
	}
}

func TestPumpSSE_CRLF(t *testing.T) {
	events := []string{}
	body := strings.NewReader("data: one\r\n\r\ndata: two\r\n\r\n")
	if err := pumpSSE(context.Background(), io.NopCloser(body), time.Minute, func(d []byte) error {
		events = append(events, string(d))
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 || events[0] != "one" || events[1] != "two" {
		t.Fatalf("events = %v, want [one two]", events)
	}
}

func TestPumpSSE_HandlerErrorStops(t *testing.T) {
	sentinel := io.NopCloser(strings.NewReader("data: x\n\ndata: y\n\n"))
	calls := 0
	err := pumpSSE(context.Background(), sentinel, time.Minute, func(d []byte) error {
		calls++
		return io.ErrUnexpectedEOF
	})
	if err != io.ErrUnexpectedEOF {
		t.Fatalf("err = %v, want ErrUnexpectedEOF", err)
	}
	if calls != 1 {
		t.Fatalf("handler calls = %d, want 1", calls)
	}
}

func TestPumpSSE_IdleTimeout(t *testing.T) {
	old := streamIdleTimeout
	streamIdleTimeout = 30 * time.Millisecond
	defer func() { streamIdleTimeout = old }()

	// A reader that never produces events.
	r, w := io.Pipe()
	defer r.Close()
	defer w.Close()
	start := time.Now()
	err := pumpSSE(context.Background(), r, streamIdleTimeout, func([]byte) error { return nil })
	if err != ErrIdleTimeout {
		t.Fatalf("err = %v, want ErrIdleTimeout", err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("idle timeout took %v, watchdog did not fire promptly", time.Since(start))
	}
}

func TestPumpSSE_KeepaliveResetsIdle(t *testing.T) {
	old := streamIdleTimeout
	streamIdleTimeout = 60 * time.Millisecond
	defer func() { streamIdleTimeout = old }()

	r, w := io.Pipe()
	defer r.Close()
	defer w.Close()
	go func() {
		for i := 0; i < 3; i++ {
			time.Sleep(30 * time.Millisecond)
			w.Write([]byte(": ka\n\n"))
		}
		w.Write([]byte("data: finally\n\n"))
		w.Close()
	}()
	var got []string
	if err := pumpSSE(context.Background(), r, streamIdleTimeout, func(d []byte) error {
		got = append(got, string(d))
		return nil
	}); err != nil {
		t.Fatalf("pumpSSE: %v", err)
	}
	if len(got) != 1 || got[0] != "finally" {
		t.Fatalf("got = %v, want [finally]", got)
	}
}
