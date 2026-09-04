package llm

import (
	"context"
	"errors"
	"io"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Regression: pumpSSE returning early (consumer abort, cancellation, idle
// timeout) must signal the parser goroutine to exit. Without the signal the
// parser blocks forever on a channel send once the 8-slot buffer fills —
// one leaked goroutine per aborted stream.
func TestPumpSSEAbortDoesNotLeakParser(t *testing.T) {
	events := strings.Repeat("data: x\n\n", 500) // far more than the 8-slot buffer
	before := runtime.NumGoroutine()
	err := pumpSSE(context.Background(), io.NopCloser(strings.NewReader(events)), 0, func([]byte) error {
		return errors.New("abort now")
	})
	if err == nil {
		t.Fatal("expected the handler abort to surface")
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.NumGoroutine() <= before {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("goroutine leak: before=%d after=%d (parser stuck on channel send)", before, runtime.NumGoroutine())
}

// Regression: a single data line larger than the 64 KiB bufio buffer must
// still be delivered (up to the 1 MiB line cap); anything larger must be
// rejected with errSSEOversized.
func TestParseSSEStreamLongLines(t *testing.T) {
	line := strings.Repeat("a", 100*1024) // 100 KiB: over the bufio buffer, under the cap
	done := make(chan struct{})
	ch := make(chan sseItem, 2)
	go parseSSEStream(strings.NewReader("data: "+line+"\n\n"), ch, done)
	it := <-ch
	if it.kind == sseErr {
		t.Fatalf("100KiB single-line event failed: %v (oversized=%v)", it.err, errors.Is(it.err, errSSEOversized))
	}
	if it.kind != sseData || len(it.data) != len(line) {
		t.Fatalf("data item = kind %v, %d bytes, want sseData with %d bytes", it.kind, len(it.data), len(line))
	}

	huge := strings.Repeat("b", (1<<20)+1) // 1 MiB + 1: over the line cap
	ch2 := make(chan sseItem, 2)
	go parseSSEStream(strings.NewReader("data: "+huge+"\n\n"), ch2, done)
	it2 := <-ch2
	if it2.kind != sseErr || !errors.Is(it2.err, errSSEOversized) {
		t.Fatalf("oversized line: kind %v err %v, want errSSEOversized", it2.kind, it2.err)
	}
}
