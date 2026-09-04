package llm

import (
	"testing"
	"time"
)

// SetStreamIdleTimeout pins the setter contract: positive values apply,
// zero and negative are ignored so a misconfigured caller cannot disable
// the watchdog by accident.
func TestSetStreamIdleTimeout(t *testing.T) {
	orig := StreamIdleTimeout()
	t.Cleanup(func() { streamIdleTimeout = orig })

	SetStreamIdleTimeout(5 * time.Second)
	if got := StreamIdleTimeout(); got != 5*time.Second {
		t.Fatalf("StreamIdleTimeout() = %v, want 5s", got)
	}

	SetStreamIdleTimeout(0)
	SetStreamIdleTimeout(-1 * time.Second)
	if got := StreamIdleTimeout(); got != 5*time.Second {
		t.Fatalf("non-positive override applied; StreamIdleTimeout() = %v, want 5s", got)
	}
}
