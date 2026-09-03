package llm

import (
	"bufio"
	"context"
	"io"
	"time"
)

// SSE (Server-Sent Events) reading with an idle watchdog. The reader runs in
// a goroutine feeding a channel; the pump selects between events, keepalive
// activity, the idle timer, and context cancellation. Keepalive comment
// lines reset the idle timer without producing an event.

const (
	sseMaxLineSize  = 1 << 20 // 1 MiB per line
	sseMaxEventSize = 4 << 20 // 4 MiB per accumulated data event
)

type sseItemKind byte

const (
	sseData sseItemKind = iota
	sseKeepalive
	sseEnd
	sseErr
)

type sseItem struct {
	kind sseItemKind
	data []byte
	err  error
}

// parseSSEStream reads SSE frames from r and emits items on ch. It closes ch
// when done. A data event is every accumulated "data:" line set terminated
// by a blank line; comment lines (':') and retry fields count as keepalive
// activity. Runs in its own goroutine; terminates when r errors (e.g. the
// caller closes the response body).
func parseSSEStream(r io.Reader, ch chan<- sseItem) {
	defer close(ch)
	br := bufio.NewReaderSize(r, 64*1024)
	var data []byte
	sawActivity := false
	for {
		line, err := readSSELine(br)
		if len(line) > 0 {
			sawActivity = true
			switch {
			case line[0] == ':':
				// comment / keepalive
			case hasSSEPrefix(line, "data:"):
				payload := trimSSEPayload(line[5:])
				if len(data)+len(payload) > sseMaxEventSize {
					ch <- sseItem{kind: sseErr, err: errSSEOversized}
					return
				}
				if len(data) > 0 {
					data = append(data, '\n')
				}
				data = append(data, payload...)
			}
		}
		if err != nil {
			if err == io.EOF && len(data) == 0 {
				ch <- sseItem{kind: sseEnd}
				return
			}
			if err == io.EOF {
				// Trailing event without blank line: deliver it.
				ch <- sseItem{kind: sseData, data: data}
				ch <- sseItem{kind: sseEnd}
				return
			}
			ch <- sseItem{kind: sseErr, err: err}
			return
		}
		if len(line) == 0 {
			// Blank line: event boundary.
			if len(data) > 0 {
				ch <- sseItem{kind: sseData, data: data}
				data = nil
			} else if sawActivity {
				ch <- sseItem{kind: sseKeepalive}
				sawActivity = false
			}
		}
	}
}

// readSSELine reads one line including the trailing newline; returns the
// line without it (also strips a leading \r for CRLF frames).
func readSSELine(br *bufio.Reader) ([]byte, error) {
	line, err := br.ReadSlice('\n')
	if len(line) > sseMaxLineSize {
		return nil, errSSEOversized
	}
	l := len(line)
	if l > 0 && line[l-1] == '\n' {
		l--
		if l > 0 && line[l-1] == '\r' {
			l--
		}
	}
	return line[:l], err
}

func hasSSEPrefix(line []byte, prefix string) bool {
	return len(line) >= len(prefix) && string(line[:len(prefix)]) == prefix
}

// trimSSEPayload strips one optional leading space after the field name.
func trimSSEPayload(b []byte) []byte {
	if len(b) > 0 && b[0] == ' ' {
		return b[1:]
	}
	return b
}

type sseError struct{ msg string }

func (e *sseError) Error() string { return "llm: sse: " + e.msg }

var errSSEOversized = &sseError{"frame exceeds size limit"}

// pumpSSE drives an SSE stream with an idle watchdog. handle is invoked for
// every data event; returning an error stops the pump and is returned to
// the caller. idle <= 0 disables the watchdog. The stream ends cleanly
// (nil error) on EOF without a trailing partial event.
func pumpSSE(ctx context.Context, r io.ReadCloser, idle time.Duration, handle func(data []byte) error) error {
	ch := make(chan sseItem, 8)
	go parseSSEStream(r, ch)

	var timer *time.Timer
	var timeout <-chan time.Time
	if idle > 0 {
		timer = time.NewTimer(idle)
		defer timer.Stop()
		timeout = timer.C
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout:
			return ErrIdleTimeout
		case it, ok := <-ch:
			if !ok {
				return nil
			}
			switch it.kind {
			case sseData:
				if timer != nil {
					timer.Reset(idle)
				}
				if err := handle(it.data); err != nil {
					return err
				}
			case sseKeepalive:
				if timer != nil {
					timer.Reset(idle)
				}
			case sseEnd:
				return nil
			case sseErr:
				return it.err
			}
		}
	}
}
