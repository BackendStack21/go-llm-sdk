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

// parseSSEStream reads SSE frames from r and emits items on ch. It closes
// ch when done. A data event is every accumulated "data:" line set
// terminated by a blank line; comment lines (':') and retry fields count as
// keepalive activity. Runs in its own goroutine; terminates when r errors
// (e.g. the caller closes the response body) or the consumer closes done —
// a pump that exits early (abort, cancel, timeout) must never strand this
// goroutine on a channel send.
func parseSSEStream(r io.Reader, ch chan<- sseItem, done <-chan struct{}) {
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
					sendSSE(ch, done, sseItem{kind: sseErr, err: errSSEOversized})
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
				sendSSE(ch, done, sseItem{kind: sseEnd})
				return
			}
			if err == io.EOF {
				// Trailing event without blank line: deliver it.
				if sendSSE(ch, done, sseItem{kind: sseData, data: data}) {
					sendSSE(ch, done, sseItem{kind: sseEnd})
				}
				return
			}
			sendSSE(ch, done, sseItem{kind: sseErr, err: err})
			return
		}
		if len(line) == 0 {
			// Blank line: event boundary.
			if len(data) > 0 {
				if !sendSSE(ch, done, sseItem{kind: sseData, data: data}) {
					return
				}
				data = nil
				sawActivity = false
			} else if sawActivity {
				if !sendSSE(ch, done, sseItem{kind: sseKeepalive}) {
					return
				}
				sawActivity = false
			}
		}
	}
}

// sendSSE delivers one item unless the consumer abandoned the stream
// (done closed); reports whether delivery succeeded.
func sendSSE(ch chan<- sseItem, done <-chan struct{}, it sseItem) bool {
	select {
	case ch <- it:
		return true
	case <-done:
		return false
	}
}

// readSSELine reads one line including the trailing newline; returns the
// line without it (also strips a trailing \r\n for CRLF frames). Lines
// longer than the bufio buffer are accumulated across ReadSlice calls,
// bounded by sseMaxLineSize.
func readSSELine(br *bufio.Reader) ([]byte, error) {
	var buf []byte
	for {
		frag, err := br.ReadSlice('\n')
		if len(buf)+len(frag) > sseMaxLineSize {
			return nil, errSSEOversized
		}
		buf = append(buf, frag...)
		if err == bufio.ErrBufferFull {
			continue // line spans the buffer boundary; keep accumulating
		}
		if err != nil {
			return buf, err // EOF or read error; trailing line lacks '\n'
		}
		l := len(buf)
		if l > 0 && buf[l-1] == '\n' {
			l--
			if l > 0 && buf[l-1] == '\r' {
				l--
			}
		}
		return buf[:l], nil
	}
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
	done := make(chan struct{})
	defer close(done) // always release the parser, even on abort/timeout
	go parseSSEStream(r, ch, done)

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
