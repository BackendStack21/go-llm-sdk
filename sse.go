package llm

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"strings"
	"time"
)

// SSE (Server-Sent Events) reading with an idle watchdog. The reader runs in
// a goroutine feeding a channel; the pump selects between events, keepalive
// activity, the idle timer, and context cancellation. Keepalive comment
// lines reset the idle timer without producing an event.

const (
	sseMaxEventSize = 4 << 20 // 4 MiB per accumulated data event
	// A single line may carry a whole event: providers send complete response
	// objects (e.g. response.completed) on one data line. The slack covers the
	// field name; the event cap still bounds the payload.
	sseMaxLineSize = sseMaxEventSize + 64
	sseReadBuffer  = 64 * 1024
)

// newSSEReader wraps r in the buffered reader the SSE parser works on. An
// existing reader of sufficient size (an already-sniffed body) is reused.
func newSSEReader(r io.Reader) *bufio.Reader { return bufio.NewReaderSize(r, sseReadBuffer) }

// isEventStream reports whether a Content-Type names text/event-stream.
// Media types are case-insensitive and may carry parameters.
func isEventStream(contentType string) bool {
	return strings.Contains(strings.ToLower(contentType), "text/event-stream")
}

// sseBOM is the UTF-8 byte order mark a stream may open with.
const sseBOM = "\xef\xbb\xbf"

// sseFieldPrefixes are the field names that can open an SSE stream.
var sseFieldPrefixes = []string{"data:", "event:", "id:", "retry:"}

// sniffSSE reports whether a 2xx body that was not labelled
// text/event-stream is SSE framing anyway (gateways that drop or rewrite the
// Content-Type). It peeks without consuming: after an optional BOM and
// blank lines, an SSE stream opens with a comment (':') or a field name,
// none of which can begin a JSON document. Anything else — JSON included —
// is not SSE.
//
// A non-nil error means the body ended (io.EOF) or failed before a decisive
// byte arrived: nothing was learned about the body, and the error is the
// read's own — a cancellation or deadline keeps its identity, which a second
// read of the dead body would not return.
func sniffSSE(br *bufio.Reader) (bool, error) {
	const maxSniff = 256
	for n := 1; n <= maxSniff; n++ {
		s, err := br.Peek(n)
		if len(s) < len(sseBOM) && strings.HasPrefix(sseBOM, string(s)) {
			if err != nil {
				return false, err // ended empty or inside a BOM: undecided
			}
			continue // possibly a partial BOM
		}
		s = bytes.TrimLeft(bytes.TrimPrefix(s, []byte(sseBOM)), " \t\r\n")
		if len(s) > 0 {
			if s[0] == ':' {
				return true, nil
			}
			partial := false
			for _, f := range sseFieldPrefixes {
				if bytes.HasPrefix(s, []byte(f)) {
					return true, nil
				}
				partial = partial || strings.HasPrefix(f, string(s))
			}
			if !partial {
				return false, nil
			}
		}
		if err != nil {
			return false, err
		}
	}
	return false, nil // nothing decisive within the window: not guessed
}

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
	lr := &sseLineReader{br: newSSEReader(r)}
	var data []byte
	sawActivity := false
	first := true
	for {
		line, err := lr.readSSELine()
		if first {
			// A leading UTF-8 BOM is not part of the first field name.
			line = bytes.TrimPrefix(line, []byte(sseBOM))
			first = false
		}
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

// sseLineReader splits SSE input on \n, \r\n or a bare \r (all three are
// line endings per the SSE spec). A \r at the end of the buffered data
// never blocks waiting for a possible \n: the pending-LF flag skips it when
// it arrives.
//
// It scans whole buffered chunks instead of single bytes and allocates only
// for a line that straddles a buffer refill, so a typical event costs no
// reader allocation at all.
type sseLineReader struct {
	br      *bufio.Reader
	skipLF  bool
	pending int    // bytes of br backing the last returned line, consumed on the next call
	buf     []byte // spill for a line that spans buffer refills; reused
	// noLF counts the leading buffered bytes known to hold no LF, so the
	// search for one never rescans them: without it a run of bare-CR lines
	// would scan the whole read buffer once per line.
	noLF int
}

// consume discards n buffered bytes (it cannot fail: they are buffered).
func (lr *sseLineReader) consume(n int) {
	_, _ = lr.br.Discard(n)
	lr.noLF = max(lr.noLF-n, 0)
}

// readSSELine reads one line and returns it without its terminator. The
// returned slice is only valid until the next call. Lines longer than
// sseMaxLineSize fail with errSSEOversized.
func (lr *sseLineReader) readSSELine() ([]byte, error) {
	if lr.pending > 0 {
		lr.consume(lr.pending)
		lr.pending = 0
	}
	lr.buf = lr.buf[:0]
	for {
		// Block for the next byte, then take everything already buffered.
		// The buffer only refills once it is empty, so between refills the
		// buffered data just shrinks from the front and noLF stays exact.
		if _, err := lr.br.Peek(1); err != nil {
			return lr.buf, err // EOF or read error; trailing line lacks a terminator
		}
		chunk, _ := lr.br.Peek(lr.br.Buffered())
		if lr.skipLF {
			lr.skipLF = false
			if chunk[0] == '\n' {
				lr.consume(1)
				continue
			}
		}
		// First terminator: an LF, or a CR that precedes it.
		end := -1
		if i := bytes.IndexByte(chunk[lr.noLF:], '\n'); i >= 0 {
			end = lr.noLF + i
			lr.noLF = end
		} else {
			lr.noLF = len(chunk)
		}
		window := chunk
		if end >= 0 {
			window = chunk[:end]
		}
		if cr := bytes.IndexByte(window, '\r'); cr >= 0 {
			end = cr
		}
		if end < 0 {
			// No terminator buffered yet: spill and refill.
			if len(lr.buf)+len(chunk) > sseMaxLineSize {
				return nil, errSSEOversized
			}
			lr.buf = append(lr.buf, chunk...)
			lr.consume(len(chunk))
			continue
		}
		if len(lr.buf)+end > sseMaxLineSize {
			return nil, errSSEOversized
		}
		lr.skipLF = chunk[end] == '\r'
		if len(lr.buf) == 0 {
			// Fast path: the line sits whole in the read buffer. Hand it out
			// in place and consume it on the next call.
			lr.pending = end + 1
			return chunk[:end], nil
		}
		lr.buf = append(lr.buf, chunk[:end]...)
		lr.consume(end + 1)
		return lr.buf, nil
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
func pumpSSE(ctx context.Context, r io.Reader, idle time.Duration, handle func(data []byte) error) error {
	ch := make(chan sseItem, 8)
	done := make(chan struct{})
	defer close(done) // always release the parser, even on abort/timeout
	go parseSSEStream(r, ch, done)

	var timer *time.Timer
	var timeout <-chan time.Time
	if idle > 0 {
		// No drain-before-Reset dance is needed: since Go 1.23 a Timer's
		// channel never delivers stale values after Reset (go.mod: 1.25).
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
				// Only provider silence counts: the consumer's handle time
				// is excluded, so the timer restarts after handle returns.
				if err := handle(it.data); err != nil {
					return err
				}
				if timer != nil {
					timer.Reset(idle)
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
