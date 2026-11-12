package llm

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"testing/iotest"
	"time"
)

// Correctness / performance / compatibility review: regression tests.

// ── MIME: canonical lower-case on the wire for inline images ─────────────

// An inline image MIME type is accepted case-insensitively (and with
// surrounding whitespace, as mime.ParseMediaType allows), so the wire must
// carry the canonical form: providers reject "IMAGE/JPG".
func TestReview_InlineImageMIMECanonicalOnWire(t *testing.T) {
	cases := map[string]string{"IMAGE/PNG": "image/png", "Image/Jpeg": "image/jpeg", "IMAGE/JPG": "image/jpeg", " image/webp ": "image/webp", "image/gif;": "image/gif"}
	// Where each format carries the type, and the exact value expected there.
	wire := map[string]func(m map[string]any) any{
		"openai": func(m map[string]any) any {
			return m6bGet(t, m, "messages", 0, "content", 0, "image_url", "url")
		},
		"responses": func(m map[string]any) any { return m6bGet(t, m, "input", 0, "content", 0, "image_url") },
		"anthropic": func(m map[string]any) any {
			return m6bGet(t, m, "messages", 0, "content", 0, "source", "media_type")
		},
		"gemini": func(m map[string]any) any {
			return m6bGet(t, m, "contents", 0, "parts", 0, "inlineData", "mimeType")
		},
	}
	for in, want := range cases {
		for _, f := range m6bFormats {
			m, _, err := m6bBuild(f.format, f.model, m6bUser(ImagePart(in, []byte{1, 2, 3})))
			if err != nil {
				t.Errorf("%s %q: %v", f.name, in, err)
				continue
			}
			exact := want
			if f.name == "openai" || f.name == "responses" {
				exact = "data:" + want + ";base64,AQID"
			}
			if got := wire[f.name](m); got != exact {
				t.Errorf("%s %q: wire MIME = %q, want %q", f.name, in, got, exact)
			}
		}
	}
}

// ── Streaming: SSE detection ─────────────────────────────────────────────

const reviewSSE = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" + m1Finish + "data: [DONE]\n\n"

func reviewServer(t *testing.T, ctype, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always set the header: an absent value makes net/http sniff one.
		w.Header()["Content-Type"] = nil
		if ctype != "" {
			w.Header().Set("Content-Type", ctype)
		}
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A 2xx SSE body must be streamed whatever the Content-Type's case or
// parameters, and also when a gateway mislabels or omits the header: the
// body's framing decides. Buffered mode must not be learned from it.
func TestReview_StreamDetectsSSE(t *testing.T) {
	bodies := map[string]string{
		"plain":     reviewSSE,
		"bom":       "\xef\xbb\xbf" + reviewSSE,
		"keepalive": ": ping\n\n" + reviewSSE,
		"event":     "event: message\n" + reviewSSE,
		"blank":     "\r\n\n" + reviewSSE,
	}
	for _, ct := range []string{"Text/Event-Stream; Charset=UTF-8", "TEXT/EVENT-STREAM", "", "text/plain; charset=utf-8", "application/octet-stream", "application/json"} {
		for name, body := range bodies {
			t.Run(ct+"/"+name, func(t *testing.T) {
				srv := reviewServer(t, ct, body)
				cc := m1OpenAIClient(t, srv)
				var got strings.Builder
				res, err := cc.CallStream(context.Background(), m1Req(), func(d Delta) error { got.WriteString(d.Text); return nil })
				if err != nil {
					t.Fatalf("SSE body under Content-Type %q failed: %v", ct, err)
				}
				if res.Content != "hi" || got.String() != "hi" || res.FinishReason != FinishStop {
					t.Fatalf("result = %+v, deltas = %q", res, got.String())
				}
				if cc.pc.learn.forceBuffered.Load() {
					t.Fatal("an SSE body must not teach buffered mode")
				}
			})
		}
	}
}

// A JSON body is still parsed directly (and teaches buffered mode) with or
// without a Content-Type, leading whitespace included.
func TestReview_StreamNonSSEStillBuffered(t *testing.T) {
	for _, ct := range []string{"application/json", "", "text/plain"} {
		for _, lead := range []string{"", " \r\n\t"} {
			srv := reviewServer(t, ct, lead+m3Reply(FormatOpenAI))
			cc := m1OpenAIClient(t, srv)
			res, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error { t.Error("buffered body emitted a delta"); return nil })
			if err != nil || res.Content != "ok" {
				t.Fatalf("ctype %q lead %q: res=%+v err=%v", ct, lead, res, err)
			}
			if !cc.pc.learn.forceBuffered.Load() {
				t.Fatalf("ctype %q: buffered mode not learned", ct)
			}
		}
	}
}

func TestReview_SniffSSE(t *testing.T) {
	cases := map[string]bool{
		"data: x\n\n":                            true,
		"data:x":                                 true,
		"event: a\ndata: x":                      true,
		"id: 1\ndata: x":                         true,
		"retry: 100\n":                           true,
		": keepalive\n\n":                        true,
		"\n\n: keepalive":                        true,
		"\xef\xbb\xbfdata:x":                     true,
		`{"choices":[]}`:                         false,
		` [{"error":{}}]`:                        false,
		"":                                       false,
		"   ":                                    false,
		"data":                                   false, // truncated: not enough to tell
		"dat":                                    false,
		"database error":                         false,
		"error: boom":                            false,
		"<html>":                                 false,
		"\xef\xbb":                               false,
		"idle":                                   false,
		"retry later please":                     false,
		strings.Repeat(" ", 300) + "data: x\n\n": false, // past the sniff window: not guessed
	}
	for body, want := range cases {
		for _, wrap := range []func(io.Reader) io.Reader{func(r io.Reader) io.Reader { return r }, iotest.OneByteReader} {
			got, err := sniffSSE(newSSEReader(wrap(strings.NewReader(body))))
			if got != want {
				t.Errorf("sniffSSE(%q) = %v, want %v", body, got, want)
			}
			// An error is reported exactly when the body ended undecided.
			if undecided := undecidedSniff[body]; (err != nil) != undecided || (err != nil && err != io.EOF) {
				t.Errorf("sniffSSE(%q) err = %v, undecided = %v", body, err, undecided)
			}
		}
	}
}

// undecidedSniff lists the TestReview_SniffSSE bodies that end before a
// decisive byte: empty, whitespace only, inside a BOM, or a truncated field name.
var undecidedSniff = map[string]bool{"": true, "   ": true, "data": true, "dat": true, "\xef\xbb": true}

// A read error that lands inside a partial BOM is returned, not swallowed.
func TestReview_SniffSSEKeepsErrorInsideBOM(t *testing.T) {
	boom := errors.New("socket gone")
	for _, lead := range []string{"", "\xef", "\xef\xbb", " \n", "dat"} {
		sse, err := sniffSSE(newSSEReader(io.MultiReader(strings.NewReader(lead), iotest.ErrReader(boom))))
		if sse || !errors.Is(err, boom) {
			t.Errorf("lead %q: sse=%v err=%v, want the read error", lead, sse, err)
		}
	}
}

// ── SSE reader ───────────────────────────────────────────────────────────

// refSSELines is the specification the optimized reader must match: lines
// end with LF, CRLF or a bare CR; the final line may lack a terminator.
func refSSELines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	lines := strings.Split(s, "\n")
	return lines
}

func TestReview_SSELineReaderMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	alphabet := []string{"a", "data: x", "\n", "\r", "\r\n", ":", " ", strings.Repeat("z", 70_000)}
	readers := map[string]func(io.Reader) io.Reader{
		"whole":   func(r io.Reader) io.Reader { return r },
		"onebyte": iotest.OneByteReader,
		"half":    iotest.HalfReader,
		"dataerr": iotest.DataErrReader,
	}
	for i := 0; i < 60; i++ { // each round runs four readers, one of them byte by byte
		var sb strings.Builder
		for n := rng.IntN(12); n > 0; n-- {
			sb.WriteString(alphabet[rng.IntN(len(alphabet))])
		}
		in := sb.String()
		want := refSSELines(in)
		for name, wrap := range readers {
			lr := &sseLineReader{br: newSSEReader(wrap(strings.NewReader(in)))}
			var got []string
			for {
				line, err := lr.readSSELine()
				got = append(got, string(line))
				if err != nil {
					if err != io.EOF {
						t.Fatalf("%s: %q: unexpected error %v", name, in, err)
					}
					break
				}
			}
			if len(got) != len(want) {
				t.Fatalf("%s: %d lines, want %d (input %d bytes)", name, len(got), len(want), len(in))
			}
			for j := range want {
				if got[j] != want[j] {
					t.Fatalf("%s: line %d = %d bytes, want %d bytes", name, j, len(got[j]), len(want[j]))
				}
			}
		}
	}
}

// A single-line event may be as large as the event cap: providers send whole
// response objects (response.completed) on one data line.
func TestReview_SSESingleLineUpToEventCap(t *testing.T) {
	read := func(n int) sseItem {
		done := make(chan struct{})
		defer close(done)
		ch := make(chan sseItem, 2)
		go parseSSEStream(strings.NewReader("data: "+strings.Repeat("a", n)+"\n\n"), ch, done)
		return <-ch
	}
	if it := read(3 << 20); it.kind != sseData || len(it.data) != 3<<20 {
		t.Fatalf("3 MiB single-line event: kind %v, %d bytes, err %v", it.kind, len(it.data), it.err)
	}
	if it := read(sseMaxEventSize); it.kind != sseData || len(it.data) != sseMaxEventSize {
		t.Fatalf("event-cap single-line event: kind %v, %d bytes, err %v", it.kind, len(it.data), it.err)
	}
	if it := read(sseMaxEventSize + 1); it.kind != sseErr || !errors.Is(it.err, errSSEOversized) {
		t.Fatalf("over the cap: kind %v err %v, want errSSEOversized", it.kind, it.err)
	}
	// Far over the cap: rejected while still spilling, before any terminator.
	if it := read(sseMaxEventSize + 1<<20); it.kind != sseErr || !errors.Is(it.err, errSSEOversized) {
		t.Fatalf("far over the cap: kind %v err %v, want errSSEOversized", it.kind, it.err)
	}
}

// The parser must not hold a reference to a line after reading the next one
// (the reader reuses its buffers): every delivered event is its own copy.
func TestReview_SSEEventsDoNotAlias(t *testing.T) {
	var in strings.Builder
	for i := 0; i < 200; i++ {
		fmt.Fprintf(&in, "data: event-%03d\n\n", i)
	}
	var got [][]byte
	if err := pumpSSE(context.Background(), strings.NewReader(in.String()), 0, func(b []byte) error {
		got = append(got, b)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 200 {
		t.Fatalf("%d events, want 200", len(got))
	}
	for i, b := range got {
		if want := fmt.Sprintf("event-%03d", i); string(b) != want {
			t.Fatalf("event %d = %q, want %q (buffer aliasing)", i, b, want)
		}
	}
}

func BenchmarkPumpSSE(b *testing.B) {
	var sb strings.Builder
	for i := 0; i < 2000; i++ {
		fmt.Fprintf(&sb, "data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"tok%d \"},\"finish_reason\":null}]}\n\n", i)
	}
	sb.WriteString("data: [DONE]\n\n")
	payload := sb.String()
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	for b.Loop() {
		n := 0
		_ = pumpSSE(context.Background(), strings.NewReader(payload), 0, func([]byte) error { n++; return nil })
		if n != 2001 {
			b.Fatal(n)
		}
	}
}

// ── OpenAI-format gateways: tolerant response shapes ─────────────────────

func TestReview_OpenAIContentAsParts(t *testing.T) {
	const msg = `{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"plan "},{"type":"text","text":"more"}]},{"type":"text","text":"hel"},{"type":"text","text":"lo"}]}`
	res, err := parseOpenAIResponse([]byte(`{"choices":[{"message":` + msg + `,"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("array content failed the parse: %v", err)
	}
	if res.Content != "hello" || res.ReasoningContent != "plan more" {
		t.Fatalf("content %q reasoning %q", res.Content, res.ReasoningContent)
	}

	acc := newStreamAccum()
	var kinds []DeltaKind
	for _, chunk := range []string{
		`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"plan"}]}]}}]}`,
		`{"choices":[{"delta":{"content":[{"type":"text","text":"hel"}]}}]}`,
		`{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
	} {
		ds, _, err := mapOpenAIStreamEvent([]byte(chunk), acc)
		if err != nil {
			t.Fatalf("stream chunk %s: %v", chunk, err)
		}
		for _, d := range ds {
			kinds = append(kinds, d.Kind)
		}
	}
	if r := acc.result(); r.Content != "hello" || r.ReasoningContent != "plan" {
		t.Fatalf("stream content %q reasoning %q", r.Content, r.ReasoningContent)
	}
	if len(kinds) != 3 || kinds[0] != DeltaReasoning || kinds[1] != DeltaContent || kinds[2] != DeltaContent {
		t.Fatalf("delta kinds = %v", kinds)
	}

	// reasoning_details without usable prose fall through to thinking chunks.
	res, err = parseOpenAIResponse([]byte(`{"choices":[{"message":{"reasoning_details":[{"type":"reasoning.encrypted","data":"x"}],"content":[{"type":"thinking","thinking":"T"},{"type":"text","text":"x"}]},"finish_reason":"stop"}]}`))
	if err != nil || res.ReasoningContent != "T" || res.Content != "x" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	// An explicit reasoning field still wins over typed thinking chunks.
	res, err = parseOpenAIResponse([]byte(`{"choices":[{"message":{"reasoning_content":"R","content":[{"type":"thinking","thinking":"T"},{"type":"text","text":"x"}]},"finish_reason":"stop"}]}`))
	if err != nil || res.ReasoningContent != "R" || res.Content != "x" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	// Unknown content shapes are skipped, never fatal.
	for _, c := range []string{`null`, `{"a":1}`, `7`, `[1,2]`, `[{"type":"image_url","image_url":{"url":"x"}}]`} {
		res, err = parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":` + c + `},"finish_reason":"stop"}]}`))
		if err != nil || res.Content != "" {
			t.Fatalf("content %s: res=%+v err=%v", c, res, err)
		}
	}
}

func TestReview_OpenAIToolArgumentsAsObject(t *testing.T) {
	res, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":{"x":1}}},{"id":"b","type":"function","function":{"name":"g","arguments":null}}]},"finish_reason":"tool_calls"}]}`))
	if err != nil {
		t.Fatalf("object arguments failed the parse: %v", err)
	}
	if len(res.ToolCalls) != 2 || res.ToolCalls[0].Arguments != `{"x":1}` || res.ToolCalls[1].Arguments != "{}" {
		t.Fatalf("tool calls = %+v", res.ToolCalls)
	}
	// Any other value is surfaced as sent, never replaced by an invented "{}".
	for _, v := range []string{`[1,2]`, `5`, `true`} {
		res, err = parseOpenAIResponse([]byte(`{"choices":[{"message":{"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":` + v + `}}]},"finish_reason":"tool_calls"}]}`))
		if err != nil || len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != v {
			t.Fatalf("arguments %s: res=%+v err=%v", v, res, err)
		}
	}
	acc := newStreamAccum()
	ds, _, err := mapOpenAIStreamEvent([]byte(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"f","arguments":{"x":1}}}]},"finish_reason":"tool_calls"}]}`), acc)
	if err != nil {
		t.Fatalf("stream object arguments: %v", err)
	}
	if r := acc.result(); len(ds) != 1 || len(r.ToolCalls) != 1 || r.ToolCalls[0].Arguments != `{"x":1}` {
		t.Fatalf("deltas=%+v calls=%+v", ds, r.ToolCalls)
	}
}

// A usage quirk (float or string counts, wrong container) never fails a
// generation that was already produced and billed.
func TestReview_OpenAIUsageTolerant(t *testing.T) {
	cases := []struct {
		usage string
		want  Usage
	}{
		{`{"prompt_tokens":12.0,"completion_tokens":3,"completion_tokens_details":{"reasoning_tokens":2.0}}`, Usage{PromptTokens: 12, CompletionTokens: 3, ReasoningTokens: 2}},
		{`{"prompt_tokens":"12","completion_tokens":"3"}`, Usage{PromptTokens: 12, CompletionTokens: 3}},
		{`{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":4.0}}`, Usage{PromptTokens: 6, CompletionTokens: 1, CacheReadTokens: 4, CachedTokens: 4, CacheReported: true}},
		{`{"prompt_tokens":null,"completion_tokens":true,"prompt_tokens_details":null}`, Usage{}},
		{`{"prompt_tokens":-5,"completion_tokens":1e400}`, Usage{}},
		{`[]`, Usage{}},
		{`"n/a"`, Usage{}},
		{`{"prompt_tokens":7,"completion_tokens":2,"prompt_tokens_details":[]}`, Usage{PromptTokens: 7, CompletionTokens: 2}},
	}
	for _, tc := range cases {
		res, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],"usage":` + tc.usage + `}`))
		if err != nil {
			t.Errorf("usage %s failed the parse: %v", tc.usage, err)
			continue
		}
		if res.Usage != tc.want {
			t.Errorf("usage %s = %+v, want %+v", tc.usage, res.Usage, tc.want)
		}
		acc := newStreamAccum()
		if _, _, err := mapOpenAIStreamEvent([]byte(`{"choices":[],"usage":`+tc.usage+`}`), acc); err != nil {
			t.Errorf("stream usage %s: %v", tc.usage, err)
		} else if acc.usage != tc.want {
			t.Errorf("stream usage %s = %+v, want %+v", tc.usage, acc.usage, tc.want)
		}
	}
}

// The "error" field of a buffered 2xx body: an object with a message is a
// provider error; a bare string is one only when there is no completion;
// "", false, 0 and null never fail a good response.
func TestReview_OpenAIBufferedErrorField(t *testing.T) {
	const ok = `"choices":[{"message":{"content":"x"},"finish_reason":"stop"}]`
	for _, e := range []string{`""`, `false`, `0`, `null`, `{}`, `"note"`} {
		res, err := parseOpenAIResponse([]byte(`{"error":` + e + `,` + ok + `}`))
		if err != nil || res.Content != "x" {
			t.Errorf("error=%s failed a good response: res=%+v err=%v", e, res, err)
		}
	}
	for _, body := range []string{`{"error":"boom"}`, `{"error":{"message":"boom"}}`, `{"error":{"message":"boom"},` + ok + `}`} {
		if _, err := parseOpenAIResponse([]byte(body)); err == nil || !strings.Contains(err.Error(), "provider error: boom") {
			t.Errorf("%s: err = %v, want provider error: boom", body, err)
		}
	}
}

func TestReview_HTTPErrorStringEnvelope(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "gw", Format: FormatOpenAI, BaseURL: "https://x"}, nil, nil)
	for body, want := range map[string]string{
		`{"error":"model not found"}`:                          "model not found",
		`{"error":{"message":"nested","code":"c"}}`:            "nested",
		`{"message":"top","code":"c"}`:                         "top",
		`{"error":{"message":"num code","code":404}}`:          "num code",
		`{"error":"","message":"fallback"}`:                    "fallback",
		`not json`:                                             "not json",
		`{"error":{"type":"x"}}`:                               `{"error":{"type":"x"}}`,
		`{"detail":"unparsed envelope stays the raw preview"}`: `{"detail":"unparsed envelope stays the raw preview"}`,
	} {
		if e := pc.httpError(400, []byte(body)); e.Message != want {
			t.Errorf("%s: Message = %q, want %q", body, e.Message, want)
		}
	}
}

// ── Responses API streaming: items delivered without deltas ──────────────

// A gateway may deliver a function call's arguments (or the answer text)
// only on the terminal item/response events. They must not be dropped: an
// empty-object argument set would silently run the tool with no input.
func TestReview_ResponsesStreamBackfill(t *testing.T) {
	run := func(events ...string) (*ChatResult, []Delta) {
		t.Helper()
		acc := newStreamAccum()
		var all []Delta
		for _, ev := range events {
			ds, _, err := mapResponsesStreamEvent([]byte(ev), acc)
			if err != nil {
				t.Fatalf("%s: %v", ev, err)
			}
			all = append(all, ds...)
		}
		return acc.result(), all
	}
	const call = `{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"x\":1}"}`

	// Arguments only on output_item.done.
	res, ds := run(
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}`,
		`{"type":"response.output_item.done","output_index":0,"item":`+call+`}`,
		`{"type":"response.completed","response":{"status":"completed","output":[`+call+`]}}`,
	)
	if len(res.ToolCalls) != 1 || res.ToolCalls[0] != (ToolCall{ID: "c1", Name: "f", Arguments: `{"x":1}`}) || res.FinishReason != FinishToolCalls {
		t.Fatalf("done-only: %+v", res)
	}
	if len(ds) != 1 || ds[0].Kind != DeltaToolArgs || ds[0].Text != `{"x":1}` || ds[0].ToolID != "c1" || ds[0].ToolName != "f" || ds[0].ToolIndex != 0 {
		t.Fatalf("done-only deltas: %+v", ds)
	}

	// Nothing but the terminal event: call and text come from response.output.
	res, ds = run(`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]},` + call + `]}}`)
	if res.Content != "hello" || len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != `{"x":1}` || res.FinishReason != FinishToolCalls {
		t.Fatalf("final-only: %+v", res)
	}
	if len(ds) != 2 || ds[0].Kind != DeltaContent || ds[0].Text != "hello" || ds[1].Kind != DeltaToolArgs {
		t.Fatalf("final-only deltas: %+v", ds)
	}

	// A refusal that only the terminal event carries.
	res, ds = run(`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"no"}]}]}}`)
	if res.Refusal != "no" || res.FinishReason != FinishContentFilter || len(ds) != 1 || ds[0].Kind != DeltaRefusal {
		t.Fatalf("final-only refusal: %+v deltas %+v", res, ds)
	}

	// Streamed deltas are never duplicated by the terminal events.
	res, ds = run(
		`{"type":"response.output_text.delta","delta":"hel"}`,
		`{"type":"response.output_text.delta","delta":"lo"}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"x\":"}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"1}"}`,
		`{"type":"response.output_item.done","output_index":1,"item":`+call+`}`,
		`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]},`+call+`]}}`,
	)
	if res.Content != "hello" || len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != `{"x":1}` {
		t.Fatalf("streamed: %+v", res)
	}
	if len(ds) != 4 {
		t.Fatalf("streamed: %d deltas, want 4 (no duplicates): %+v", len(ds), ds)
	}
}

// ── Tool arguments: null input is an empty object ────────────────────────

func TestReview_NullToolInputIsEmptyObject(t *testing.T) {
	res, err := parseAnthropicResponse([]byte(`{"content":[{"type":"tool_use","id":"t","name":"f","input":null}],"stop_reason":"tool_use"}`))
	if err != nil || len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != "{}" {
		t.Fatalf("anthropic: %+v %v", res, err)
	}
	res, err = parseGeminiResponse([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":null}}]},"finishReason":"STOP"}]}`))
	if err != nil || len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != "{}" {
		t.Fatalf("gemini: %+v %v", res, err)
	}
}

// ── Retry ladder: permanent transport failures fail fast ─────────────────

// A TLS certificate that does not verify cannot be fixed by retrying: the
// ladder must stop after one attempt instead of backing off for ~90s.
func TestReview_CertificateErrorNotRetried(t *testing.T) {
	fastBackoff(t)
	var hits atomic.Int32
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	srv.Config.ErrorLog = log.New(io.Discard, "", 0) // the refused handshakes are the point
	srv.StartTLS()
	t.Cleanup(srv.Close)

	var attempts atomic.Int32
	sdk := New(
		WithProvider("gw", WithFormat(FormatOpenAI), WithBaseURL(srv.URL), WithAPIKey("k")),
		WithAttemptObserver(func(AttemptEvent) { attempts.Add(1) }),
	)
	cc, err := sdk.Chat("gw", "m")
	if err != nil {
		t.Fatal(err)
	}
	for name, call := range map[string]func() error{
		"buffered": func() error { _, err := cc.Call(context.Background(), m1Req()); return err },
		"stream": func() error {
			_, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error { return nil })
			return err
		},
		"embed": func() error {
			_, err := sdk.Embed(context.Background(), "gw", "m", EmbedRequest{Inputs: []string{"x"}})
			return err
		},
	} {
		attempts.Store(0)
		err := call()
		var cv *tls.CertificateVerificationError
		if !errors.As(err, &cv) {
			t.Fatalf("%s: err = %v, want a certificate verification error", name, err)
		}
		if strings.Contains(err.Error(), "retry exhausted") {
			t.Errorf("%s: reported as retry exhaustion: %v", name, err)
		}
		if n := attempts.Load(); n != 1 {
			t.Errorf("%s: %d attempts, want 1", name, n)
		}
	}
	if hits.Load() != 0 {
		t.Fatalf("handler reached %d times despite the failed handshake", hits.Load())
	}
}

// Ordinary transport failures (connection refused) are still retried.
func TestReview_TransientTransportErrorStillRetried(t *testing.T) {
	fastBackoff(t)
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close() // nothing listens any more
	var attempts atomic.Int32
	sdk := New(
		WithProvider("gw", WithFormat(FormatOpenAI), WithBaseURL(url), WithAPIKey("k")),
		WithRetryPolicy(RetryPolicy{MaxAttempts: 3}),
		WithAttemptObserver(func(AttemptEvent) { attempts.Add(1) }),
	)
	cc, _ := sdk.Chat("gw", "m")
	if _, err := cc.Call(context.Background(), m1Req()); err == nil || !strings.Contains(err.Error(), "retry exhausted (3 attempts)") {
		t.Fatalf("err = %v", err)
	}
	if n := attempts.Load(); n != 3 {
		t.Fatalf("%d attempts, want 3", n)
	}
}

// ── Transport ────────────────────────────────────────────────────────────

func TestReview_PooledTransportBounds(t *testing.T) {
	tr := newPooledTransport()
	if tr.TLSHandshakeTimeout <= 0 || tr.TLSHandshakeTimeout > 30*time.Second {
		t.Errorf("TLSHandshakeTimeout = %v: a stalled handshake must not consume the whole call budget", tr.TLSHandshakeTimeout)
	}
	if tr.HTTP2 == nil || tr.HTTP2.SendPingTimeout <= 0 || tr.HTTP2.PingTimeout <= 0 {
		t.Errorf("HTTP2 = %+v: dead HTTP/2 connections must be detected by health pings", tr.HTTP2)
	}
	if tr.MaxIdleConnsPerHost < 10 || tr.MaxIdleConns < tr.MaxIdleConnsPerHost {
		t.Errorf("idle pool = %d total / %d per host", tr.MaxIdleConns, tr.MaxIdleConnsPerHost)
	}
	if !tr.DisableCompression || !tr.ForceAttemptHTTP2 || tr.Proxy == nil {
		t.Errorf("existing transport contract changed: %+v", tr)
	}
}

// ── Anthropic ────────────────────────────────────────────────────────────

// Message.Cache puts the breakpoint on the message's FIRST block, whatever
// its type. A message opening with an image or document used to be left
// unmarked silently; text-first messages keep their exact wire, so a stable
// leading part stays cached while the rest of the message varies.
func TestReview_AnthropicCacheMarksFirstBlock(t *testing.T) {
	cases := map[string][]ContentPart{
		"image then text":    {ImagePart("image/png", []byte{1}), TextPart("describe")},
		"text then image":    {TextPart("describe"), ImagePart("image/png", []byte{1})},
		"document then text": {DocumentPart("application/pdf", []byte{1}, "a.pdf"), TextPart("summarize")},
		"context then query": {TextPart("big stable context"), TextPart("question")},
		"single text part":   {TextPart("hello")},
	}
	for name, parts := range cases {
		b, err := buildAnthropicRequest(&ChatRequest{Messages: []Message{{Role: RoleUser, Cache: true, Parts: parts}}}, "claude", false)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var wire struct {
			Messages []struct {
				Content []map[string]any `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(b, &wire); err != nil {
			t.Fatal(err)
		}
		for i, blk := range wire.Messages[0].Content {
			if _, marked := blk["cache_control"]; marked != (i == 0) {
				t.Errorf("%s: block %d cache_control=%v, want the marker on the first block only: %s", name, i, marked, b)
			}
		}
	}
	// Without Cache nothing is marked; a plain-Content message keeps its marker.
	b, _ := buildAnthropicRequest(&ChatRequest{Messages: []Message{{Role: RoleUser, Parts: cases["image then text"]}, {Role: RoleUser, Content: "x", Cache: true}}}, "claude", false)
	if strings.Count(string(b), "cache_control") != 1 || !strings.Contains(string(b), `{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}`) {
		t.Errorf("wire = %s", b)
	}
}

// model_context_window_exceeded truncates the generation at a token limit:
// canonical "length" on both paths, like max_tokens.
func TestReview_AnthropicContextWindowExceededIsLength(t *testing.T) {
	res, err := parseAnthropicResponse([]byte(`{"content":[{"type":"text","text":"par"}],"stop_reason":"model_context_window_exceeded"}`))
	if err != nil || res.FinishReason != FinishLength {
		t.Fatalf("buffered: %+v %v", res, err)
	}
	acc := newStreamAccum()
	if _, _, err := mapAnthropicStreamEvent([]byte(`{"type":"message_delta","delta":{"stop_reason":"model_context_window_exceeded"}}`), acc); err != nil {
		t.Fatal(err)
	}
	if !acc.sawFinish || acc.finishReason != FinishLength {
		t.Fatalf("stream: finish %q sawFinish %v", acc.finishReason, acc.sawFinish)
	}
}

// Anthropic reports the reasoning subset of output_tokens as
// usage.output_tokens_details.thinking_tokens (on the final message_delta
// when streaming). It is a subset: CompletionTokens is unchanged.
func TestReview_AnthropicThinkingTokens(t *testing.T) {
	res, err := parseAnthropicResponse([]byte(`{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":40,"output_tokens_details":{"thinking_tokens":30}}}`))
	if err != nil || res.Usage.ReasoningTokens != 30 || res.Usage.CompletionTokens != 40 || res.Usage.PromptTokens != 5 {
		t.Fatalf("buffered: %+v %v", res, err)
	}
	acc := newStreamAccum()
	for _, ev := range []string{
		`{"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":1}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":40,"output_tokens_details":{"thinking_tokens":30}}}`,
		`{"type":"message_delta","delta":{},"usage":{"output_tokens":41}}`, // a sparse delta never zeroes it
	} {
		if _, _, err := mapAnthropicStreamEvent([]byte(ev), acc); err != nil {
			t.Fatal(err)
		}
	}
	if acc.usage.ReasoningTokens != 30 || acc.usage.CompletionTokens != 41 || acc.usage.PromptTokens != 5 {
		t.Fatalf("stream: %+v", acc.usage)
	}
	// Unreported stays zero.
	res, _ = parseAnthropicResponse([]byte(`{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":40}}`))
	if res.Usage.ReasoningTokens != 0 {
		t.Fatalf("unreported thinking tokens = %d", res.Usage.ReasoningTokens)
	}
}

// ── Panel review: wire-compat regressions ────────────────────────────────

// Array-shaped content is folded in linear time: a hostile body nesting
// thinking chunks thousands deep must not buy seconds of uncancellable CPU
// (the request timeout is a whole-call budget). Only one level of nesting
// carries reasoning prose, so deeper levels are not descended into.
func TestReview_ContentChunksNestingIsLinear(t *testing.T) {
	// Bounds are on allocated bytes, not wall-clock time: quadratic folding
	// re-copied the payload at every level (gigabytes here), and an
	// allocation count does not move with the load on a CI runner.
	allocated := func(fn func()) uint64 {
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		fn()
		runtime.ReadMemStats(&after)
		return after.TotalAlloc - before.TotalAlloc
	}
	nest := func(depth int, leaf string) string {
		return strings.Repeat(`[{"type":"thinking","thinking":`, depth) + `"` + leaf + `"` + strings.Repeat(`}]`, depth)
	}

	// Deep: one small chunk nested 4000 levels (132 KB).
	acc := newStreamAccum()
	if n := allocated(func() {
		if _, _, err := mapOpenAIStreamEvent([]byte(`{"choices":[{"delta":{"content":`+nest(4000, "x")+`}}]}`), acc); err != nil {
			t.Fatal(err)
		}
	}); n > 16<<20 {
		t.Fatalf("132 KB chunk nested 4000 deep allocated %d MiB", n>>20)
	}
	if r := acc.result(); r.Content != "" || r.ReasoningContent != "" {
		t.Fatalf("nested thinking below the first level leaked: %+v", r)
	}
	// Wide and deep: a 3 MiB body nested 1000 levels.
	if n := allocated(func() {
		if _, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":` + nest(1000, strings.Repeat("x", 3<<20)) + `},"finish_reason":"stop"}]}`)); err != nil {
			t.Fatal(err)
		}
	}); n > 256<<20 {
		t.Fatalf("3 MiB body nested 1000 deep allocated %d MiB", n>>20)
	}
	// One level (the documented Mistral shape) still folds.
	res, err := parseOpenAIResponse([]byte(`{"choices":[{"message":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"plan"},{"type":"thinking","thinking":"deeper"}]},{"type":"text","text":"x"}]},"finish_reason":"stop"}]}`))
	if err != nil || res.ReasoningContent != "plan" || res.Content != "x" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
}

// Framework-default error envelopes ({"error":"Bad Request","message":…})
// keep their detail message: learn-once and billing classification read it.
func TestReview_HTTPErrorBareStringDoesNotShadowMessage(t *testing.T) {
	pc := newProviderClient(ProviderConfig{ID: "gw", Format: FormatOpenAI, BaseURL: "https://x"}, nil, nil)
	e := pc.httpError(400, []byte(`{"error":"Bad Request","message":"stream_options is not supported","statusCode":400}`))
	if e.Message != "stream_options is not supported" || !streamOptionsRejected(e) {
		t.Errorf("stream_options: %+v", e)
	}
	e = pc.httpError(400, []byte(`{"statusCode":400,"message":"Streaming is not supported for this route","error":"Bad Request"}`))
	if !streamRejected(e) {
		t.Errorf("streaming: %+v", e)
	}
	e = pc.httpError(429, []byte(`{"timestamp":"t","status":429,"error":"Too Many Requests","message":"insufficient_quota for tenant","path":"/v1"}`))
	if e.Retryable || !billingExhausted(e) {
		t.Errorf("billing: %+v", e)
	}
	// A non-string type or code never costs the message.
	e = pc.httpError(400, []byte(`{"error":{"message":"m","type":123,"code":7}}`))
	if e.Message != "m" || e.Code != "" {
		t.Errorf("typed: %+v", e)
	}
	e = pc.httpError(400, []byte(`{"error":{"message":"m","code":"c"}}`))
	if e.Message != "m" || e.Code != "c" {
		t.Errorf("nested code: %+v", e)
	}
	e = pc.httpError(400, []byte(`{"message":"top","code":"tc"}`))
	if e.Message != "top" || e.Code != "tc" {
		t.Errorf("top-level code: %+v", e)
	}
}

// The backfill identifies a function call by its call_id, not by its
// position in response.output: a gateway whose output_index differs from the
// final array position must not end up with the call twice.
func TestReview_ResponsesBackfillNeverDuplicatesCalls(t *testing.T) {
	const c1 = `{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"x\":1}"}`
	const c2 = `{"type":"function_call","call_id":"c2","name":"g","arguments":"{\"y\":2}"}`
	run := func(events ...string) (*ChatResult, []Delta) {
		t.Helper()
		acc := newStreamAccum()
		var all []Delta
		for _, ev := range events {
			ds, _, err := mapResponsesStreamEvent([]byte(ev), acc)
			if err != nil {
				t.Fatalf("%s: %v", ev, err)
			}
			all = append(all, ds...)
		}
		return acc.result(), all
	}
	// Events carry no output_index (0); the final output has a reasoning item first.
	res, ds := run(
		`{"type":"response.output_item.added","item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","delta":"{\"x\":1}"}`,
		`{"type":"response.output_item.done","item":`+c1+`}`,
		`{"type":"response.completed","response":{"status":"completed","output":[{"type":"reasoning","summary":[]},`+c1+`]}}`,
	)
	if len(res.ToolCalls) != 1 || len(ds) != 1 {
		t.Fatalf("mismatched index duplicated the call: calls=%+v deltas=%+v", res.ToolCalls, ds)
	}
	// A different call already sits at the final array position: the
	// unstreamed call is appended, never merged into it or dropped.
	res, ds = run(
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"x\":1}"}`,
		`{"type":"response.completed","response":{"status":"completed","output":[`+c1+`,`+c2+`]}}`,
	)
	if len(res.ToolCalls) != 2 || res.ToolCalls[0].ID != "c1" || res.ToolCalls[1] != (ToolCall{ID: "c2", Name: "g", Arguments: `{"y":2}`}) {
		t.Fatalf("calls = %+v", res.ToolCalls)
	}
	if len(ds) != 2 || ds[1].ToolIndex != 1 || ds[1].ToolID != "c2" {
		t.Fatalf("deltas = %+v", ds)
	}
	// output_item.done for a call announced under another index.
	res, _ = run(
		`{"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}`,
		`{"type":"response.output_item.done","output_index":0,"item":`+c1+`}`,
		`{"type":"response.completed","response":{"status":"completed","output":[`+c1+`]}}`,
	)
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Arguments != `{"x":1}` {
		t.Fatalf("done under another index: %+v", res.ToolCalls)
	}
}

// The wire MIME is the parsed media type: whatever validation accepts
// (case, whitespace, a trailing ';') is normalized the same way.
func TestReview_CanonMIMEMatchesValidation(t *testing.T) {
	for in, want := range map[string]string{
		"image/png;": "image/png", "image/jpg;": "image/jpeg", " IMAGE/PNG ; ": "image/png",
		"application/pdf;": "application/pdf", "TEXT/Plain;": "text/plain", "image/webp": "image/webp",
		"": "", " Not A Type ": "not a type", // unparseable input is only lower-cased and trimmed
	} {
		if got := canonMIME(in); got != want {
			t.Errorf("canonMIME(%q) = %q, want %q", in, got, want)
		}
	}
	m, _, err := m6bBuild(FormatAnthropic, "claude", m6bUser(DocumentPart("text/plain;", []byte("hello"), "n.txt")))
	if err != nil {
		t.Fatal(err)
	}
	jsonEq(t, "anthropic", m6bGet(t, m, "messages", 0, "content", 0),
		`{"type":"document","source":{"type":"text","media_type":"text/plain","data":"hello"},"title":"n.txt"}`)
	if _, _, err := m6bBuild(FormatOpenAI, "gpt-4o", m6bUser(DocumentPart("application/pdf;", m6bPDF, "a.pdf"))); err != nil {
		t.Errorf("openai rejected a PDF it had validated: %v", err)
	}
}

// ── Panel review: streaming regressions ──────────────────────────────────

// A cancel or deadline that lands while the SDK is still waiting for the
// first body byte of an unlabelled 2xx keeps its identity: the sniff must
// not swallow the read error and re-read a dead body.
func TestReview_SniffKeepsContextErrorIdentity(t *testing.T) {
	for _, ct := range []string{"", "application/json"} {
		for _, mode := range []string{"deadline", "cancel"} {
			t.Run(ct+"/"+mode, func(t *testing.T) {
				release := make(chan struct{})
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header()["Content-Type"] = nil
					if ct != "" {
						w.Header().Set("Content-Type", ct)
					}
					w.WriteHeader(http.StatusOK)
					w.(http.Flusher).Flush() // headers out, no body byte yet
					select {
					case <-r.Context().Done():
					case <-release:
					}
				}))
				t.Cleanup(srv.Close)
				t.Cleanup(func() { close(release) })
				cc := m1OpenAIClient(t, srv)
				cc.pc.opts.retry = RetryPolicy{MaxAttempts: 1}

				ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
				want := context.DeadlineExceeded
				if mode == "cancel" {
					ctx, cancel = context.WithCancel(context.Background())
					want = context.Canceled
					time.AfterFunc(150*time.Millisecond, cancel)
				}
				defer cancel()
				_, err := cc.CallStream(ctx, m1Req(), func(Delta) error { return nil })
				if !errors.Is(err, want) {
					t.Fatalf("err = %v, want errors.Is %v", err, want)
				}
				// A declared type teaches buffered mode by itself (as it
				// always did); an unlabelled body that never produced a byte
				// teaches nothing.
				if got := cc.pc.learn.forceBuffered.Load(); got != (ct != "") {
					t.Fatalf("forceBuffered = %v under Content-Type %q", got, ct)
				}
			})
		}
	}
}

// An unlabelled 2xx whose body is empty is still a terminal parse error (as
// for any non-SSE body), but it decides nothing about the provider.
func TestReview_EmptyUnlabelledBodyTeachesNothing(t *testing.T) {
	srv := reviewServer(t, "", "")
	cc := m1OpenAIClient(t, srv)
	if _, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error { return nil }); err == nil {
		t.Fatal("empty body returned no error")
	}
	if cc.pc.learn.forceBuffered.Load() {
		t.Fatal("an empty body must not teach buffered mode")
	}
}

// Every line ending costs the same: a stream of bare-CR terminators must not
// rescan the whole read buffer per line.
func TestReview_SSEBareCRIsNotQuadratic(t *testing.T) {
	const n = 512 << 10
	read := func(term string) time.Duration {
		best := time.Duration(1 << 62)
		for range 3 { // best of three: a ratio of minimums shrugs off load spikes
			lr := &sseLineReader{br: newSSEReader(strings.NewReader(strings.Repeat(term, n)))}
			start := time.Now()
			lines := 0
			for {
				if _, err := lr.readSSELine(); err != nil {
					break
				}
				lines++
			}
			if lines != n {
				t.Fatalf("%q: %d lines, want %d", term, lines, n)
			}
			best = min(best, time.Since(start))
		}
		return best
	}
	// Rescanning the buffer per line made CR 3x slower than LF under the race
	// detector and ~30x without it; the two now cost the same.
	if lf, cr := read("\n"), read("\r"); cr > 2*lf+50*time.Millisecond {
		t.Fatalf("bare CR took %v against %v for LF", cr, lf)
	}
	// A far LF behind many CRs is found once, not once per line.
	lr := &sseLineReader{br: newSSEReader(strings.NewReader(strings.Repeat("\r", 60_000) + "\n" + "x\n"))}
	for i := 0; i < 60_000; i++ {
		if line, err := lr.readSSELine(); err != nil || len(line) != 0 {
			t.Fatalf("line %d: %q %v", i, line, err)
		}
	}
	if line, err := lr.readSSELine(); err != nil || string(line) != "x" {
		t.Fatalf("after the CR run (CR LF pair skipped): %q %v", line, err)
	}
}

// ── Panel review: per-call lookups stay constant-time ────────────────────

// reviewScaling runs fn at size n and 4n and fails when the larger run costs
// more than a linear algorithm could (4x, with 2x headroom plus a constant):
// a quadratic one costs 16x. A ratio does not move with machine load the way
// an absolute bound does.
func reviewScaling(t *testing.T, name string, n int, fn func(n int)) {
	t.Helper()
	timed := func(n int) time.Duration {
		best := time.Duration(1 << 62)
		for range 2 {
			start := time.Now()
			fn(n)
			best = min(best, time.Since(start))
		}
		return best
	}
	if small, large := timed(n), timed(4*n); large > 8*small+100*time.Millisecond {
		t.Fatalf("%s: %d items took %v, %d items took %v (super-linear)", name, n, small, 4*n, large)
	}
}

// One terminal Responses event may list any number of function calls: each
// is resolved without rescanning the calls already known.
func TestReview_ResponsesBackfillScalesLinearly(t *testing.T) {
	event := func(n int, id func(i int) string) []byte {
		var b strings.Builder
		b.WriteString(`{"type":"response.completed","response":{"status":"completed","output":[`)
		for i := 0; i < n; i++ {
			if i > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, `{"type":"function_call","call_id":%q,"name":"f","arguments":"{}"}`, id(i))
		}
		b.WriteString(`]}}`)
		return []byte(b.String())
	}
	distinct := func(i int) string { return fmt.Sprintf("c%d", i) }
	reviewScaling(t, "distinct call ids", 10_000, func(n int) {
		acc := newStreamAccum()
		if _, _, err := mapResponsesStreamEvent(event(n, distinct), acc); err != nil {
			t.Fatal(err)
		}
		if len(acc.calls) != n {
			t.Fatalf("%d calls, want %d", len(acc.calls), n)
		}
	})
	// Every final item collides with a different identified call at its index.
	reviewScaling(t, "colliding indices", 10_000, func(n int) {
		acc := newStreamAccum()
		for i := 0; i < n; i++ {
			acc.setCallID(acc.call(i), fmt.Sprintf("s%d", i))
		}
		if _, _, err := mapResponsesStreamEvent(event(n, distinct), acc); err != nil {
			t.Fatal(err)
		}
		if len(acc.calls) != 2*n {
			t.Fatalf("%d calls, want %d", len(acc.calls), 2*n)
		}
	})

	// The id index follows a call whose id changes after it was announced.
	acc := newStreamAccum()
	for _, ev := range []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"tmp","name":"f","arguments":""}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"c1","name":"f","arguments":""}}`,
		`{"type":"response.output_item.done","output_index":5,"item":{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"x\":1}"}}`,
		`{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"x\":1}"}]}}`,
	} {
		if _, _, err := mapResponsesStreamEvent([]byte(ev), acc); err != nil {
			t.Fatal(err)
		}
	}
	if r := acc.result(); len(r.ToolCalls) != 1 || r.ToolCalls[0] != (ToolCall{ID: "c1", Name: "f", Arguments: `{"x":1}`}) {
		t.Fatalf("re-identified call: %+v", r.ToolCalls)
	}
}

// Index-less chat-completions tool-call deltas with new ids each start a
// call; finding the next free key must not rescan every call so far.
func TestReview_OpenAIIndexlessToolCallsScaleLinearly(t *testing.T) {
	reviewScaling(t, "index-less tool calls", 10_000, func(n int) {
		var b strings.Builder
		b.WriteString(`{"choices":[{"delta":{"tool_calls":[`)
		for i := 0; i < n; i++ {
			if i > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, `{"id":"c%d","function":{"name":"f","arguments":"{}"}}`, i)
		}
		b.WriteString(`]}}]}`)
		acc := newStreamAccum()
		if _, _, err := mapOpenAIStreamEvent([]byte(b.String()), acc); err != nil {
			t.Fatal(err)
		}
		if len(acc.calls) != n || acc.calls[n-1].id != fmt.Sprintf("c%d", n-1) || acc.calls[n-1].pos != n-1 {
			t.Fatalf("%d calls, last %+v", len(acc.calls), acc.calls[len(acc.calls)-1])
		}
	})
}
