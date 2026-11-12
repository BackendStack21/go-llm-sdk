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
	cases := map[string]string{"IMAGE/PNG": "image/png", "Image/Jpeg": "image/jpeg", "IMAGE/JPG": "image/jpeg", " image/webp ": "image/webp"}
	for in, want := range cases {
		for _, f := range m6bFormats {
			m, _, err := m6bBuild(f.format, f.model, m6bUser(ImagePart(in, []byte{1, 2, 3})))
			if err != nil {
				t.Errorf("%s %q: %v", f.name, in, err)
				continue
			}
			b, _ := json.Marshal(m)
			if strings.Contains(string(b), strings.TrimSpace(in)) && strings.TrimSpace(in) != want {
				t.Errorf("%s %q: non-canonical MIME reached the wire: %s", f.name, in, b)
			}
			if !strings.Contains(string(b), want) {
				t.Errorf("%s %q: wire lacks %q: %s", f.name, in, want, b)
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
			if got := sniffSSE(newSSEReader(wrap(strings.NewReader(body)))); got != want {
				t.Errorf("sniffSSE(%q) = %v, want %v", body, got, want)
			}
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
	for i := 0; i < 300; i++ {
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
