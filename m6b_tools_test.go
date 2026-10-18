package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// m6b SL02: RunTools (AC-3).

// toolScript is a scripted multi-turn provider: response i answers request i;
// every request body is recorded.
type toolScript struct {
	mu     sync.Mutex
	bodies []map[string]any
	paths  []string
	resp   []string
}

func newToolServer(t *testing.T, resp ...string) (*httptest.Server, *toolScript) {
	t.Helper()
	s := &toolScript{resp: resp}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(raw, &m)
		s.mu.Lock()
		i := len(s.bodies)
		s.bodies = append(s.bodies, m)
		s.paths = append(s.paths, r.URL.Path)
		s.mu.Unlock()
		if i >= len(s.resp) {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"message":"script exhausted"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, s.resp[i])
	}))
	t.Cleanup(srv.Close)
	return srv, s
}

func (s *toolScript) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.bodies)
}

func (s *toolScript) body(t *testing.T, i int) map[string]any {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	if i >= len(s.bodies) {
		t.Fatalf("only %d requests recorded, want index %d", len(s.bodies), i)
	}
	return s.bodies[i]
}

// toolFormat describes one wire format's scripted responses.
type toolFormat struct {
	name   string
	format Format
	model  string
	// callTurn answers with the given calls; stop/length answer without calls.
	call   func(calls ...[2]string) string // {id,name}
	stop   string
	length string
	// replay checks the second request body against the first assistant turn.
	replay func(t *testing.T, body map[string]any)
}

func openaiCalls(calls ...[2]string) string {
	var cs []string
	for _, c := range calls {
		cs = append(cs, fmt.Sprintf(`{"id":%q,"type":"function","function":{"name":%q,"arguments":"{}"}}`, c[0], c[1]))
	}
	return `{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[` + strings.Join(cs, ",") + `]},"finish_reason":"tool_calls"}]}`
}

func responsesCalls(calls ...[2]string) string {
	var cs []string
	for _, c := range calls {
		cs = append(cs, fmt.Sprintf(`{"type":"function_call","call_id":%q,"name":%q,"arguments":"{}"}`, c[0], c[1]))
	}
	return `{"status":"completed","output":[` + strings.Join(cs, ",") + `]}`
}

func anthropicCalls(calls ...[2]string) string {
	cs := []string{`{"type":"thinking","thinking":"plan A","signature":"SIG-A"}`, `{"type":"redacted_thinking","data":"ENC-B"}`}
	for _, c := range calls {
		cs = append(cs, fmt.Sprintf(`{"type":"tool_use","id":%q,"name":%q,"input":{}}`, c[0], c[1]))
	}
	return `{"role":"assistant","content":[` + strings.Join(cs, ",") + `],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
}

func geminiCalls(calls ...[2]string) string {
	var cs []string
	for i, c := range calls {
		sig := ""
		if i == 0 {
			sig = `,"thoughtSignature":"GSIG-1"`
		}
		cs = append(cs, fmt.Sprintf(`{"functionCall":{"name":%q,"args":{}}%s}`, c[1], sig))
	}
	return `{"candidates":[{"content":{"role":"model","parts":[` + strings.Join(cs, ",") + `]},"finishReason":"STOP"}]}`
}

var toolFormats = []toolFormat{
	{
		name: "openai", format: FormatOpenAI, model: "gpt-4o", call: openaiCalls,
		stop:   `{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`,
		length: `{"choices":[{"message":{"role":"assistant","content":"cut"},"finish_reason":"length"}]}`,
		replay: func(t *testing.T, b map[string]any) {
			msgs := b["messages"].([]any)
			asst := msgs[1].(map[string]any)
			tcs, _ := asst["tool_calls"].([]any)
			if len(tcs) == 0 {
				t.Fatalf("assistant tool_calls not replayed: %v", asst)
			}
			if first := tcs[0].(map[string]any); first["id"] != "c1" {
				t.Errorf("first tool call id = %v", first["id"])
			}
		},
	},
	{
		name: "responses", format: FormatOpenAI, model: "gpt-5.6", call: responsesCalls,
		stop:   `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`,
		length: `{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"cut"}]}]}`,
		replay: func(t *testing.T, b map[string]any) {
			var types []string
			var ids []string
			for _, it := range b["input"].([]any) {
				m := it.(map[string]any)
				types = append(types, fmt.Sprint(m["type"]))
				if m["type"] == "function_call_output" {
					ids = append(ids, fmt.Sprint(m["call_id"]))
				}
			}
			joined := strings.Join(types, ",")
			if !strings.Contains(joined, "function_call,") && !strings.HasSuffix(joined, "function_call") && !strings.Contains(joined, "function_call,function_call_output") {
				t.Errorf("input item types = %s, want function_call then function_call_output", joined)
			}
			if len(ids) == 0 || ids[0] != "c1" {
				t.Errorf("function_call_output ids = %v", ids)
			}
		},
	},
	{
		name: "anthropic", format: FormatAnthropic, model: "claude-x", call: anthropicCalls,
		stop:   `{"role":"assistant","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}`,
		length: `{"role":"assistant","content":[{"type":"text","text":"cut"}],"stop_reason":"max_tokens"}`,
		replay: func(t *testing.T, b map[string]any) {
			msgs := b["messages"].([]any)
			asst := msgs[1].(map[string]any)
			blocks := asst["content"].([]any)
			if len(blocks) < 3 {
				t.Fatalf("assistant blocks = %v", blocks)
			}
			b0, b1, b2 := blocks[0].(map[string]any), blocks[1].(map[string]any), blocks[2].(map[string]any)
			if b0["type"] != "thinking" || b0["signature"] != "SIG-A" || b0["thinking"] != "plan A" {
				t.Errorf("block 0 = %v, want signed thinking first", b0)
			}
			if b1["type"] != "redacted_thinking" || b1["data"] != "ENC-B" {
				t.Errorf("block 1 = %v, want redacted_thinking", b1)
			}
			if b2["type"] != "tool_use" || b2["id"] != "c1" {
				t.Errorf("block 2 = %v, want tool_use c1", b2)
			}
			// tool results ride one user message, in call order.
			user := msgs[2].(map[string]any)
			if user["role"] != "user" {
				t.Errorf("results role = %v", user["role"])
			}
		},
	},
	{
		name: "gemini", format: FormatGemini, model: "gemini-x", call: geminiCalls,
		stop:   `{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}]}`,
		length: `{"candidates":[{"content":{"role":"model","parts":[{"text":"cut"}]},"finishReason":"MAX_TOKENS"}]}`,
		replay: func(t *testing.T, b map[string]any) {
			contents := b["contents"].([]any)
			model := contents[1].(map[string]any)
			parts := model["parts"].([]any)
			p0 := parts[0].(map[string]any)
			if _, ok := p0["functionCall"]; !ok || p0["thoughtSignature"] != "GSIG-1" {
				t.Errorf("first model part = %v, want functionCall with thoughtSignature GSIG-1", p0)
			}
		},
	},
}

func (f toolFormat) client(t *testing.T, srv *httptest.Server) *ChatClient {
	t.Helper()
	cc := newTestClient(t, ProviderConfig{ID: string(f.format), Format: f.format, BaseURL: srv.URL, APIKey: "k"}, srv)
	cc.model = f.model
	return cc
}

func (f toolFormat) req() ChatRequest {
	r := ChatRequest{
		Model:    f.model,
		Messages: []Message{{Role: RoleUser, Content: "go"}},
		Tools:    []ToolDef{{Name: "a", Parameters: json.RawMessage(`{"type":"object"}`)}, {Name: "b", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	if f.format == FormatAnthropic {
		r.Thinking = "enabled"
		r.MaxTokens = 4096
	}
	return r
}

func mustLoop(t *testing.T, res *ChatResult, msgs []Message, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("RunTools: %v", err)
	}
	if res == nil {
		t.Fatal("RunTools returned a nil result")
	}
	if len(msgs) == 0 {
		t.Fatal("RunTools returned an empty transcript")
	}
}

func echoHandler(_ context.Context, c ToolCall) (string, error) { return "ok-" + c.Name, nil }

// idOK compares a tool-call id; Gemini has no call ids, the SDK synthesises
// them, so only the other formats pin the scripted id.
func (f toolFormat) idOK(got, want string) bool { return f.format == FormatGemini || got == want }

func TestRunTools_TwoTurnPerFormat(t *testing.T) {
	for _, f := range toolFormats {
		t.Run(f.name, func(t *testing.T) {
			srv, sc := newToolServer(t, f.call([2]string{"c1", "a"}, [2]string{"c2", "b"}), f.stop)
			cc := f.client(t, srv)
			req := f.req()
			orig := append([]Message(nil), req.Messages...)
			var turns []int
			res, msgs, err := RunTools(context.Background(), cc, req,
				map[string]ToolHandler{"a": echoHandler, "b": echoHandler},
				ToolLoopOptions{OnTurn: func(n int, r *ChatResult) {
					if r == nil {
						t.Error("OnTurn got nil result")
					}
					turns = append(turns, n)
				}})
			mustLoop(t, res, msgs, err)
			if res.Content != "done" || res.FinishReason != FinishStop {
				t.Errorf("final = %q / %q", res.Content, res.FinishReason)
			}
			if sc.count() != 2 {
				t.Fatalf("requests = %d, want 2", sc.count())
			}
			f.replay(t, sc.body(t, 1))
			if !reflect.DeepEqual(turns, []int{1, 2}) {
				t.Errorf("OnTurn turns = %v, want [1 2]", turns)
			}
			// transcript = original + [assistant, tool, tool, assistant]
			if len(msgs) != 5 {
				t.Fatalf("transcript len = %d: %+v", len(msgs), msgs)
			}
			if !reflect.DeepEqual(msgs[0], orig[0]) {
				t.Errorf("transcript[0] = %+v", msgs[0])
			}
			if msgs[1].Role != RoleAssistant || len(msgs[1].ToolCalls) != 2 {
				t.Errorf("transcript[1] = %+v", msgs[1])
			}
			for i, id := range []string{"c1", "c2"} {
				m := msgs[2+i]
				if m.Role != RoleTool || !f.idOK(m.ToolCallID, id) || m.Content != "ok-"+m.ToolName || m.IsError {
					t.Errorf("transcript[%d] = %+v", 2+i, m)
				}
			}
			if msgs[2].ToolName != "a" || msgs[3].ToolName != "b" {
				t.Errorf("tool names = %q %q", msgs[2].ToolName, msgs[3].ToolName)
			}
			if msgs[4].Role != RoleAssistant || msgs[4].Content != "done" {
				t.Errorf("transcript[4] = %+v", msgs[4])
			}
			// request value and caller slice are never mutated
			if !reflect.DeepEqual(req.Messages, orig) || len(req.Messages) != 1 {
				t.Errorf("caller messages mutated: %+v", req.Messages)
			}
		})
	}
}

func TestRunTools_AnthropicReplaysBlocksFromTranscript(t *testing.T) {
	f := toolFormats[2]
	srv, _ := newToolServer(t, f.call([2]string{"c1", "a"}), f.stop)
	res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(),
		map[string]ToolHandler{"a": echoHandler}, ToolLoopOptions{})
	mustLoop(t, res, msgs, err)
	want := []ThinkingBlock{{Text: "plan A", Signature: "SIG-A"}, {Redacted: "ENC-B"}}
	if !reflect.DeepEqual(msgs[1].ThinkingBlocks, want) {
		t.Errorf("transcript thinking blocks = %+v, want %+v", msgs[1].ThinkingBlocks, want)
	}
}

func TestRunTools_HandlerErrorAndUnknownTool(t *testing.T) {
	for _, f := range toolFormats {
		t.Run(f.name, func(t *testing.T) {
			srv, sc := newToolServer(t, f.call([2]string{"c1", "a"}, [2]string{"c2", "nope"}, [2]string{"c3", "b"}), f.stop)
			handlers := map[string]ToolHandler{
				"a": func(context.Context, ToolCall) (string, error) { return "partial", errors.New("boom") },
				"b": nil, // nil handler counts as unknown
			}
			res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(), handlers, ToolLoopOptions{})
			mustLoop(t, res, msgs, err)
			if sc.count() != 2 {
				t.Fatalf("loop must continue after tool errors: requests = %d", sc.count())
			}
			if len(msgs) != 6 {
				t.Fatalf("transcript len = %d: %+v", len(msgs), msgs)
			}
			if m := msgs[2]; !m.IsError || m.Content != "boom" || !f.idOK(m.ToolCallID, "c1") {
				t.Errorf("handler error result = %+v", m)
			}
			for i, id := range []string{"c2", "c3"} {
				if m := msgs[3+i]; !m.IsError || m.Content != "unknown tool" || !f.idOK(m.ToolCallID, id) {
					t.Errorf("unknown tool result %d = %+v", i, m)
				}
			}
		})
	}
}

func TestRunTools_HandlerPanicBecomesErrorResult(t *testing.T) {
	f := toolFormats[0]
	srv, _ := newToolServer(t, f.call([2]string{"c1", "a"}), f.stop)
	res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(),
		map[string]ToolHandler{"a": func(context.Context, ToolCall) (string, error) { panic("kaboom") }}, ToolLoopOptions{})
	mustLoop(t, res, msgs, err)
	if m := msgs[2]; !m.IsError || !strings.Contains(m.Content, "tool a panicked") || !strings.Contains(m.Content, "kaboom") {
		t.Errorf("panic result = %+v", m)
	}
}

func TestRunTools_ParallelRunsConcurrentlyInCallOrder(t *testing.T) {
	for _, f := range toolFormats {
		t.Run(f.name, func(t *testing.T) {
			srv, _ := newToolServer(t, f.call([2]string{"c1", "a"}, [2]string{"c2", "b"}), f.stop)
			var started sync.WaitGroup
			started.Add(2)
			release := make(chan struct{})
			go func() {
				started.Wait()
				close(release)
			}()
			h := func(_ context.Context, c ToolCall) (string, error) {
				started.Done()
				select {
				case <-release: // both handlers started before either returns
				case <-time.After(3 * time.Second):
					return "", errors.New("handlers did not overlap")
				}
				if c.Name == "a" {
					time.Sleep(20 * time.Millisecond) // finish out of call order
				}
				return "r-" + c.Name, nil
			}
			res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(),
				map[string]ToolHandler{"a": h, "b": h}, ToolLoopOptions{Parallel: true})
			mustLoop(t, res, msgs, err)
			if len(msgs) != 5 {
				t.Fatalf("transcript len = %d", len(msgs))
			}
			for i, id := range []string{"c1", "c2"} {
				m := msgs[2+i]
				if !f.idOK(m.ToolCallID, id) || m.Content != "r-"+m.ToolName || m.IsError {
					t.Errorf("result %d = %+v", i, m)
				}
			}
		})
	}
}

func TestRunTools_SequentialByDefault(t *testing.T) {
	f := toolFormats[0]
	srv, _ := newToolServer(t, f.call([2]string{"c1", "a"}, [2]string{"c2", "b"}), f.stop)
	var cur, peak int32
	h := func(_ context.Context, c ToolCall) (string, error) {
		n := atomic.AddInt32(&cur, 1)
		for {
			p := atomic.LoadInt32(&peak)
			if n <= p || atomic.CompareAndSwapInt32(&peak, p, n) {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
		atomic.AddInt32(&cur, -1)
		return "x", nil
	}
	res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(),
		map[string]ToolHandler{"a": h, "b": h}, ToolLoopOptions{})
	mustLoop(t, res, msgs, err)
	if peak != 1 {
		t.Errorf("peak concurrency = %d, want 1", peak)
	}
}

func TestRunTools_NonToolFinishEndsLoop(t *testing.T) {
	for _, f := range toolFormats {
		for _, fin := range []struct{ name, body, want string }{
			{"stop", f.stop, FinishStop}, {"length", f.length, FinishLength},
		} {
			t.Run(f.name+"/"+fin.name, func(t *testing.T) {
				srv, sc := newToolServer(t, fin.body)
				var calls int32
				h := func(context.Context, ToolCall) (string, error) { atomic.AddInt32(&calls, 1); return "x", nil }
				var turns int
				res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(),
					map[string]ToolHandler{"a": h, "b": h}, ToolLoopOptions{OnTurn: func(int, *ChatResult) { turns++ }})
				mustLoop(t, res, msgs, err)
				if res.FinishReason != fin.want {
					t.Errorf("finish = %q, want %q", res.FinishReason, fin.want)
				}
				if sc.count() != 1 || turns != 1 || atomic.LoadInt32(&calls) != 0 {
					t.Errorf("requests=%d turns=%d handler calls=%d, want 1/1/0", sc.count(), turns, calls)
				}
				if len(msgs) != 2 || msgs[1].Role != RoleAssistant || msgs[1].Content != res.Content {
					t.Errorf("transcript = %+v", msgs)
				}
			})
		}
	}
}

func TestRunTools_MaxTurnsExceeded(t *testing.T) {
	for _, f := range toolFormats {
		t.Run(f.name, func(t *testing.T) {
			srv, sc := newToolServer(t, f.call([2]string{"c1", "a"}), f.call([2]string{"c2", "a"}), f.stop)
			var turns int
			res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(),
				map[string]ToolHandler{"a": echoHandler},
				ToolLoopOptions{MaxTurns: 2, OnTurn: func(int, *ChatResult) { turns++ }})
			if !errors.Is(err, ErrMaxToolTurns) {
				t.Fatalf("err = %v, want ErrMaxToolTurns", err)
			}
			if res == nil || len(res.ToolCalls) != 1 || !f.idOK(res.ToolCalls[0].ID, "c2") {
				t.Fatalf("last result = %+v", res)
			}
			if sc.count() != 2 || turns != 2 {
				t.Errorf("requests=%d turns=%d, want 2/2", sc.count(), turns)
			}
			// original + [assistant c1, tool c1]; the final tool-call assistant turn is dropped.
			if len(msgs) != 3 {
				t.Fatalf("transcript len = %d: %+v", len(msgs), msgs)
			}
			if last := msgs[len(msgs)-1]; last.Role != RoleTool || !f.idOK(last.ToolCallID, "c1") {
				t.Errorf("last message = %+v, want tool result c1 (no orphan assistant)", last)
			}
		})
	}
}

func TestRunTools_DefaultMaxTurnsIs8(t *testing.T) {
	f := toolFormats[0]
	var resp []string
	for i := 0; i < 12; i++ {
		resp = append(resp, f.call([2]string{fmt.Sprintf("c%d", i), "a"}))
	}
	srv, sc := newToolServer(t, resp...)
	_, _, err := RunTools(context.Background(), f.client(t, srv), f.req(),
		map[string]ToolHandler{"a": echoHandler}, ToolLoopOptions{})
	if !errors.Is(err, ErrMaxToolTurns) {
		t.Fatalf("err = %v, want ErrMaxToolTurns", err)
	}
	if sc.count() != 8 {
		t.Errorf("requests = %d, want 8", sc.count())
	}
}

func TestRunTools_ChatErrorReturnsTranscript(t *testing.T) {
	f := toolFormats[0]
	srv, _ := newToolServer(t, f.call([2]string{"c1", "a"})) // second request: 400
	res, msgs, err := RunTools(context.Background(), f.client(t, srv), f.req(),
		map[string]ToolHandler{"a": echoHandler}, ToolLoopOptions{})
	if err == nil {
		t.Fatal("want chat error")
	}
	if errors.Is(err, ErrMaxToolTurns) {
		t.Errorf("chat error must pass through, got %v", err)
	}
	if res == nil || len(res.ToolCalls) != 1 {
		t.Errorf("last successful result = %+v", res)
	}
	if len(msgs) != 3 {
		t.Errorf("transcript len = %d: %+v", len(msgs), msgs)
	}
}

func TestRunTools_NilClient(t *testing.T) {
	_, _, err := RunTools(context.Background(), nil, ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}, nil, ToolLoopOptions{})
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Fatalf("err = %T %v, want *ConfigError", err, err)
	}
}

func TestRunTools_ContextCancel(t *testing.T) {
	f := toolFormats[0]
	srv, sc := newToolServer(t, f.call([2]string{"c1", "a"}), f.stop)
	ctx, cancel := context.WithCancel(context.Background())
	h := func(context.Context, ToolCall) (string, error) { cancel(); return "x", nil }
	res, msgs, err := RunTools(ctx, f.client(t, srv), f.req(), map[string]ToolHandler{"a": h}, ToolLoopOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if res == nil || len(msgs) == 0 {
		t.Errorf("want last result and transcript, got %v / %d", res, len(msgs))
	}
	if sc.count() != 1 {
		t.Errorf("requests = %d, want 1 (no chat after cancel)", sc.count())
	}
}

func TestRunTools_ContextCancelledBeforeStart(t *testing.T) {
	f := toolFormats[0]
	srv, sc := newToolServer(t, f.stop)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := RunTools(ctx, f.client(t, srv), f.req(), nil, ToolLoopOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if sc.count() != 0 {
		t.Errorf("requests = %d, want 0", sc.count())
	}
}
