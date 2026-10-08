package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ── Responses API: every reasoning item, stream error event ──────────────

func TestParseResponsesAPI_AllReasoningItems(t *testing.T) {
	res, err := parseResponsesAPI([]byte(`{"status":"completed","output":[
		{"type":"reasoning","encrypted_content":"E1","summary":[{"type":"summary_text","text":"s1"}]},
		{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},
		{"type":"reasoning","encrypted_content":"E2","summary":[{"type":"summary_text","text":"s2a"},{"type":"summary_text","text":"s2b"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	want := []ThinkingBlock{{Text: "s1", Signature: "E1"}, {Text: "s2a\ns2b", Signature: "E2"}}
	if len(res.ThinkingBlocks) != 2 || res.ThinkingBlocks[0] != want[0] || res.ThinkingBlocks[1] != want[1] {
		t.Fatalf("blocks = %+v", res.ThinkingBlocks)
	}
	if res.ThinkingSignature != "E2" {
		t.Errorf("legacy signature = %q", res.ThinkingSignature)
	}
}

func TestMapResponsesStreamEvent_ReasoningBlocksAndError(t *testing.T) {
	acc := newStreamAccum()
	for _, e := range []string{
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","encrypted_content":"E1","summary":[{"type":"summary_text","text":"s1"}]}}`,
		`{"type":"response.output_item.done","output_index":2,"item":{"type":"reasoning","encrypted_content":"E2","summary":[]}}`,
		`{"type":"response.completed","response":{"status":"completed","output":[]}}`,
	} {
		if _, _, err := mapResponsesStreamEvent([]byte(e), acc); err != nil {
			t.Fatal(err)
		}
	}
	res := acc.result()
	if len(res.ThinkingBlocks) != 2 || res.ThinkingBlocks[0].Signature != "E1" || res.ThinkingBlocks[0].Text != "s1" || res.ThinkingBlocks[1].Signature != "E2" {
		t.Fatalf("stream blocks = %+v", res.ThinkingBlocks)
	}
	// Fallback: blocks only in the completed response.
	acc = newStreamAccum()
	_, _, _ = mapResponsesStreamEvent([]byte(`{"type":"response.completed","response":{"status":"completed","output":[{"type":"reasoning","encrypted_content":"E9","summary":[]}]}}`), acc)
	if r := acc.result(); len(r.ThinkingBlocks) != 1 || r.ThinkingBlocks[0].Signature != "E9" || r.ThinkingSignature != "E9" {
		t.Errorf("fallback blocks = %+v", r)
	}
	// A top-level error event fails the stream with the provider message.
	_, done, err := mapResponsesStreamEvent([]byte(`{"type":"error","code":"server_error","message":"upstream exploded"}`), newStreamAccum())
	if err == nil || !done || !strings.Contains(err.Error(), "upstream exploded") {
		t.Errorf("error event: done=%v err=%v", done, err)
	}
}

func TestBuildResponsesInput_ReplaysThinkingBlocks(t *testing.T) {
	_, input := buildResponsesInput(&ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ThinkingSignature: "legacy-ignored", ThinkingBlocks: []ThinkingBlock{
			{Text: "s1", Signature: "E1"}, {Text: "unsigned"}, {Signature: "E2"},
		}, ToolCalls: []ToolCall{{ID: "c", Name: "f", Arguments: "{}"}}},
	}})
	b, _ := json.Marshal(input)
	var items []map[string]any
	_ = json.Unmarshal(b, &items)
	var encs []string
	for _, it := range items {
		if it["type"] == "reasoning" {
			encs = append(encs, it["encrypted_content"].(string))
		}
	}
	if strings.Join(encs, ",") != "E1,E2" {
		t.Errorf("replayed reasoning items = %v (unsigned skipped, legacy ignored)", encs)
	}
}

// ── tool results: IsError and content parts ──────────────────────────────

func toolResultReq(m Message) *ChatRequest {
	return &ChatRequest{Messages: []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "shot", Arguments: "{}"}}},
		m,
	}}
}

func TestToolResult_IsError(t *testing.T) {
	req := toolResultReq(Message{Role: RoleTool, ToolCallID: "c1", Content: "disk full", IsError: true})
	m, _ := buildBody(t, FormatAnthropic, "claude", req)
	tr := m["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if tr["is_error"] != true || tr["content"] != "disk full" {
		t.Errorf("anthropic tool_result = %v", tr)
	}
	m, _ = buildBody(t, FormatGemini, "gemini", req)
	fr := m["contents"].([]any)[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	jsonEq(t, "gemini error response", fr["response"], `{"error":"disk full"}`)
	// IsError outside the tool role is a config error.
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "https://x"}, nil, nil)
	_, _, err := pc.buildChatRequest(&ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x", IsError: true}}}, "m", false)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Errorf("IsError on user: err = %v", err)
	}
}

func TestToolResult_ContentParts(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G'}
	req := toolResultReq(Message{Role: RoleTool, ToolCallID: "c1", Parts: []ContentPart{TextPart("see image"), ImagePart("image/png", png)}})

	// Anthropic: tool_result content blocks.
	m, _ := buildBody(t, FormatAnthropic, "claude", req)
	tr := m["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	blocks := tr["content"].([]any)
	if len(blocks) != 2 || blocks[0].(map[string]any)["text"] != "see image" || blocks[1].(map[string]any)["type"] != "image" {
		t.Errorf("anthropic tool_result content = %v", tr)
	}

	// OpenAI chat: text stays on the tool message; images follow in a user
	// message (chat completions tool messages cannot carry images).
	m, _ = buildBody(t, FormatOpenAI, "gpt-4o", req)
	msgs := m["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("openai messages = %v", msgs)
	}
	tool := msgs[2].(map[string]any)
	if tool["role"] != "tool" || tool["content"] != "see image" {
		t.Errorf("openai tool message = %v", tool)
	}
	user := msgs[3].(map[string]any)
	parts := user["content"].([]any)
	if user["role"] != "user" || parts[len(parts)-1].(map[string]any)["type"] != "image_url" {
		t.Errorf("openai image follow-up = %v", user)
	}

	// Responses: function_call_output carries input_text + input_image.
	rs := *req
	rs.Tools = []ToolDef{{Name: "shot"}}
	m, _ = buildBody(t, FormatOpenAI, "gpt-5.6", &rs)
	var out map[string]any
	for _, it := range m["input"].([]any) {
		if it.(map[string]any)["type"] == "function_call_output" {
			out = it.(map[string]any)
		}
	}
	op := out["output"].([]any)
	if len(op) != 2 || op[0].(map[string]any)["type"] != "input_text" || op[1].(map[string]any)["type"] != "input_image" {
		t.Errorf("responses output = %v", out)
	}

	// Gemini: functionResponse text + inlineData in the same user turn.
	m, _ = buildBody(t, FormatGemini, "gemini", req)
	gparts := m["contents"].([]any)[2].(map[string]any)["parts"].([]any)
	fr := gparts[0].(map[string]any)["functionResponse"].(map[string]any)
	jsonEq(t, "gemini tool text", fr["response"], `{"result":"see image"}`)
	if len(gparts) != 2 || gparts[1].(map[string]any)["inlineData"] == nil {
		t.Errorf("gemini parts = %v", gparts)
	}

	// Parts stay forbidden on assistant/system turns.
	pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "https://x"}, nil, nil)
	_, _, err := pc.buildChatRequest(&ChatRequest{Messages: []Message{{Role: RoleAssistant, Parts: []ContentPart{TextPart("x")}}}}, "m", false)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Errorf("assistant parts: err = %v", err)
	}
}

// ── unauthenticated custom provider message ──────────────────────────────

func TestUnauthenticatedMessageNamesRealEnvKeys(t *testing.T) {
	sdk := New(WithProvider("gw", WithFormat(FormatOpenAI), WithBaseURL("https://x")))
	_, err := sdk.Chat("gw", "m")
	if err == nil || strings.Contains(err.Error(), "GW_API_KEY") || !strings.Contains(err.Error(), "WithEnvKeys") {
		t.Errorf("custom provider without EnvKeys: %v", err)
	}
	sdk = New(WithProvider("gw", WithFormat(FormatOpenAI), WithBaseURL("https://x"), WithEnvKeys("GW_TOKEN")))
	if _, err := sdk.Chat("gw", "m"); err == nil || !strings.Contains(err.Error(), "GW_TOKEN") {
		t.Errorf("custom provider with EnvKeys: %v", err)
	}
	sdk = New()
	if _, err := sdk.Chat("openai", "m"); err == nil || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Errorf("built-in: %v", err)
	}
	for _, fn := range []func() error{
		func() error {
			_, err := sdk.Speak(context.Background(), "openai", "m", SpeakRequest{Text: "x", Voice: "v"})
			return err
		},
		func() error {
			_, err := sdk.Transcribe(context.Background(), "openai", "m", TranscribeRequest{Audio: []byte("a")})
			return err
		},
		func() error {
			_, err := sdk.Embed(context.Background(), "openai", "m", EmbedRequest{Inputs: []string{"a"}})
			return err
		},
	} {
		if err := fn(); err == nil || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
			t.Errorf("entry point message: %v", err)
		}
	}
}

// ── embeddings ───────────────────────────────────────────────────────────

func TestEmbed_OpenAI(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" || r.Header.Get("Authorization") != "Bearer k" {
			t.Errorf("path=%s auth=%q", r.URL.Path, r.Header.Get("Authorization"))
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		// Out of order on purpose: the SDK sorts by index.
		_, _ = fmt.Fprint(w, `{"data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}],"model":"emb-x","usage":{"prompt_tokens":5}}`)
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	res, err := sdk.Embed(context.Background(), "openai", "emb", EmbedRequest{Inputs: []string{"a", "b"}, Dimensions: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Embeddings) != 2 || res.Embeddings[0][0] != 0.1 || res.Embeddings[1][1] != 0.4 {
		t.Errorf("embeddings = %v", res.Embeddings)
	}
	if res.Model != "emb" || res.Usage.PromptTokens != 5 {
		t.Errorf("result = %+v", res)
	}
	if got["model"] != "emb" || got["dimensions"] != float64(2) || got["encoding_format"] != "float" {
		t.Errorf("request = %v", got)
	}
}

func TestEmbed_Gemini(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models/text-embedding-004:batchEmbedContents" || r.Header.Get("x-goog-api-key") != "k" {
			t.Errorf("path=%s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		_, _ = fmt.Fprint(w, `{"embeddings":[{"values":[1,2]},{"values":[3,4]}]}`)
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("gemini", WithAPIKey("k"), WithBaseURL(srv.URL)))
	res, err := sdk.Embed(context.Background(), "gemini", "text-embedding-004", EmbedRequest{Inputs: []string{"a", "b"}, Dimensions: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Embeddings) != 2 || res.Embeddings[1][0] != 3 {
		t.Errorf("embeddings = %v", res.Embeddings)
	}
	r0 := got["requests"].([]any)[0].(map[string]any)
	if r0["model"] != "models/text-embedding-004" || r0["outputDimensionality"] != float64(2) {
		t.Errorf("request = %v", r0)
	}
}

func TestEmbed_ErrorsAndValidation(t *testing.T) {
	ctx := context.Background()
	sdk := New(WithProvider("anthropic", WithAPIKey("k")), WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1")))
	var ce *ConfigError
	for name, fn := range map[string]func() error{
		"no inputs": func() error { _, err := sdk.Embed(ctx, "openai", "m", EmbedRequest{}); return err },
		"empty input": func() error {
			_, err := sdk.Embed(ctx, "openai", "m", EmbedRequest{Inputs: []string{"a", " "}})
			return err
		},
		"no model": func() error { _, err := sdk.Embed(ctx, "openai", "", EmbedRequest{Inputs: []string{"a"}}); return err },
		"neg dims": func() error {
			_, err := sdk.Embed(ctx, "openai", "m", EmbedRequest{Inputs: []string{"a"}, Dimensions: -1})
			return err
		},
		"anthropic": func() error {
			_, err := sdk.Embed(ctx, "anthropic", "m", EmbedRequest{Inputs: []string{"a"}})
			return err
		},
		"unknown": func() error { _, err := sdk.Embed(ctx, "nope", "m", EmbedRequest{Inputs: []string{"a"}}); return err },
	} {
		if err := fn(); !errors.As(err, &ce) {
			t.Errorf("%s: err = %v, want ConfigError", name, err)
		}
	}

	// A vector-count mismatch or malformed body is a typed protocol error.
	for _, body := range []string{`{"data":[{"index":0,"embedding":[1]}]}`, `{not json`, `{"data":[{"index":5,"embedding":[1]},{"index":0,"embedding":[1]}]}`} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = fmt.Fprint(w, body) }))
		s2 := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
		_, err := s2.Embed(ctx, "openai", "m", EmbedRequest{Inputs: []string{"a", "b"}})
		srv.Close()
		var ae *APIError
		if !errors.As(err, &ae) || ae.Status != http.StatusOK {
			t.Errorf("body %s: err = %v, want APIError at 200", body, err)
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = fmt.Fprint(w, `{"embeddings":[]}`) }))
	t.Cleanup(srv.Close)
	s3 := New(WithProvider("gemini", WithAPIKey("k"), WithBaseURL(srv.URL)))
	if _, err := s3.Embed(ctx, "gemini", "m", EmbedRequest{Inputs: []string{"a"}}); err == nil {
		t.Error("gemini count mismatch must error")
	}

	// Provider errors keep the shared retry taxonomy.
	fastBackoff(t)
	errSrv, n := countingServer(t, http.StatusBadRequest, nil)
	s4 := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(errSrv.URL)))
	var ae *APIError
	if _, err := s4.Embed(ctx, "openai", "m", EmbedRequest{Inputs: []string{"a"}}); !errors.As(err, &ae) || ae.Status != 400 || n.Load() != 1 {
		t.Errorf("400: err = %v, attempts = %d", err, n.Load())
	}
}
