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
	"sync/atomic"
	"testing"
)

// A cancelled single-attempt call reports the context error, never
// "retry exhausted".
func TestWithRetry_LastAttemptContextError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	rt := rtFunc(func(r *http.Request) (*http.Response, error) {
		cancel()
		return nil, r.Context().Err()
	})
	sdk := New(WithTransport(rt), WithRetryPolicy(RetryPolicy{MaxAttempts: 1}),
		WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://x")))
	c, _ := sdk.Chat("openai", "m")
	_, err := c.Call(ctx, userMsg())
	if !errors.Is(err, context.Canceled) || strings.Contains(err.Error(), "retry exhausted") {
		t.Errorf("err = %v", err)
	}
}

// Anthropic-named and OpenAI-named cache fields describing the same read
// are never summed; Moonshot's top-level cached_tokens is understood.
func TestUsageFromOpenAI_GatewayShapes(t *testing.T) {
	u := usageFromOpenAI(&oaRespUsage{PromptTokens: 1000, CacheReadTokens: 600, PromptTokensDetails: &oaPromptDetails{CachedTokens: 600}})
	if u.InputTokens() != 1000 || u.CacheReadTokens != 600 || u.PromptTokens != 400 {
		t.Errorf("gateway both-shapes usage = %+v (input %d)", u, u.InputTokens())
	}
	u = usageFromOpenAI(&oaRespUsage{PromptTokens: 1000, TopCachedTokens: 300})
	if u.PromptTokens != 700 || u.CacheReadTokens != 300 || !u.CacheReported {
		t.Errorf("moonshot cached_tokens usage = %+v", u)
	}
}

// DeepSeek only accepts json_object: json_schema fails fast instead of a 400.
func TestResponseFormat_JSONSchemaQuirk(t *testing.T) {
	var deepseek ProviderConfig
	for _, c := range builtinProviders() {
		if c.ID == "deepseek" {
			deepseek = c
		}
	}
	if !deepseek.Quirks.NoJSONSchema {
		t.Fatal("deepseek must carry Quirks.NoJSONSchema")
	}
	deepseek.BaseURL = "https://x"
	pc := newProviderClient(deepseek, nil, nil)
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: testSchema}}
	var ce *ConfigError
	if _, _, err := pc.buildChatRequest(req, "deepseek-chat", false); !errors.As(err, &ce) {
		t.Errorf("json_schema on deepseek: err = %v", err)
	}
	req.ResponseFormat = &ResponseFormat{Type: ResponseJSONObject}
	if _, _, err := pc.buildChatRequest(req, "deepseek-chat", false); err != nil {
		t.Errorf("json_object on deepseek: %v", err)
	}
}

func TestAnthropic_ControlEdges(t *testing.T) {
	no := false
	// disable_parallel_tool_use never rides a "none" choice.
	req := toolReq(&ToolChoice{Mode: ToolChoiceNone})
	req.ParallelToolCalls = &no
	m, _ := buildBody(t, FormatAnthropic, "claude", req)
	jsonEq(t, "none choice", m["tool_choice"], `{"type":"none"}`)

	var ce *ConfigError
	// JSON mode forces its own tool, so user tools could never be called.
	withTools := toolReq(nil)
	withTools.ResponseFormat = &ResponseFormat{Type: ResponseJSONObject}
	if _, err := buildAnthropicRequest(withTools, "claude", false); !errors.As(err, &ce) {
		t.Errorf("JSON mode + tools: err = %v", err)
	}
	// Anthropic input_schema must be an object schema.
	arr := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Type: ResponseJSONSchema, Schema: json.RawMessage(`{"type":"array"}`)}}
	if _, err := buildAnthropicRequest(arr, "claude", false); !errors.As(err, &ce) {
		t.Errorf("non-object schema: err = %v", err)
	}
}

func TestGemini_SeedRange(t *testing.T) {
	big := 1 << 40
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, Seed: &big}
	var ce *ConfigError
	if _, err := buildGeminiRequest(req, "g", false); !errors.As(err, &ce) {
		t.Errorf("out-of-range seed: err = %v", err)
	}
}

// An image-only tool result keeps a non-empty tool message on chat
// completions.
func TestOpenAI_ImageOnlyToolResultPlaceholder(t *testing.T) {
	req := toolResultReq(Message{Role: RoleTool, ToolCallID: "c1", Parts: []ContentPart{ImagePart("image/png", []byte{1})}})
	m, _ := buildBody(t, FormatOpenAI, "gpt-4o", req)
	if c := m["messages"].([]any)[2].(map[string]any)["content"]; c == "" {
		t.Errorf("image-only tool content must not be empty: %v", c)
	}
}

func TestEmbed_GeminiModelPrefixAndBatching(t *testing.T) {
	var calls atomic.Int32
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		paths = append(paths, r.URL.EscapedPath())
		b, _ := io.ReadAll(r.Body)
		var req gmEmbedRequest
		_ = json.Unmarshal(b, &req)
		var sb strings.Builder
		sb.WriteString(`{"embeddings":[`)
		for i, it := range req.Requests {
			if i > 0 {
				sb.WriteString(",")
			}
			if it.Model != "models/text-embedding-004" {
				t.Errorf("body model = %q", it.Model)
			}
			fmt.Fprintf(&sb, `{"values":[%d]}`, len(it.Content.Parts[0].Text))
		}
		sb.WriteString(`]}`)
		_, _ = fmt.Fprint(w, sb.String())
	}))
	t.Cleanup(srv.Close)
	inputs := make([]string, 250)
	for i := range inputs {
		inputs[i] = strings.Repeat("a", i+1)
	}
	sdk := New(WithProvider("gemini", WithAPIKey("k"), WithBaseURL(srv.URL)))
	res, err := sdk.Embed(context.Background(), "gemini", "models/text-embedding-004", EmbedRequest{Inputs: inputs})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 3 {
		t.Errorf("batches = %d, want 3 (100 per call)", calls.Load())
	}
	if paths[0] != "/v1beta/models/text-embedding-004:batchEmbedContents" {
		t.Errorf("path = %s", paths[0])
	}
	for i, v := range res.Embeddings {
		if int(v[0]) != i+1 {
			t.Fatalf("vector %d out of order: %v", i, v)
		}
	}
}

// Gateways that omit index (all zero) are trusted in response order.
func TestEmbed_OpenAIMissingIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{"data":[{"embedding":[1]},{"embedding":[2]}]}`)
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	res, err := sdk.Embed(context.Background(), "openai", "m", EmbedRequest{Inputs: []string{"a", "b"}})
	if err != nil || res.Embeddings[1][0] != 2 {
		t.Fatalf("res=%+v err=%v", res, err)
	}
}

// Credentials never follow a redirect to another host.
func TestNoCrossHostRedirect(t *testing.T) {
	var leaked atomic.Int32
	evil := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "" || r.Header.Get("api-key") != "" {
			leaked.Add(1)
		}
		_, _ = fmt.Fprint(w, `{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn"}`)
	}))
	t.Cleanup(evil.Close)
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, strings.Replace(evil.URL, "127.0.0.1", "localhost", 1)+r.URL.Path, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(origin.Close)
	fastBackoff(t)
	sdk := New(WithRetryPolicy(RetryPolicy{MaxAttempts: 1}),
		WithProvider("anthropic", WithAPIKey("secret"), WithBaseURL(origin.URL), WithHeaders(map[string]string{"api-key": "secret"})))
	c, _ := sdk.Chat("anthropic", "claude")
	_, _ = c.Call(context.Background(), userMsg())
	if leaked.Load() != 0 {
		t.Error("credentials were forwarded to a redirect target on another host")
	}
}

// fmt verbs on a Provider or config never print the key or header values.
func TestProviderFormattingRedactsSecrets(t *testing.T) {
	sdk := New(WithProvider("openai", WithAPIKey("SECRETKEY123"), WithHeaders(map[string]string{"X-Gw": "HDRSECRET456"})))
	p, _ := sdk.Provider("openai")
	for _, s := range []string{fmt.Sprintf("%v", p), fmt.Sprintf("%+v", p), fmt.Sprintf("%#v", p), fmt.Sprintf("%#v", p.Config()), fmt.Sprint(p.Config())} {
		if strings.Contains(s, "SECRETKEY123") || strings.Contains(s, "HDRSECRET456") {
			t.Errorf("secret leaked: %s", s)
		}
	}
}

// Extra cannot replace the conversation or tool catalog the SDK validated.
func TestChatRequestExtra_StructuralKeysReserved(t *testing.T) {
	for _, k := range []string{"messages", "contents", "input", "system", "systemInstruction", "instructions", "tools", "model"} {
		r := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}, Extra: map[string]any{k: []any{}}}
		pc := newProviderClient(ProviderConfig{ID: "x", Format: FormatOpenAI, BaseURL: "https://x"}, nil, nil)
		var ce *ConfigError
		if _, _, err := pc.buildChatRequest(r, "m", false); !errors.As(err, &ce) {
			t.Errorf("Extra[%q]: err = %v, want ConfigError", k, err)
		}
	}
}

// A control character in the upload filename can never inject a
// multipart header.
func TestTranscribe_FilenameControlChars(t *testing.T) {
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1")))
	_, err := sdk.Transcribe(context.Background(), "openai", "m", TranscribeRequest{Audio: []byte("a"), Filename: "a.wav\r\nX-Injected: 1"})
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Errorf("err = %v, want ConfigError", err)
	}
}
