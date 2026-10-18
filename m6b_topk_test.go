package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"
)

// m6b SL04: TopK, Gemini generationConfig deep-merge, embedding TaskType (AC-5).

const (
	tkOpenAIOK = `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	tkRespOK   = `{"id":"r1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}`
	tkAnthOK   = `{"id":"m1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	tkGemOK    = `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`
)

func tkClient(t *testing.T, id string, f Format, model, body string) (*ChatClient, *countRec, *httptest.Server) {
	t.Helper()
	srv, rec := countServer(t, obsStep{status: 200, body: body})
	cc := newTestClient(t, ProviderConfig{ID: id, Format: f, BaseURL: srv.URL, APIKey: "k"}, srv)
	cc.model = model
	return cc, rec, srv
}

func tkReq() *ChatRequest {
	return &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}, MaxTokens: 100}
}

func tkSent(t *testing.T, rec *countRec) map[string]any {
	t.Helper()
	if rec.n() != 1 {
		t.Fatalf("requests = %d, want 1", rec.n())
	}
	return rec.bodies[0]
}

func tkGenCfg(t *testing.T, b map[string]any) map[string]any {
	t.Helper()
	g, ok := b["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing or not an object: %v", b)
	}
	return g
}

func TestTopK_PerFormatWire(t *testing.T) {
	t.Run("anthropic", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "anthropic", FormatAnthropic, "claude-x", tkAnthOK)
		req := tkReq()
		req.TopK = 40
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if got := tkSent(t, rec)["top_k"]; got != float64(40) {
			t.Errorf("top_k = %v, want 40", got)
		}
	})
	t.Run("anthropic thinking omits", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "anthropic", FormatAnthropic, "claude-x", tkAnthOK)
		req := tkReq()
		req.TopK = 40
		req.Thinking = "enabled"
		req.ThinkingBudget = 2048
		req.MaxTokens = 8192
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		b := tkSent(t, rec)
		if _, ok := b["thinking"]; !ok {
			t.Fatalf("thinking not enabled in body: %v", b)
		}
		if v, ok := b["top_k"]; ok {
			t.Errorf("top_k = %v with thinking on, want absent", v)
		}
	})
	t.Run("gemini", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		req := tkReq()
		req.TopK = 40
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if got := tkGenCfg(t, tkSent(t, rec))["topK"]; got != float64(40) {
			t.Errorf("generationConfig.topK = %v, want 40", got)
		}
	})
	t.Run("gemini without TopK emits none", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		if _, err := cc.Call(context.Background(), tkReq()); err != nil {
			t.Fatal(err)
		}
		if _, ok := tkGenCfg(t, tkSent(t, rec))["topK"]; ok {
			t.Error("topK present without TopK")
		}
	})
	t.Run("openai chat ignores", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "openai", FormatOpenAI, "gpt-4o", tkOpenAIOK)
		req := tkReq()
		req.TopK = 40
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		tkNoTopK(t, tkSent(t, rec))
	})
	t.Run("responses ignores", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "openai", FormatOpenAI, "gpt-5.6", tkRespOK)
		req := tkReq()
		req.TopK = 40
		req.Tools = []ToolDef{{Name: "f", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if rec.paths[0] != "/responses" && rec.paths[0] != "/v1/responses" {
			t.Fatalf("path = %q, want a responses endpoint", rec.paths[0])
		}
		tkNoTopK(t, tkSent(t, rec))
	})
}

func tkNoTopK(t *testing.T, b map[string]any) {
	t.Helper()
	for _, k := range []string{"top_k", "topK"} {
		if v, ok := b[k]; ok {
			t.Errorf("body has %s = %v", k, v)
		}
	}
}

func TestTopK_NegativeIsConfigError(t *testing.T) {
	for _, f := range []struct {
		id     string
		format Format
		model  string
		body   string
	}{
		{"openai", FormatOpenAI, "gpt-4o", tkOpenAIOK},
		{"anthropic", FormatAnthropic, "claude-x", tkAnthOK},
		{"gemini", FormatGemini, "gemini-x", tkGemOK},
	} {
		t.Run(f.id, func(t *testing.T) {
			cc, rec, _ := tkClient(t, f.id, f.format, f.model, f.body)
			req := tkReq()
			req.TopK = -1
			_, err := cc.Call(context.Background(), req)
			var ce *ConfigError
			if !errors.As(err, &ce) {
				t.Fatalf("err = %v, want *ConfigError", err)
			}
			if rec.n() != 0 {
				t.Errorf("requests = %d, want 0", rec.n())
			}
		})
	}
}

func TestGeminiExtraGenerationConfigDeepMerge(t *testing.T) {
	t.Run("extra keys merged, sdk keys kept", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		req := tkReq()
		req.Temperature = 0.5
		req.Extra = map[string]any{"generationConfig": map[string]any{"topK": 5, "candidateCount": 2}}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		g := tkGenCfg(t, tkSent(t, rec))
		want := map[string]any{"maxOutputTokens": float64(100), "temperature": 0.5, "topK": float64(5), "candidateCount": float64(2)}
		for k, v := range want {
			if g[k] != v {
				t.Errorf("generationConfig[%s] = %v, want %v (got %v)", k, g[k], v, g)
			}
		}
	})
	t.Run("extra overrides sdk key", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		req := tkReq()
		req.TopK = 40
		req.Extra = map[string]any{"generationConfig": map[string]any{"maxOutputTokens": 7, "topK": 5}}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		g := tkGenCfg(t, tkSent(t, rec))
		if g["maxOutputTokens"] != float64(7) || g["topK"] != float64(5) {
			t.Errorf("generationConfig = %v, want Extra to win", g)
		}
	})
	t.Run("no sdk generationConfig uses extra as is", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}},
			Extra: map[string]any{"generationConfig": map[string]any{"candidateCount": 3}}}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		g := tkGenCfg(t, tkSent(t, rec))
		if len(g) != 1 || g["candidateCount"] != float64(3) {
			t.Errorf("generationConfig = %v", g)
		}
	})
	t.Run("nested objects replaced whole", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		req := tkReq()
		req.Thinking = "high"
		req.Extra = map[string]any{"generationConfig": map[string]any{"thinkingConfig": map[string]any{"includeThoughts": false}}}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		g := tkGenCfg(t, tkSent(t, rec))
		tc, _ := g["thinkingConfig"].(map[string]any)
		if len(tc) != 1 || tc["includeThoughts"] != false {
			t.Errorf("thinkingConfig = %v, want exactly the Extra object", tc)
		}
		if g["maxOutputTokens"] != float64(100) {
			t.Errorf("maxOutputTokens dropped: %v", g)
		}
	})
	t.Run("other extra keys stay shallow", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		req := tkReq()
		req.Extra = map[string]any{"safetySettings": []any{map[string]any{"category": "X"}}}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		b := tkSent(t, rec)
		if _, ok := b["safetySettings"]; !ok {
			t.Errorf("safetySettings missing: %v", b)
		}
		if tkGenCfg(t, b)["maxOutputTokens"] != float64(100) {
			t.Errorf("generationConfig changed: %v", b["generationConfig"])
		}
	})
	t.Run("non-object extra replaces", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "gemini", FormatGemini, "gemini-x", tkGemOK)
		req := tkReq()
		req.Extra = map[string]any{"generationConfig": "raw"}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if got := tkSent(t, rec)["generationConfig"]; got != "raw" {
			t.Errorf("generationConfig = %v, want raw replacement", got)
		}
	})
	t.Run("anthropic stays shallow", func(t *testing.T) {
		cc, rec, _ := tkClient(t, "anthropic", FormatAnthropic, "claude-x", tkAnthOK)
		req := tkReq()
		req.Extra = map[string]any{"metadata": map[string]any{"user_id": "u"}}
		if _, err := cc.Call(context.Background(), req); err != nil {
			t.Fatal(err)
		}
		if m, _ := tkSent(t, rec)["metadata"].(map[string]any); m["user_id"] != "u" {
			t.Errorf("metadata = %v", m)
		}
	})
}

func TestEmbedTaskType(t *testing.T) {
	embedOpenAI := `{"data":[{"index":0,"embedding":[0.5,1]},{"index":1,"embedding":[1,2]}],"usage":{"prompt_tokens":2}}`
	embedGemini := `{"embeddings":[{"values":[0.5,1]},{"values":[1,2]}]}`
	run := func(t *testing.T, f Format, id, body, taskType string) map[string]any {
		t.Helper()
		srv, rec := countServer(t, obsStep{status: 200, body: body})
		s := newTestSDK(t, ProviderConfig{ID: id, Format: f, APIKey: "k"}, srv)
		if _, err := s.Embed(context.Background(), id, "m", EmbedRequest{Inputs: []string{"a", "b"}, TaskType: taskType}); err != nil {
			t.Fatal(err)
		}
		return tkSent(t, rec)
	}
	t.Run("gemini every item", func(t *testing.T) {
		b := run(t, FormatGemini, "gemini", embedGemini, "  RETRIEVAL_QUERY ")
		reqs, _ := b["requests"].([]any)
		if len(reqs) != 2 {
			t.Fatalf("requests = %v", b)
		}
		for i, r := range reqs {
			if got := r.(map[string]any)["taskType"]; got != "RETRIEVAL_QUERY" {
				t.Errorf("item %d taskType = %v", i, got)
			}
		}
	})
	t.Run("gemini unset omits", func(t *testing.T) {
		b := run(t, FormatGemini, "gemini", embedGemini, "")
		for i, r := range b["requests"].([]any) {
			if _, ok := r.(map[string]any)["taskType"]; ok {
				t.Errorf("item %d has taskType", i)
			}
		}
	})
	t.Run("openai ignores", func(t *testing.T) {
		b := run(t, FormatOpenAI, "openai", embedOpenAI, "RETRIEVAL_QUERY")
		for _, k := range []string{"task_type", "taskType"} {
			if _, ok := b[k]; ok {
				t.Errorf("body has %s: %v", k, b)
			}
		}
	})
}
