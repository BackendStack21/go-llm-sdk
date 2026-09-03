package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestBuildGeminiRequest_Golden(t *testing.T) {
	req := &ChatRequest{
		System: []SystemBlock{{Text: "Be terse."}},
		Messages: []Message{
			{Role: RoleUser, Content: "Hi"},
			{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{{Name: "f", Arguments: `{"a":1}`}}},
			{Role: RoleTool, ToolName: "f", ToolCallID: "x", Content: `{"answer":42}`},
			{Role: RoleTool, ToolName: "g", ToolCallID: "y", Content: "plain text"},
		},
		Tools:     []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Thinking:  "low",
		MaxTokens: 1000,
	}
	body, err := buildGeminiRequest(req, "gemini-2.5-pro", false)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("decode: %v\n%s", err, body)
	}

	// systemInstruction with the system text.
	si := m["systemInstruction"].(map[string]any)
	siParts := si["parts"].([]any)
	if siParts[0].(map[string]any)["text"] != "Be terse." {
		t.Errorf("systemInstruction = %v", si)
	}

	// contents: user, model(functionCall), merged user(functionResponse×2)
	contents := m["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %d, want 3", len(contents))
	}
	modelTurn := contents[1].(map[string]any)
	if modelTurn["role"] != "model" {
		t.Errorf("assistant role = %v, want model", modelTurn["role"])
	}
	fc := modelTurn["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
	if fc["name"] != "f" {
		t.Errorf("functionCall = %v", fc)
	}
	toolTurn := contents[2].(map[string]any)
	if toolTurn["role"] != "user" {
		t.Errorf("functionResponse role = %v, want user", toolTurn["role"])
	}
	frs := toolTurn["parts"].([]any)
	if len(frs) != 2 {
		t.Fatalf("functionResponse parts = %d, want 2 (merged consecutive tool messages)", len(frs))
	}
	fr0 := frs[0].(map[string]any)["functionResponse"].(map[string]any)
	if fr0["name"] != "f" {
		t.Errorf("fr0 name = %v", fr0["name"])
	}
	// JSON-object tool result passes through as an object.
	respObj := fr0["response"].(map[string]any)
	if respObj["answer"].(float64) != 42 {
		t.Errorf("fr0 response = %v, want pass-through object", fr0["response"])
	}
	// Plain-text tool result wraps under "result".
	fr1 := frs[1].(map[string]any)["functionResponse"].(map[string]any)
	respObj2 := fr1["response"].(map[string]any)
	if respObj2["result"] != "plain text" {
		t.Errorf("fr1 response = %v, want wrapped {result}", fr1["response"])
	}

	// generationConfig: thinkingConfig low + maxOutputTokens.
	gc := m["generationConfig"].(map[string]any)
	tc := gc["thinkingConfig"].(map[string]any)
	if tc["thinkingBudget"].(float64) != 1024 {
		t.Errorf("thinkingBudget = %v, want 1024 (low)", tc["thinkingBudget"])
	}
	if gc["maxOutputTokens"].(float64) != 1000 {
		t.Errorf("maxOutputTokens = %v", gc["maxOutputTokens"])
	}

	// tools: functionDeclarations
	tools := m["tools"].([]any)
	decls := tools[0].(map[string]any)["functionDeclarations"].([]any)
	if decls[0].(map[string]any)["name"] != "f" {
		t.Errorf("decl = %v", decls[0])
	}
}

func TestBuildGeminiRequest_ThinkingVariants(t *testing.T) {
	base := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "x"}}}
	cases := []struct {
		thinking string
		budget   int
		absent   bool
	}{
		{"disabled", 0, false},
		{"enabled", -1, false},
		{"high", 24576, false},
		{"", 0, true},
	}
	for _, c := range cases {
		r := *base
		r.Thinking = c.thinking
		body, _ := buildGeminiRequest(&r, "gemini-2.5-pro", false)
		var m map[string]any
		json.Unmarshal(body, &m)
		gcAny, ok := m["generationConfig"]
		if c.absent {
			if ok {
				gc := gcAny.(map[string]any)
				if _, has := gc["thinkingConfig"]; has {
					t.Errorf("thinking=%q: thinkingConfig present, must be omitted", c.thinking)
				}
			}
			continue
		}
		gc := gcAny.(map[string]any)
		tc := gc["thinkingConfig"].(map[string]any)
		if tc["thinkingBudget"].(float64) != float64(c.budget) {
			t.Errorf("thinking=%q budget = %v, want %d", c.thinking, tc["thinkingBudget"], c.budget)
		}
	}
}

func TestParseGeminiResponse(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {
				"role": "model",
				"parts": [
					{"text":"ponder","thought":true},
					{"text":"Answer"},
					{"functionCall":{"name":"f","args":{"a":1}}}
				]
			},
			"finishReason": "STOP"
		}],
		"usageMetadata": {"promptTokenCount":9,"candidatesTokenCount":4,"thoughtsTokenCount":6}
	}`)
	res, err := parseGeminiResponse(body)
	if err != nil {
		t.Fatal(err)
	}
	if res.Content != "Answer" || res.ReasoningContent != "ponder" {
		t.Errorf("content: %+v", res)
	}
	if res.FinishReason != FinishStop {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID == "" || res.ToolCalls[0].Name != "f" {
		t.Errorf("tool calls (synthesized id): %+v", res.ToolCalls)
	}
	if res.ToolCalls[0].Arguments != `{"a":1}` {
		t.Errorf("args = %q", res.ToolCalls[0].Arguments)
	}
	if res.Usage.ReasoningTokens != 6 {
		t.Errorf("usage = %+v, want thoughtsTokenCount→ReasoningTokens", res.Usage)
	}
}

func TestParseGeminiResponse_FinishReasonMapping(t *testing.T) {
	cases := map[string]string{
		"STOP":       FinishStop,
		"MAX_TOKENS": FinishLength,
		"SAFETY":     FinishContentFilter,
		"RECITATION": FinishContentFilter,
	}
	for in, want := range cases {
		body := fmt.Sprintf(`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":%q}]}`, in)
		res, err := parseGeminiResponse([]byte(body))
		if err != nil {
			t.Fatalf("%s: %v", in, err)
		}
		if res.FinishReason != want {
			t.Errorf("%s → %q, want %q", in, res.FinishReason, want)
		}
	}
}

func TestMapGeminiStreamEvent_Chunks(t *testing.T) {
	acc := newStreamAccum()
	chunks := []string{
		`{"candidates":[{"content":{"parts":[{"text":"think","thought":true}]}}],"usageMetadata":{"promptTokenCount":5,"thoughtsTokenCount":2}}`,
		`{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"thoughtsTokenCount":2}}`,
	}
	for _, c := range chunks {
		if _, _, err := mapGeminiStreamEvent([]byte(c), acc); err != nil {
			t.Fatalf("%s: %v", c, err)
		}
	}
	res := acc.result()
	if res.ReasoningContent != "think" || res.Content != "Hi" {
		t.Errorf("content: %+v", res)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "f" {
		t.Errorf("tool calls: %+v", res.ToolCalls)
	}
	if res.FinishReason != FinishStop {
		t.Errorf("finish = %q", res.FinishReason)
	}
	if res.Usage.CompletionTokens != 3 {
		t.Errorf("usage = %+v (last chunk wins)", res.Usage)
	}
}

func TestListModelsGemini_PaginationAndLimits(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("x-goog-api-key") != "k" {
			w.WriteHeader(400)
			return
		}
		if r.URL.Query().Get("pageToken") == "" {
			fmt.Fprint(w, `{"models":[{"name":"models/gemini-2.5-pro","displayName":"Gemini 2.5 Pro","inputTokenLimit":1048576,"outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","embedContent"]}],"nextPageToken":"p2"}`)
			return
		}
		fmt.Fprint(w, `{"models":[{"name":"models/gemini-2.5-flash","inputTokenLimit":1048576,"outputTokenLimit":65536}]}`)
	}))
	defer srv.Close()

	pc := newProviderClient(ProviderConfig{ID: "gemini", Format: FormatGemini, BaseURL: srv.URL, APIKey: "k"},
		srv.Client(), srv.Client())
	models, err := listModelsGemini(context.Background(), pc)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2 (paged)", calls)
	}
	if len(models) != 2 {
		t.Fatalf("models = %d", len(models))
	}
	m0 := models[0]
	if m0.ID != "gemini-2.5-pro" {
		t.Errorf("ID = %q (models/ prefix must be trimmed)", m0.ID)
	}
	if m0.ContextWindow != 1048576 || m0.MaxOutputTokens != 65536 {
		t.Errorf("limits = %+v", m0)
	}
	if len(m0.Capabilities) != 2 {
		t.Errorf("capabilities = %v", m0.Capabilities)
	}
}
