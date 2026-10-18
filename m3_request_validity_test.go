package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

// ── Bug 1: image/jpg alias accepted at the boundary ─────────────────────

func TestM3JPGAliasAcceptedEndToEnd(t *testing.T) {
	png := []byte{1, 2, 3}
	for _, format := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
		t.Run(string(format)+"/user", func(t *testing.T) {
			req := &ChatRequest{Messages: []Message{{Role: RoleUser, Parts: []ContentPart{ImagePart("image/jpg", png)}}}}
			rec, err := m3Chat(t, format, req)
			if err != nil {
				t.Fatalf("image/jpg must be accepted (README): %v", err)
			}
			_, _, _, body := rec.snapshot()
			if strings.Contains(string(body), "image/jpg") || !strings.Contains(string(body), "image/jpeg") {
				t.Fatalf("wire must carry image/jpeg only: %s", body)
			}
		})
	}
	// Tool-result images ride the same validation + normalization.
	for _, format := range []Format{FormatAnthropic, FormatGemini} {
		t.Run(string(format)+"/tool", func(t *testing.T) {
			req := &ChatRequest{Messages: []Message{
				{Role: RoleUser, Content: "look"},
				{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "shot", Arguments: "{}"}}},
				{Role: RoleTool, ToolCallID: "c1", ToolName: "shot", Parts: []ContentPart{ImagePart("image/jpg", png)}},
			}}
			rec, err := m3Chat(t, format, req)
			if err != nil {
				t.Fatalf("image/jpg in a tool result must be accepted: %v", err)
			}
			_, _, _, body := rec.snapshot()
			if strings.Contains(string(body), "image/jpg") || !strings.Contains(string(body), "image/jpeg") {
				t.Fatalf("wire must carry image/jpeg only: %s", body)
			}
		})
	}
}

// ── Bug 2: empty text never yields an empty/invalid block ───────────────
// Policy: an empty text block is skipped when the message/system still has
// other content; a user message left with no content at all is a ConfigError
// before any network I/O. Empty system text simply contributes nothing.

// m3Blocks returns the content blocks (anthropic) or parts (gemini) of every
// message plus the system blocks/parts.
func m3AllBlocks(t *testing.T, format Format, body []byte) [][]any {
	t.Helper()
	m := m3JSON(t, body)
	var groups [][]any
	if format == FormatAnthropic {
		if s, ok := m["system"].([]any); ok {
			groups = append(groups, s)
		}
		for _, msg := range m["messages"].([]any) {
			groups = append(groups, msg.(map[string]any)["content"].([]any))
		}
		return groups
	}
	if si, ok := m["systemInstruction"].(map[string]any); ok {
		groups = append(groups, si["parts"].([]any))
	}
	for _, c := range m["contents"].([]any) {
		groups = append(groups, c.(map[string]any)["parts"].([]any))
	}
	return groups
}

func m3AssertValidBlocks(t *testing.T, format Format, body []byte) {
	t.Helper()
	for _, g := range m3AllBlocks(t, format, body) {
		if len(g) == 0 {
			t.Errorf("empty content array on wire: %s", body)
		}
		for _, b := range g {
			blk := b.(map[string]any)
			if format == FormatAnthropic {
				if blk["type"] == "text" {
					if s, ok := blk["text"].(string); !ok || s == "" {
						t.Errorf("text block without non-empty text: %v in %s", blk, body)
					}
				}
			} else if len(blk) == 0 {
				t.Errorf("empty Gemini part {} on wire: %s", body)
			}
		}
	}
}

func TestM3EmptyTextSkippedWhenOtherContent(t *testing.T) {
	cases := []struct {
		name string
		req  *ChatRequest
	}{
		{"in-band empty system message", &ChatRequest{Messages: []Message{
			{Role: RoleSystem, Content: ""}, {Role: RoleUser, Content: "hi"}}}},
		{"empty SystemBlock", &ChatRequest{System: []SystemBlock{{Text: ""}},
			Messages: []Message{{Role: RoleUser, Content: "hi"}}}},
		{"empty SystemBlock beside real one", &ChatRequest{System: []SystemBlock{{Text: ""}, {Text: "be terse"}},
			Messages: []Message{{Role: RoleUser, Content: "hi"}}}},
	}
	for _, format := range []Format{FormatAnthropic, FormatGemini} {
		for _, c := range cases {
			t.Run(string(format)+"/"+c.name, func(t *testing.T) {
				rec, err := m3Chat(t, format, c.req)
				if err != nil {
					t.Fatalf("empty system text must be skipped, not fail: %v", err)
				}
				_, _, _, body := rec.snapshot()
				m3AssertValidBlocks(t, format, body)
				if !strings.Contains(string(body), "hi") {
					t.Fatalf("user content lost: %s", body)
				}
				if strings.Contains(c.name, "beside") && !strings.Contains(string(body), "be terse") {
					t.Fatalf("real system block lost: %s", body)
				}
			})
		}
	}
}

func TestM3MessageWithNoContentRejected(t *testing.T) {
	cases := []struct {
		name string
		msgs []Message
	}{
		{"user empty content", []Message{{Role: RoleUser, Content: ""}}},
		{"user empty after history", []Message{{Role: RoleUser, Content: "a"}, {Role: RoleAssistant, Content: "b"}, {Role: RoleUser, Content: ""}}},
	}
	for _, format := range []Format{FormatAnthropic, FormatGemini} {
		for _, c := range cases {
			t.Run(string(format)+"/"+c.name, func(t *testing.T) {
				rec, err := m3Chat(t, format, &ChatRequest{Messages: c.msgs})
				var ce *ConfigError
				if !errors.As(err, &ce) {
					t.Fatalf("err = %T (%v), want *ConfigError", err, err)
				}
				if hits, _, _, _ := rec.snapshot(); hits != 0 {
					t.Fatalf("invalid request reached the network (%d hits)", hits)
				}
			})
		}
	}
}

// ── Bug 3: empty assistant turn leaves no whitespace-only block ─────────

func TestM3EmptyAssistantTurnDropped(t *testing.T) {
	cases := []struct {
		name      string
		msgs      []Message
		wantRoles []string
	}{
		{"between users merges them", []Message{
			{Role: RoleUser, Content: "first"}, {Role: RoleAssistant, Content: ""}, {Role: RoleUser, Content: "second"}},
			[]string{"user"}},
		{"trailing empty assistant dropped", []Message{
			{Role: RoleUser, Content: "first"}, {Role: RoleAssistant, Content: ""}},
			[]string{"user"}},
		{"mid-history keeps alternation", []Message{
			{Role: RoleUser, Content: "a"}, {Role: RoleAssistant, Content: "b"},
			{Role: RoleUser, Content: "c"}, {Role: RoleAssistant, Content: ""}, {Role: RoleUser, Content: "d"}},
			[]string{"user", "assistant", "user"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec, err := m3Chat(t, FormatAnthropic, &ChatRequest{Messages: c.msgs})
			if err != nil {
				t.Fatalf("call: %v", err)
			}
			_, _, _, body := rec.snapshot()
			m3AssertValidBlocks(t, FormatAnthropic, body)
			for _, g := range m3AllBlocks(t, FormatAnthropic, body) {
				for _, b := range g {
					if s, _ := b.(map[string]any)["text"].(string); s != "" && strings.TrimSpace(s) == "" {
						t.Errorf("whitespace-only text block on wire: %s", body)
					}
				}
			}
			var roles []string
			for _, msg := range m3JSON(t, body)["messages"].([]any) {
				roles = append(roles, msg.(map[string]any)["role"].(string))
			}
			if fmt.Sprint(roles) != fmt.Sprint(c.wantRoles) {
				t.Fatalf("roles = %v, want %v (alternation)\n%s", roles, c.wantRoles, body)
			}
			// merged user turn keeps every original text
			for _, m := range c.msgs {
				if m.Role == RoleUser && !strings.Contains(string(body), `"`+m.Content+`"`) {
					t.Errorf("user text %q lost: %s", m.Content, body)
				}
			}
		})
	}
}

// ── Bug 4: Gemini tool schemas travel as parametersJsonSchema ───────────

func TestM3GeminiToolSchemaIsJSONSchema(t *testing.T) {
	schemas := map[string]string{
		"nullable type array":  `{"type":"object","properties":{"a":{"type":["string","null"]}}}`,
		"ref and defs":         `{"type":"object","$defs":{"x":{"type":"string"}},"properties":{"a":{"$ref":"#/$defs/x"}}}`,
		"const and additional": `{"type":"object","additionalProperties":false,"properties":{"k":{"const":"v"}}}`,
	}
	for name, schema := range schemas {
		t.Run(name, func(t *testing.T) {
			req := &ChatRequest{
				Messages: []Message{{Role: RoleUser, Content: "hi"}},
				Tools:    []ToolDef{{Name: "f", Description: "d", Parameters: json.RawMessage(schema)}},
			}
			rec, err := m3Chat(t, FormatGemini, req)
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, body := rec.snapshot()
			tools := m3JSON(t, body)["tools"].([]any)
			decl := tools[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
			if _, bad := decl["parameters"]; bad {
				t.Errorf("JSON Schema must not ride the restricted OpenAPI `parameters` field: %v", decl)
			}
			got, ok := decl["parametersJsonSchema"]
			if !ok {
				t.Fatalf("parametersJsonSchema missing: %v", decl)
			}
			gb, _ := json.Marshal(got)
			var want, have any
			_ = json.Unmarshal([]byte(schema), &want)
			_ = json.Unmarshal(gb, &have)
			if fmt.Sprint(want) != fmt.Sprint(have) {
				t.Errorf("schema altered: got %s want %s", gb, schema)
			}
		})
	}
	t.Run("no schema sends neither field", func(t *testing.T) {
		req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}, Tools: []ToolDef{{Name: "f"}}}
		rec, err := m3Chat(t, FormatGemini, req)
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, body := rec.snapshot()
		if strings.Contains(string(body), `"parameters"`) {
			t.Errorf("unexpected parameters key: %s", body)
		}
	})
}

// ── Bug 5: every Gemini URL builder normalizes the model id ─────────────

func TestM3GeminiModelURLNormalized(t *testing.T) {
	models := []struct{ in, wantSeg string }{
		{"models/gemini-x", "gemini-x"},
		{"gemini-x", "gemini-x"},
		{"gem/ini?x=1", url.PathEscape("gem/ini?x=1")},
	}
	ttsBody := `{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/L16;rate=24000","data":"AQID"}}]}}]}`
	sseBody := "data: " + m3Reply(FormatGemini) + "\n\n"
	for _, mm := range models {
		check := func(t *testing.T, rec *m3Wire, suffix string) {
			t.Helper()
			_, esc, rawQ, _ := rec.snapshot()
			if want := "/v1beta/models/" + mm.wantSeg + suffix; esc != want {
				t.Errorf("path = %q, want %q", esc, want)
			}
			if rawQ != "" && !strings.HasPrefix(suffix, ":streamGenerate") {
				t.Errorf("model smuggled a query: %q", rawQ)
			}
		}
		t.Run("chat/"+mm.in, func(t *testing.T) {
			srv, rec := m3Server(t, m3Reply(FormatGemini), false)
			c := newTestClient(t, ProviderConfig{ID: "g", Format: FormatGemini, BaseURL: srv.URL, APIKey: "k"}, srv)
			c.model = mm.in
			if _, err := c.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err != nil {
				t.Fatal(err)
			}
			check(t, rec, ":generateContent")
		})
		t.Run("stream/"+mm.in, func(t *testing.T) {
			srv, rec := m3Server(t, sseBody, true)
			c := newTestClient(t, ProviderConfig{ID: "g", Format: FormatGemini, BaseURL: srv.URL, APIKey: "k"}, srv)
			c.model = mm.in
			if _, err := c.CallStream(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}, func(Delta) error { return nil }); err != nil {
				t.Fatal(err)
			}
			check(t, rec, ":streamGenerateContent")
		})
		t.Run("embed/"+mm.in, func(t *testing.T) {
			srv, rec := m3Server(t, `{"embeddings":[{"values":[0.1]}]}`, false)
			s := newTestSDK(t, ProviderConfig{ID: "g", Format: FormatGemini, APIKey: "k"}, srv)
			if _, err := s.Embed(context.Background(), "g", mm.in, EmbedRequest{Inputs: []string{"a"}}); err != nil {
				t.Fatal(err)
			}
			check(t, rec, ":batchEmbedContents")
		})
		t.Run("tts/"+mm.in, func(t *testing.T) {
			srv, rec := m3Server(t, ttsBody, false)
			s := newTestSDK(t, ProviderConfig{ID: "g", Format: FormatGemini, APIKey: "k"}, srv)
			if _, err := s.Speak(context.Background(), "g", mm.in, SpeakRequest{Text: "hi", Voice: "Kore"}); err != nil {
				t.Fatal(err)
			}
			check(t, rec, ":generateContent")
		})
	}
}

// ── Bug 6: non-JSON transcription formats return the raw body as Text ───

func TestM3TranscribeNonJSONFormatsReturnRawBody(t *testing.T) {
	cases := []struct {
		format, ctype, body string
	}{
		{"text", "text/plain; charset=utf-8", "hello world\n"},
		{"srt", "application/x-subrip", "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"},
		{"vtt", "text/vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n"},
	}
	for _, c := range cases {
		t.Run(c.format, func(t *testing.T) {
			srv, rec := newSTTServer(http.StatusOK, c.ctype, c.body)
			defer srv.Close()
			s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
			res, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}, Format: c.format})
			if err != nil {
				t.Fatalf("billed transcription must not fail on a %s body: %v", c.format, err)
			}
			if res.Text != c.body {
				t.Errorf("Text = %q, want raw body %q", res.Text, c.body)
			}
			if res.Model != "whisper-1" {
				t.Errorf("Model = %q", res.Model)
			}
			vals, _ := sttParse(t, *rec)
			if vals["response_format"] != c.format {
				t.Errorf("response_format = %q, want %q", vals["response_format"], c.format)
			}
		})
	}
	// JSON formats keep decoding (guard against over-broad raw handling).
	for _, f := range []string{"", "json", "verbose_json"} {
		t.Run("json/"+f, func(t *testing.T) {
			srv, _ := newSTTServer(http.StatusOK, "application/json", `{"text":"hi","language":"en","duration":1.5}`)
			defer srv.Close()
			s := newTestSDK(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, APIKey: "k"}, srv)
			res, err := s.Transcribe(t.Context(), "openai", "whisper-1", TranscribeRequest{Audio: []byte{1}, Format: f})
			if err != nil || res.Text != "hi" {
				t.Fatalf("res=%+v err=%v", res, err)
			}
		})
	}
}
