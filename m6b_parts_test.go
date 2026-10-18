package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// m6b: DocumentPart and ImageURLPart on every format (AC-1, AC-2).

var m6bFormats = []struct {
	name   string
	format Format
	model  string
}{
	{"openai", FormatOpenAI, "gpt-4o"},
	{"responses", FormatOpenAI, "gpt-5.6"},
	{"anthropic", FormatAnthropic, "claude"},
	{"gemini", FormatGemini, "gemini"},
}

var m6bTools = []ToolDef{{Name: "wx", Parameters: json.RawMessage(`{"type":"object"}`)}}

// m6bBuild renders req through the real boundary + builder path. Tools are
// attached so gpt-5.6 routes to /responses.
func m6bBuild(format Format, model string, msgs ...Message) (map[string]any, string, error) {
	pc := newProviderClient(ProviderConfig{ID: string(format), Format: format, BaseURL: "https://x"}, nil, nil)
	b, url, err := pc.buildChatRequest(&ChatRequest{Messages: msgs, Tools: m6bTools}, model, false)
	if err != nil {
		return nil, url, err
	}
	var m map[string]any
	if jerr := json.Unmarshal(b, &m); jerr != nil {
		return nil, url, jerr
	}
	return m, url, nil
}

func m6bMustBuild(t *testing.T, format Format, model string, msgs ...Message) map[string]any {
	t.Helper()
	m, _, err := m6bBuild(format, model, msgs...)
	if err != nil {
		t.Fatalf("%s: build: %v", format, err)
	}
	return m
}

// m6bWantConfigErr requires a *ConfigError that is a deliberate rejection:
// never the generic unknown-part-type error a not-yet-supported type hits.
func m6bWantConfigErr(t *testing.T, name string, err error, contains ...string) {
	t.Helper()
	if err == nil {
		t.Errorf("%s: accepted, want *ConfigError", name)
		return
	}
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Errorf("%s: err = %T %v, want *ConfigError", name, err, err)
		return
	}
	if strings.Contains(err.Error(), "unknown content part type") {
		t.Errorf("%s: rejected as unknown part type, want a specific rejection: %v", name, err)
	}
	for _, c := range contains {
		if !strings.Contains(err.Error(), c) {
			t.Errorf("%s: error %q lacks %q", name, err.Error(), c)
		}
	}
}

func m6bGet(t *testing.T, v any, path ...any) any {
	t.Helper()
	cur := v
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, ok := cur.(map[string]any)
			if !ok {
				t.Fatalf("path %v: %T is not an object at %q", path, cur, k)
			}
			cur = m[k]
		case int:
			a, ok := cur.([]any)
			if !ok || k >= len(a) {
				t.Fatalf("path %v: no index %d in %T", path, k, cur)
			}
			cur = a[k]
		}
	}
	return cur
}

var (
	m6bPDF  = []byte{1, 2, 3} // base64 "AQID"
	m6bUser = func(parts ...ContentPart) Message { return Message{Role: RoleUser, Parts: parts} }
)

func TestM6bConstructors(t *testing.T) {
	d := DocumentPart("application/pdf", m6bPDF, "a.pdf")
	if d.Type != ContentPartDocument || d.MIMEType != "application/pdf" || !bytes.Equal(d.Data, m6bPDF) || d.Name != "a.pdf" || d.Text != "" || d.Image != nil || d.URL != "" {
		t.Errorf("DocumentPart = %+v", d)
	}
	u := ImageURLPart("https://x/y.png")
	if u.Type != ContentPartImageURL || u.URL != "https://x/y.png" || u.MIMEType != "" || u.Image != nil {
		t.Errorf("ImageURLPart = %+v", u)
	}
	um := ImageURLPartMIME("https://x/y.png", "image/png")
	if um.Type != ContentPartImageURL || um.URL != "https://x/y.png" || um.MIMEType != "image/png" {
		t.Errorf("ImageURLPartMIME = %+v", um)
	}
	w := u.WithMIME("image/webp")
	if w.Type != ContentPartImageURL || w.URL != u.URL || w.MIMEType != "image/webp" {
		t.Errorf("WithMIME = %+v", w)
	}
	if u.MIMEType != "" {
		t.Errorf("WithMIME mutated the receiver: %+v", u)
	}
	if ContentPartDocument != "document" || ContentPartImageURL != "image_url" {
		t.Errorf("part type consts = %q %q", ContentPartDocument, ContentPartImageURL)
	}
	if MaxDocumentBytes != 20<<20 {
		t.Errorf("MaxDocumentBytes = %d", MaxDocumentBytes)
	}
}

// ── AC-1: documents on every format ──────────────────────────────────────

func TestM6bDocumentWirePDF(t *testing.T) {
	msg := m6bUser(TextPart("read"), DocumentPart("application/pdf", m6bPDF, "a.pdf"))

	m := m6bMustBuild(t, FormatOpenAI, "gpt-4o", msg)
	jsonEq(t, "openai content", m6bGet(t, m, "messages", 0, "content"),
		`[{"type":"text","text":"read"},{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,AQID"}}]`)

	m = m6bMustBuild(t, FormatOpenAI, "gpt-5.6", msg)
	jsonEq(t, "responses content", m6bGet(t, m, "input", 0, "content"),
		`[{"type":"input_text","text":"read"},{"type":"input_file","filename":"a.pdf","file_data":"data:application/pdf;base64,AQID"}]`)

	m = m6bMustBuild(t, FormatAnthropic, "claude", msg)
	jsonEq(t, "anthropic content", m6bGet(t, m, "messages", 0, "content"),
		`[{"type":"text","text":"read"},{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AQID"},"title":"a.pdf"}]`)

	m = m6bMustBuild(t, FormatGemini, "gemini", msg)
	jsonEq(t, "gemini parts", m6bGet(t, m, "contents", 0, "parts"),
		`[{"text":"read"},{"inlineData":{"mimeType":"application/pdf","data":"AQID"}}]`)
}

func TestM6bDocumentDefaultNameOpenAIFormats(t *testing.T) {
	msg := m6bUser(DocumentPart("application/pdf", m6bPDF, ""))
	m := m6bMustBuild(t, FormatOpenAI, "gpt-4o", msg)
	jsonEq(t, "openai", m6bGet(t, m, "messages", 0, "content", 0),
		`{"type":"file","file":{"filename":"document.pdf","file_data":"data:application/pdf;base64,AQID"}}`)
	m = m6bMustBuild(t, FormatOpenAI, "gpt-5.6", msg)
	jsonEq(t, "responses", m6bGet(t, m, "input", 0, "content", 0),
		`{"type":"input_file","filename":"document.pdf","file_data":"data:application/pdf;base64,AQID"}`)
	// Anthropic title is omitted when no name is given.
	m = m6bMustBuild(t, FormatAnthropic, "claude", msg)
	jsonEq(t, "anthropic", m6bGet(t, m, "messages", 0, "content", 0),
		`{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AQID"}}`)
}

func TestM6bDocumentTextPlain(t *testing.T) {
	msg := m6bUser(DocumentPart("text/plain", []byte("hello"), "n.txt"))

	m := m6bMustBuild(t, FormatAnthropic, "claude", msg)
	jsonEq(t, "anthropic", m6bGet(t, m, "messages", 0, "content", 0),
		`{"type":"document","source":{"type":"text","media_type":"text/plain","data":"hello"},"title":"n.txt"}`)

	m = m6bMustBuild(t, FormatGemini, "gemini", msg)
	jsonEq(t, "gemini", m6bGet(t, m, "contents", 0, "parts", 0),
		`{"inlineData":{"mimeType":"text/plain","data":"aGVsbG8="}}`)

	_, _, err := m6bBuild(FormatOpenAI, "gpt-4o", msg)
	m6bWantConfigErr(t, "openai text/plain", err, "application/pdf only")
	_, _, err = m6bBuild(FormatOpenAI, "gpt-5.6", msg)
	m6bWantConfigErr(t, "responses text/plain", err, "application/pdf only")
}

func TestM6bDocumentBoundaryErrors(t *testing.T) {
	big := make([]byte, MaxDocumentBytes+1)
	cases := []struct {
		name string
		part ContentPart
	}{
		{"image mime", ContentPart{Type: ContentPartDocument, MIMEType: "image/png", Data: m6bPDF}},
		{"empty mime", ContentPart{Type: ContentPartDocument, Data: m6bPDF}},
		{"mime with params", ContentPart{Type: ContentPartDocument, MIMEType: "application/pdf; charset=x", Data: m6bPDF}},
		{"unsupported mime", ContentPart{Type: ContentPartDocument, MIMEType: "application/zip", Data: m6bPDF}},
		{"empty data", ContentPart{Type: ContentPartDocument, MIMEType: "application/pdf"}},
		{"oversized", ContentPart{Type: ContentPartDocument, MIMEType: "application/pdf", Data: big}},
		{"text set", ContentPart{Type: ContentPartDocument, MIMEType: "application/pdf", Data: m6bPDF, Text: "x"}},
		{"image set", ContentPart{Type: ContentPartDocument, MIMEType: "application/pdf", Data: m6bPDF, Image: []byte{1}}},
		{"url set", ContentPart{Type: ContentPartDocument, MIMEType: "application/pdf", Data: m6bPDF, URL: "https://x/a.pdf"}},
	}
	for _, f := range m6bFormats {
		for _, c := range cases {
			_, _, err := m6bBuild(f.format, f.model, m6bUser(c.part))
			m6bWantConfigErr(t, f.name+"/"+c.name, err, "message 0 part 0")
		}
	}
}

func TestM6bDocumentAtLimitAccepted(t *testing.T) {
	msg := m6bUser(DocumentPart("application/pdf", make([]byte, MaxDocumentBytes), "big.pdf"))
	m := m6bMustBuild(t, FormatAnthropic, "claude", msg)
	if got := m6bGet(t, m, "messages", 0, "content", 0, "type"); got != "document" {
		t.Errorf("type = %v", got)
	}
}

func TestM6bDocumentRoleRules(t *testing.T) {
	doc := DocumentPart("application/pdf", m6bPDF, "a.pdf")
	for _, role := range []Role{RoleAssistant, RoleSystem} {
		_, _, err := m6bBuild(FormatAnthropic, "claude", Message{Role: role, Parts: []ContentPart{doc}})
		m6bWantConfigErr(t, string(role), err, "require the user or tool role")
	}
	// Content and Parts together stay rejected.
	_, _, err := m6bBuild(FormatAnthropic, "claude", Message{Role: RoleUser, Content: "x", Parts: []ContentPart{doc}})
	m6bWantConfigErr(t, "content+parts", err, "cannot both be set")
}

func TestM6bTextAndImagePartsRejectForeignFields(t *testing.T) {
	cases := []struct {
		name string
		part ContentPart
	}{
		{"text+url", ContentPart{Type: ContentPartText, Text: "x", URL: "https://x/y.png"}},
		{"text+data", ContentPart{Type: ContentPartText, Text: "x", Data: m6bPDF}},
		{"text+name", ContentPart{Type: ContentPartText, Text: "x", Name: "n"}},
		{"image+data", ContentPart{Type: ContentPartImage, MIMEType: "image/png", Image: []byte{1}, Data: m6bPDF}},
		{"image+url", ContentPart{Type: ContentPartImage, MIMEType: "image/png", Image: []byte{1}, URL: "https://x/y.png"}},
		{"image+name", ContentPart{Type: ContentPartImage, MIMEType: "image/png", Image: []byte{1}, Name: "n"}},
	}
	for _, c := range cases {
		_, _, err := m6bBuild(FormatOpenAI, "gpt-4o", m6bUser(c.part))
		m6bWantConfigErr(t, c.name, err, "message 0 part 0")
	}
}

func TestM6bAggregateInlineCapCountsDocuments(t *testing.T) {
	doc := func(n int) ContentPart { return DocumentPart("application/pdf", make([]byte, n), "d.pdf") }
	cases := []struct {
		name  string
		parts []ContentPart
	}{
		{"documents only", []ContentPart{doc(12 << 20), doc(12 << 20), doc(12 << 20)}},
		{"images plus document", []ContentPart{
			ImagePart("image/png", make([]byte, 10<<20)), ImagePart("image/png", make([]byte, 10<<20)), doc(13 << 20)}},
	}
	for _, f := range m6bFormats {
		for _, c := range cases {
			_, _, err := m6bBuild(f.format, f.model, m6bUser(c.parts...))
			m6bWantConfigErr(t, f.name+"/"+c.name, err, "exceed")
		}
	}
	// Exactly at the cap is accepted.
	half := (MaxRequestImageBytes - MaxDocumentBytes) / 2
	ok := []ContentPart{doc(MaxDocumentBytes), ImagePart("image/png", make([]byte, half)), ImagePart("image/png", make([]byte, half))}
	if _, _, err := m6bBuild(FormatAnthropic, "claude", m6bUser(ok...)); err != nil {
		t.Errorf("at-cap request rejected: %v", err)
	}
}

// ── AC-2: image URLs on every format ─────────────────────────────────────

func TestM6bImageURLWire(t *testing.T) {
	const u = "https://x/y.png"
	plain := m6bUser(TextPart("see"), ImageURLPart(u))
	hinted := m6bUser(TextPart("see"), ImageURLPartMIME(u, "image/png"))

	m := m6bMustBuild(t, FormatOpenAI, "gpt-4o", plain)
	jsonEq(t, "openai", m6bGet(t, m, "messages", 0, "content", 1), `{"type":"image_url","image_url":{"url":"https://x/y.png"}}`)

	m = m6bMustBuild(t, FormatOpenAI, "gpt-5.6", plain)
	jsonEq(t, "responses", m6bGet(t, m, "input", 0, "content", 1), `{"type":"input_image","image_url":"https://x/y.png"}`)

	for _, msg := range []Message{plain, hinted} { // Anthropic ignores a MIME hint
		m = m6bMustBuild(t, FormatAnthropic, "claude", msg)
		jsonEq(t, "anthropic", m6bGet(t, m, "messages", 0, "content", 1), `{"type":"image","source":{"type":"url","url":"https://x/y.png"}}`)
	}

	for name, p := range map[string]ContentPart{
		"ImageURLPartMIME": ImageURLPartMIME(u, "image/png"),
		"WithMIME":         ImageURLPart(u).WithMIME("image/png"),
	} {
		m = m6bMustBuild(t, FormatGemini, "gemini", m6bUser(TextPart("see"), p))
		jsonEq(t, "gemini "+name, m6bGet(t, m, "contents", 0, "parts", 1), `{"fileData":{"fileUri":"https://x/y.png","mimeType":"image/png"}}`)
	}

	m = m6bMustBuild(t, FormatGemini, "gemini", m6bUser(ImageURLPartMIME(u, "image/jpg")))
	jsonEq(t, "gemini jpg alias", m6bGet(t, m, "contents", 0, "parts", 0), `{"fileData":{"fileUri":"https://x/y.png","mimeType":"image/jpeg"}}`)
}

func TestM6bImageURLSchemeCaseInsensitive(t *testing.T) {
	for _, f := range m6bFormats {
		msg := m6bUser(ImageURLPartMIME("HTTP://x/y.png", "image/png"))
		if _, _, err := m6bBuild(f.format, f.model, msg); err != nil {
			t.Errorf("%s: uppercase http scheme rejected: %v", f.name, err)
		}
	}
}

func TestM6bGeminiImageURLRequiresMIME(t *testing.T) {
	_, _, err := m6bBuild(FormatGemini, "gemini", m6bUser(ImageURLPart("https://x/y.png")))
	m6bWantConfigErr(t, "gemini no mime", err, "require a MIME type")
	// The other formats do not need a MIME for a URL image.
	for _, f := range m6bFormats[:3] {
		if _, _, err := m6bBuild(f.format, f.model, m6bUser(ImageURLPart("https://x/y.png"))); err != nil {
			t.Errorf("%s: URL image without MIME rejected: %v", f.name, err)
		}
	}
}

func TestM6bImageURLBoundaryErrors(t *testing.T) {
	cases := []struct {
		name string
		part ContentPart
	}{
		{"data url", m6bURLPart("data:image/png;base64,AAAA")},
		{"ftp", m6bURLPart("ftp://x/y.png")},
		{"file", m6bURLPart("file:///etc/passwd")},
		{"empty host", m6bURLPart("https:///y.png")},
		{"malformed", m6bURLPart("http://[::1")},
		{"empty url", m6bURLPart("")},
		{"relative", m6bURLPart("/y.png")},
		{"too long", m6bURLPart("https://x/" + strings.Repeat("a", 9<<10))},
		{"image bytes set", ContentPart{Type: ContentPartImageURL, URL: "https://x/y.png", MIMEType: "image/png", Image: []byte{1}}},
		{"text set", ContentPart{Type: ContentPartImageURL, URL: "https://x/y.png", MIMEType: "image/png", Text: "x"}},
		{"data set", ContentPart{Type: ContentPartImageURL, URL: "https://x/y.png", MIMEType: "image/png", Data: m6bPDF}},
		{"non-image mime hint", m6bURLPart("https://x/y.png").WithMIME("text/plain")},
		{"pdf mime hint", m6bURLPart("https://x/y.png").WithMIME("application/pdf")},
	}
	for _, f := range m6bFormats {
		for _, c := range cases {
			_, _, err := m6bBuild(f.format, f.model, m6bUser(c.part))
			m6bWantConfigErr(t, f.name+"/"+c.name, err, "message 0 part 0")
		}
	}
}

// m6bURLPart builds an image_url part directly (not via the constructor)
// so boundary cases do not depend on the constructors under test.
func m6bURLPart(u string) ContentPart { return ContentPart{Type: ContentPartImageURL, URL: u} }

// The SDK never fetches an image URL itself: it only forwards the string.
func TestM6bImageURLNeverFetchedBySDK(t *testing.T) {
	var imageHits atomic.Int32
	imgSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		imageHits.Add(1)
		w.WriteHeader(200)
	}))
	defer imgSrv.Close()
	imgURL := imgSrv.URL + "/y.png"

	var gotBody atomic.Value
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b := new(bytes.Buffer)
		_, _ = b.ReadFrom(r.Body)
		gotBody.Store(b.String())
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer api.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: api.URL, APIKey: "k"}, api)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{m6bUser(TextPart("see"), ImageURLPart(imgURL))}})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if res.Content != "ok" {
		t.Fatalf("content = %q", res.Content)
	}
	if body, _ := gotBody.Load().(string); !strings.Contains(body, imgURL) {
		t.Errorf("request body does not carry the URL verbatim: %s", body)
	}
	if n := imageHits.Load(); n != 0 {
		t.Errorf("image URL fetched %d times by the SDK, want 0", n)
	}
}

// Invalid parts fail before any HTTP request is made.
func TestM6bInvalidPartsFailBeforeNetwork(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(200)
	}))
	defer srv.Close()
	bad := []ContentPart{
		{Type: ContentPartDocument, MIMEType: "image/png", Data: m6bPDF},
		m6bURLPart("ftp://x/y.png"),
	}
	for _, f := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
		cc := newTestClient(t, ProviderConfig{ID: string(f), Format: f, BaseURL: srv.URL, APIKey: "k"}, srv)
		for i, p := range bad {
			_, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{m6bUser(p)}})
			m6bWantConfigErr(t, fmt.Sprintf("%s/%d", f, i), err, "message 0 part 0")
		}
	}
	if n := hits.Load(); n != 0 {
		t.Errorf("%d HTTP requests made for invalid parts", n)
	}
}

// ── tool-result placement ────────────────────────────────────────────────

func m6bToolTurn(parts ...ContentPart) []Message {
	return []Message{
		{Role: RoleUser, Content: "go"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "wx", Arguments: `{}`}}},
		{Role: RoleTool, ToolCallID: "c1", ToolName: "wx", Parts: parts},
	}
}

func TestM6bToolResultAnthropic(t *testing.T) {
	m := m6bMustBuild(t, FormatAnthropic, "claude", m6bToolTurn(
		TextPart("r"), DocumentPart("application/pdf", m6bPDF, "a.pdf"), ImageURLPart("https://x/y.png"))...)
	jsonEq(t, "tool_result", m6bGet(t, m, "messages", 2, "content", 0, "content"),
		`[{"type":"text","text":"r"},`+
			`{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AQID"},"title":"a.pdf"},`+
			`{"type":"image","source":{"type":"url","url":"https://x/y.png"}}]`)
}

func TestM6bToolResultGemini(t *testing.T) {
	m := m6bMustBuild(t, FormatGemini, "gemini", m6bToolTurn(
		TextPart("r"), DocumentPart("application/pdf", m6bPDF, "a.pdf"), ImageURLPartMIME("https://x/y.png", "image/png"))...)
	parts := m6bGet(t, m, "contents", 2, "parts")
	jsonEq(t, "functionResponse", m6bGet(t, parts, 0), `{"functionResponse":{"name":"wx","response":{"result":"r"}}}`)
	jsonEq(t, "document", m6bGet(t, parts, 1), `{"inlineData":{"mimeType":"application/pdf","data":"AQID"}}`)
	jsonEq(t, "image url", m6bGet(t, parts, 2), `{"fileData":{"fileUri":"https://x/y.png","mimeType":"image/png"}}`)
}

func TestM6bToolResultOpenAIChat(t *testing.T) {
	m := m6bMustBuild(t, FormatOpenAI, "gpt-4o", m6bToolTurn(TextPart("chart"), ImageURLPart("https://x/y.png"))...)
	msgs := m6bGet(t, m, "messages").([]any)
	if len(msgs) != 4 {
		t.Fatalf("messages = %d, want user, assistant, tool, user", len(msgs))
	}
	jsonEq(t, "tool message", m6bGet(t, msgs, 2, "content"), `"chart"`)
	if role := m6bGet(t, msgs, 3, "role"); role != "user" {
		t.Errorf("post-tool message role = %v, want user", role)
	}
	content := m6bGet(t, msgs, 3, "content").([]any)
	jsonEq(t, "image rides user message", content[len(content)-1], `{"type":"image_url","image_url":{"url":"https://x/y.png"}}`)

	_, _, err := m6bBuild(FormatOpenAI, "gpt-4o", m6bToolTurn(TextPart("r"), DocumentPart("application/pdf", m6bPDF, "a.pdf"))...)
	m6bWantConfigErr(t, "openai tool doc", err, "tool results cannot carry documents on OpenAI-format providers")
}

func TestM6bToolResultResponses(t *testing.T) {
	m := m6bMustBuild(t, FormatOpenAI, "gpt-5.6", m6bToolTurn(TextPart("chart"), ImageURLPart("https://x/y.png"))...)
	jsonEq(t, "function_call_output", m6bGet(t, m, "input", 2),
		`{"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"chart"},{"type":"input_image","image_url":"https://x/y.png"}]}`)

	_, _, err := m6bBuild(FormatOpenAI, "gpt-5.6", m6bToolTurn(TextPart("r"), DocumentPart("application/pdf", m6bPDF, "a.pdf"))...)
	m6bWantConfigErr(t, "responses tool doc", err, "tool results cannot carry documents on the Responses API")
}
