package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func multimodalRequest() *ChatRequest {
	return &ChatRequest{Messages: []Message{{Role: RoleUser, Parts: []ContentPart{TextPart("before"), ImagePart("image/png", []byte{1, 2, 3}), TextPart("after")}}}}
}

func TestMultimodalContentPartsOpenAI(t *testing.T) {
	b := buildOpenAIRequest(ProviderConfig{}, multimodalRequest(), "gpt-4o", false, false)
	if len(b.Messages) != 1 {
		t.Fatalf("messages=%d", len(b.Messages))
	}
	raw, err := json.Marshal(b.Messages[0])
	if err != nil {
		t.Fatal(err)
	}
	got := string(raw)
	if !strings.Contains(got, `"content":[{"type":"text"`) || !strings.Contains(got, `data:image/png;base64,AQID`) {
		t.Fatalf("content=%s", got)
	}
}

func TestMultimodalContentPartsOtherFormats(t *testing.T) {
	for _, format := range []Format{FormatAnthropic, FormatGemini} {
		var body []byte
		var err error
		if format == FormatAnthropic {
			body, err = buildAnthropicRequest(multimodalRequest(), "claude", false)
		} else {
			body, err = buildGeminiRequest(multimodalRequest(), "gemini", false)
		}
		if err != nil {
			t.Fatal(err)
		}
		got := string(body)
		if !strings.Contains(got, `"image/png"`) || !strings.Contains(got, `AQID`) || !strings.Contains(got, "before") || !strings.Contains(got, "after") {
			t.Fatalf("%s body=%s", format, got)
		}
	}
	_, input := buildResponsesInput(multimodalRequest())
	got, _ := json.Marshal(input)
	if !strings.Contains(string(got), `"input_image"`) || !strings.Contains(string(got), `data:image/png;base64,AQID`) {
		t.Fatalf("responses=%s", got)
	}
}

func TestMultimodalValidationBeforeNetwork(t *testing.T) {
	pc := newProviderClient(ProviderConfig{Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1"}, nil, nil)
	for _, m := range []Message{
		{Role: RoleUser, Content: "legacy", Parts: []ContentPart{TextPart("part")}},
		{Role: RoleUser, Parts: []ContentPart{{Type: ContentPartImage, MIMEType: "text/plain", Image: []byte("x")}}},
		{Role: RoleUser, Parts: []ContentPart{{Type: ContentPartImage, MIMEType: "image/png; charset=utf-8", Text: "hidden", Image: []byte("x")}}},
		{Role: RoleUser, Parts: []ContentPart{{Type: ContentPartImage, MIMEType: "image/bmp", Image: []byte("x")}}},
		{Role: RoleSystem, Parts: []ContentPart{ImagePart("image/png", []byte("x"))}},
		{Role: RoleUser, Parts: []ContentPart{{Type: "bad", Text: "x"}}},
	} {
		if _, _, err := pc.buildChatRequest(&ChatRequest{Messages: []Message{m}}, "gpt-4o", false); err == nil {
			t.Fatalf("invalid content accepted: %+v", m)
		}
	}
}

func TestMultimodalAggregateImageCap(t *testing.T) {
	parts := make([]ContentPart, 0, 4)
	for i := 0; i < 4; i++ {
		parts = append(parts, ImagePart("image/png", make([]byte, 9<<20)))
	}
	pc := newProviderClient(ProviderConfig{Format: FormatOpenAI, BaseURL: "http://example.invalid"}, nil, nil)
	if _, _, err := pc.buildChatRequest(&ChatRequest{Messages: []Message{{Role: RoleUser, Parts: parts}}}, "model", false); err == nil {
		t.Fatal("aggregate image cap accepted oversized request")
	}
}

func TestPlainContentSerializationUnchanged(t *testing.T) {
	b := buildOpenAIRequest(ProviderConfig{}, &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hello"}}}, "gpt-4o", false, false)
	raw, _ := json.Marshal(b.Messages[0].Content)
	if string(raw) != `"hello"` {
		t.Fatalf("content=%s", raw)
	}
}

func TestMultimodalStreamRequestPath(t *testing.T) {
	for _, format := range []Format{FormatOpenAI, FormatAnthropic, FormatGemini} {
		pc := newProviderClient(ProviderConfig{Format: format, BaseURL: "http://example.invalid"}, nil, nil)
		body, _, err := pc.buildChatRequest(multimodalRequest(), "model", true)
		if err != nil {
			t.Fatalf("%s: %v", format, err)
		}
		if !strings.Contains(string(body), "AQID") {
			t.Fatalf("%s stream body omitted image", format)
		}
	}
}

func TestMultimodalJPGAliasNormalizedOnWire(t *testing.T) {
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Parts: []ContentPart{ImagePart("image/jpg", []byte{1})}}}}
	b, err := buildAnthropicRequest(req, "claude", false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "image/jpg") || !strings.Contains(string(b), "image/jpeg") {
		t.Fatalf("anthropic media_type not normalized: %s", b)
	}
	g, err := buildGeminiRequest(req, "gemini", false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(g), "image/jpg") || !strings.Contains(string(g), "image/jpeg") {
		t.Fatalf("gemini mime_type not normalized: %s", g)
	}
	o := buildOpenAIRequest(ProviderConfig{}, req, "gpt-4o", false, false)
	raw, _ := json.Marshal(o.Messages[0])
	if strings.Contains(string(raw), "image/jpg") || !strings.Contains(string(raw), "image/jpeg") {
		t.Fatalf("openai data url not normalized: %s", raw)
	}
}

func TestMultimodalEmptyTextPartRejected(t *testing.T) {
	pc := newProviderClient(ProviderConfig{Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1"}, nil, nil)
	_, _, err := pc.buildChatRequest(&ChatRequest{Messages: []Message{{Role: RoleUser, Parts: []ContentPart{ImagePart("image/png", []byte("x")), {Type: ContentPartText}}}}}, "m", false)
	if err == nil || !strings.Contains(err.Error(), "text part is empty") {
		t.Fatalf("empty text part accepted: %v", err)
	}
}
