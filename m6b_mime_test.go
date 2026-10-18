package llm

import "testing"

// MIME types are case-insensitive (RFC 2045): any case is accepted at the
// boundary and the canonical lower-case form goes on the wire.

func TestM6bDocumentMIMECaseInsensitive(t *testing.T) {
	for _, mt := range []string{"APPLICATION/PDF", "Application/Pdf"} {
		msg := m6bUser(DocumentPart(mt, m6bPDF, "a.pdf"))

		m, _, err := m6bBuild(FormatOpenAI, "gpt-4o", msg)
		if err != nil {
			t.Errorf("openai %q: %v", mt, err)
		} else {
			jsonEq(t, "openai "+mt, m6bGet(t, m, "messages", 0, "content", 0),
				`{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,AQID"}}`)
		}

		m, _, err = m6bBuild(FormatOpenAI, "gpt-5.6", msg)
		if err != nil {
			t.Errorf("responses %q: %v", mt, err)
		} else {
			jsonEq(t, "responses "+mt, m6bGet(t, m, "input", 0, "content", 0),
				`{"type":"input_file","filename":"a.pdf","file_data":"data:application/pdf;base64,AQID"}`)
		}

		m, _, err = m6bBuild(FormatAnthropic, "claude", msg)
		if err != nil {
			t.Errorf("anthropic %q: %v", mt, err)
		} else {
			jsonEq(t, "anthropic "+mt, m6bGet(t, m, "messages", 0, "content", 0),
				`{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AQID"},"title":"a.pdf"}`)
		}

		m, _, err = m6bBuild(FormatGemini, "gemini", msg)
		if err != nil {
			t.Errorf("gemini %q: %v", mt, err)
		} else {
			jsonEq(t, "gemini "+mt, m6bGet(t, m, "contents", 0, "parts", 0),
				`{"inlineData":{"mimeType":"application/pdf","data":"AQID"}}`)
		}
	}
}

func TestM6bDocumentTextPlainMIMECaseInsensitive(t *testing.T) {
	msg := m6bUser(DocumentPart("TEXT/Plain", []byte("hello"), "n.txt"))
	m, _, err := m6bBuild(FormatAnthropic, "claude", msg)
	if err != nil {
		t.Fatalf("anthropic: %v", err)
	}
	jsonEq(t, "anthropic", m6bGet(t, m, "messages", 0, "content", 0),
		`{"type":"document","source":{"type":"text","media_type":"text/plain","data":"hello"},"title":"n.txt"}`)
	m, _, err = m6bBuild(FormatGemini, "gemini", msg)
	if err != nil {
		t.Fatalf("gemini: %v", err)
	}
	jsonEq(t, "gemini", m6bGet(t, m, "contents", 0, "parts", 0), `{"inlineData":{"mimeType":"text/plain","data":"aGVsbG8="}}`)
}

func TestM6bImageURLMIMECaseInsensitive(t *testing.T) {
	const u = "https://x/y.png"
	cases := map[string]string{"IMAGE/PNG": "image/png", "Image/Jpeg": "image/jpeg", "IMAGE/JPG": "image/jpeg"}
	for in, want := range cases {
		for _, f := range m6bFormats {
			if _, _, err := m6bBuild(f.format, f.model, m6bUser(ImageURLPartMIME(u, in))); err != nil {
				t.Errorf("%s %q: %v", f.name, in, err)
			}
		}
		m, _, err := m6bBuild(FormatGemini, "gemini", m6bUser(ImageURLPartMIME(u, in)))
		if err != nil {
			continue
		}
		jsonEq(t, "gemini "+in, m6bGet(t, m, "contents", 0, "parts", 0),
			`{"fileData":{"fileUri":"https://x/y.png","mimeType":"`+want+`"}}`)
	}
}
