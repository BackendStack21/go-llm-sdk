package llm

import (
	"fmt"
	"mime"
	"net/url"
	"strings"
)

// MaxRequestImageBytes bounds the aggregate inline image payload in one
// request, preventing many individually valid parts from causing an
// unexpectedly large allocation or HTTP body.
const MaxRequestImageBytes = 32 << 20

func validateRequestContent(messages []Message) error {
	total := 0
	for i, m := range messages {
		if err := validateMessageContent(m, i); err != nil {
			return err
		}
		for _, p := range m.Parts {
			total += len(p.Image) + len(p.Data)
			if total > MaxRequestImageBytes {
				return &ConfigError{Msg: fmt.Sprintf("message %d: request inline images exceed %d bytes", i, MaxRequestImageBytes)}
			}
		}
	}
	return nil
}

func validateMessageContent(m Message, index int) error {
	if m.IsError && m.Role != RoleTool {
		return &ConfigError{Msg: fmt.Sprintf("message %d: IsError requires the tool role", index)}
	}
	if len(m.Parts) == 0 {
		return nil
	}
	if m.Content != "" {
		return &ConfigError{Msg: fmt.Sprintf("message %d: Content and Parts cannot both be set", index)}
	}
	if m.Role != RoleUser && m.Role != RoleTool {
		return &ConfigError{Msg: fmt.Sprintf("message %d: content parts require the user or tool role", index)}
	}
	for j, p := range m.Parts {
		switch p.Type {
		case ContentPartText:
			if p.Text == "" {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: text part is empty", index, j)}
			}
			if p.Image != nil || p.MIMEType != "" {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: text part has image fields", index, j)}
			}
			if p.URL != "" || p.Data != nil || p.Name != "" {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: text part has non-text fields", index, j)}
			}
		case ContentPartImage:
			if p.Text != "" {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: image part has text fields", index, j)}
			}
			if p.URL != "" || p.Data != nil || p.Name != "" {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: image part has non-image fields", index, j)}
			}
			media, params, err := mime.ParseMediaType(p.MIMEType)
			if err != nil || len(params) != 0 || !supportedImageMIME[strings.ToLower(media)] || media != strings.ToLower(media) {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: MIMEType must be an image type", index, j)}
			}
			if len(p.Image) == 0 {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: image data is empty", index, j)}
			}
			if len(p.Image) > MaxImageBytes {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: image exceeds %d bytes", index, j, MaxImageBytes)}
			}
		case ContentPartDocument:
			if err := validateDocumentPart(p, index, j); err != nil {
				return err
			}
		case ContentPartImageURL:
			if err := validateImageURLPart(p, index, j); err != nil {
				return err
			}
		default:
			return &ConfigError{Msg: fmt.Sprintf("message %d part %d: unknown content part type %q", index, j, p.Type)}
		}
	}
	return nil
}

// maxImageURLBytes is a sanity bound on a remote image URL.
const maxImageURLBytes = 8 << 10

var supportedDocumentMIME = map[string]bool{"application/pdf": true, "text/plain": true}

func validateDocumentPart(p ContentPart, index, j int) error {
	media, params, err := mime.ParseMediaType(p.MIMEType)
	if err != nil || len(params) != 0 || !supportedDocumentMIME[media] || media != strings.ToLower(media) {
		return &ConfigError{Msg: fmt.Sprintf("message %d part %d: document MIMEType must be application/pdf or text/plain", index, j)}
	}
	if p.Text != "" || p.Image != nil || p.URL != "" {
		return &ConfigError{Msg: fmt.Sprintf("message %d part %d: document part has non-document fields", index, j)}
	}
	if len(p.Data) == 0 {
		return &ConfigError{Msg: fmt.Sprintf("message %d part %d: document data is empty", index, j)}
	}
	if len(p.Data) > MaxDocumentBytes {
		return &ConfigError{Msg: fmt.Sprintf("message %d part %d: document exceeds %d bytes", index, j, MaxDocumentBytes)}
	}
	return nil
}

func validateImageURLPart(p ContentPart, index, j int) error {
	if p.Text != "" || p.Image != nil || p.Data != nil || p.Name != "" {
		return &ConfigError{Msg: fmt.Sprintf("message %d part %d: image URL part has non-URL fields", index, j)}
	}
	u, err := url.Parse(p.URL)
	if err != nil || len(p.URL) > maxImageURLBytes || u.Host == "" || (!strings.EqualFold(u.Scheme, "http") && !strings.EqualFold(u.Scheme, "https")) {
		return &ConfigError{Msg: fmt.Sprintf("message %d part %d: image URL must be a valid http(s) URL (at most %d bytes)", index, j, maxImageURLBytes)}
	}
	if p.MIMEType != "" {
		media, params, err := mime.ParseMediaType(p.MIMEType)
		if err != nil || len(params) != 0 || !supportedImageMIME[media] || media != strings.ToLower(media) {
			return &ConfigError{Msg: fmt.Sprintf("message %d part %d: MIMEType must be an image type", index, j)}
		}
	}
	return nil
}

var supportedImageMIME = map[string]bool{
	"image/png": true, "image/jpeg": true, "image/jpg": true,
	"image/gif": true, "image/webp": true,
}

// wireMIME maps an accepted MIME type to the form providers expect on the
// wire. The informal image/jpg alias is accepted at the API boundary (some
// callers derive MIME from file extensions) but must not reach providers:
// Anthropic and Gemini reject it while image/jpeg is universally valid.
func wireMIME(m string) string {
	if m == "image/jpg" {
		return "image/jpeg"
	}
	return m
}

// partsText returns a message's text: Content, or its text parts joined by
// newlines when it carries Parts.
func partsText(m Message) string {
	if len(m.Parts) == 0 {
		return m.Content
	}
	var texts []string
	for _, p := range m.Parts {
		if p.Type == ContentPartText {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// attachmentParts returns a message's image, image-URL and document parts.
func attachmentParts(m Message) []ContentPart {
	var out []ContentPart
	for _, p := range m.Parts {
		if p.Type != ContentPartText {
			out = append(out, p)
		}
	}
	return out
}

const defaultDocumentName = "document.pdf"

func documentName(p ContentPart) string {
	if p.Name == "" {
		return defaultDocumentName
	}
	return p.Name
}

// validateOpenAIFormatParts rejects documents the OpenAI-format wires cannot
// carry. api names the endpoint in the tool-result error.
func validateOpenAIFormatParts(msgs []Message, responses bool) error {
	for i, m := range msgs {
		for j, p := range m.Parts {
			if p.Type != ContentPartDocument {
				continue
			}
			if m.Role == RoleTool {
				if responses {
					return &ConfigError{Msg: "tool results cannot carry documents on the Responses API"}
				}
				return &ConfigError{Msg: "tool results cannot carry documents on OpenAI-format providers"}
			}
			if p.MIMEType != "application/pdf" {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: OpenAI-format documents support application/pdf only", i, j)}
			}
		}
	}
	return nil
}

// imageLikeParts returns inline-image and image-URL parts.
func imageLikeParts(m Message) []ContentPart {
	var out []ContentPart
	for _, p := range m.Parts {
		if p.Type == ContentPartImage || p.Type == ContentPartImageURL {
			out = append(out, p)
		}
	}
	return out
}
