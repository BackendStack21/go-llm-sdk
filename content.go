package llm

import (
	"fmt"
	"mime"
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
			total += len(p.Image)
			if total > MaxRequestImageBytes {
				return &ConfigError{Msg: fmt.Sprintf("message %d: request inline images exceed %d bytes", i, MaxRequestImageBytes)}
			}
		}
	}
	return nil
}

func validateMessageContent(m Message, index int) error {
	if len(m.Parts) == 0 {
		return nil
	}
	if m.Content != "" {
		return &ConfigError{Msg: fmt.Sprintf("message %d: Content and Parts cannot both be set", index)}
	}
	if m.Role != RoleUser {
		return &ConfigError{Msg: fmt.Sprintf("message %d: content parts require user role", index)}
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
		case ContentPartImage:
			if p.Text != "" {
				return &ConfigError{Msg: fmt.Sprintf("message %d part %d: image part has text fields", index, j)}
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
		default:
			return &ConfigError{Msg: fmt.Sprintf("message %d part %d: unknown content part type %q", index, j, p.Type)}
		}
	}
	return nil
}

var supportedImageMIME = map[string]bool{
	"image/png": true, "image/jpeg": true,
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
