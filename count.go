package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// ErrUnsupported marks an operation the provider format cannot perform.
var ErrUnsupported = errors.New("llm: operation not supported")

// UnsupportedError reports an unsupported operation; it unwraps to
// ErrUnsupported.
type UnsupportedError struct{ Op, Provider, Reason string }

func (e *UnsupportedError) Error() string {
	return fmt.Sprintf("llm: %s unsupported on %s: %s", e.Op, e.Provider, e.Reason)
}

func (e *UnsupportedError) Unwrap() error { return ErrUnsupported }

// CountTokens reports the input token count of req using the provider's
// counting endpoint (Anthropic /v1/messages/count_tokens, Gemini
// :countTokens). OpenAI-format providers, including the Responses API, have
// none: *UnsupportedError (errors.Is ErrUnsupported), with no network call
// and never an estimate. Request validation matches Call; Extra and
// sampling parameters are ignored. Retries, the whole-call timeout, headers
// and the attempt observer (operation "count_tokens") behave as for chat.
func (c *ChatClient) CountTokens(ctx context.Context, req *ChatRequest) (int, error) {
	return c.pc.countTokens(ctx, req, c.model)
}

func (pc *providerClient) countTokens(ctx context.Context, req *ChatRequest, model string) (int, error) {
	model, err := validateChatRequest(req, model)
	if err != nil {
		return 0, err
	}
	if pc.cfg.Format != FormatAnthropic && pc.cfg.Format != FormatGemini {
		return 0, &UnsupportedError{Op: "CountTokens", Provider: pc.cfg.ID, Reason: "no token counting endpoint"}
	}
	ctx, herr := withReqHeaders(ctx, chatReqHeaders(req))
	if herr != nil {
		return 0, herr
	}
	body, url, err := pc.buildCountRequest(req, model)
	if err != nil {
		return 0, err
	}
	ctx = pc.withAttempts(ctx, "count_tokens", model)
	ctx, cancel := context.WithTimeout(ctx, pc.requestTimeout())
	defer cancel()
	var n int
	err = pc.withRetry(ctx, func() (time.Duration, error) {
		data, _, ra, err := pc.post(ctx, pc.buffered(), url, body)
		if err != nil {
			return ra, err
		}
		v, perr := pc.parseCountResponse(data)
		if perr != nil {
			return 0, terminal(&APIError{Provider: pc.cfg.ID, Status: http.StatusOK, Message: perr.Error()})
		}
		n = v
		return 0, nil
	}, nil)
	if err != nil {
		return 0, err
	}
	return n, nil
}

func (pc *providerClient) buildCountRequest(req *ChatRequest, model string) ([]byte, string, error) {
	if pc.cfg.Format == FormatAnthropic {
		full, err := buildAnthropicRequestQ(req, model, false, pc.cfg.Quirks)
		if err != nil {
			return nil, "", err
		}
		b, err := pickKeys(full, "model", "messages", "system", "tools", "tool_choice", "thinking")
		return b, pc.base + "/v1/messages/count_tokens", err
	}
	full, err := buildGeminiRequest(req, model, false)
	if err != nil {
		return nil, "", err
	}
	inner, err := pickKeys(full, "contents", "systemInstruction", "tools")
	if err != nil {
		return nil, "", err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(inner, &m); err != nil {
		return nil, "", err
	}
	m["model"], _ = json.Marshal("models/" + model)
	b, err := json.Marshal(map[string]any{"generateContentRequest": m})
	return b, fmt.Sprintf("%s/v1beta/models/%s:countTokens", pc.base, geminiModelPath(model)), err
}

// pickKeys keeps only the listed top-level keys of a JSON object.
func pickKeys(body []byte, keys ...string) ([]byte, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	out := make(map[string]json.RawMessage, len(keys))
	for _, k := range keys {
		if v, ok := m[k]; ok {
			out[k] = v
		}
	}
	return json.Marshal(out)
}

func (pc *providerClient) parseCountResponse(data []byte) (int, error) {
	var r struct {
		InputTokens *int `json:"input_tokens"`
		TotalTokens *int `json:"totalTokens"`
	}
	if err := json.Unmarshal(data, &r); err != nil {
		return 0, fmt.Errorf("llm: parse token count: %w", err)
	}
	p := r.InputTokens
	if pc.cfg.Format == FormatGemini {
		p = r.TotalTokens
	}
	if p == nil || *p < 0 {
		return 0, errors.New("llm: token count missing or invalid in response")
	}
	return *p, nil
}
