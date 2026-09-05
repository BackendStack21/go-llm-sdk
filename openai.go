package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// OpenAI chat-completions format: the canonical passthrough. DeepSeek, Z.ai
// (GLM) and Kimi ride this format; quirk flags on ProviderConfig decide the
// thinking-field shape (reasoning_effort vs thinking object vs neither).

// ── request ──────────────────────────────────────────────────────────────

type oaThinking struct {
	Type string `json:"type"` // "enabled" | "disabled"
}

type oaFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type oaToolCall struct {
	ID       string     `json:"id"`
	Type     string     `json:"type"`
	Function oaFunction `json:"function"`
}

type oaMessage struct {
	Role             string       `json:"role"`
	Content          *string      `json:"content"` // nil keeps JSON null for tool calls
	ReasoningContent string       `json:"reasoning_content,omitempty"`
	ToolCalls        []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string       `json:"tool_call_id,omitempty"`
}

type oaToolDef struct {
	Type     string          `json:"type"` // "function"
	Function json.RawMessage `json:"function"`
}

type oaToolFn struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type oaStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaRequest struct {
	Model               string           `json:"model"`
	Messages            []oaMessage      `json:"messages"`
	Tools               []oaToolDef      `json:"tools,omitempty"`
	MaxTokens           int              `json:"max_tokens,omitempty"`
	MaxCompletionTokens int              `json:"max_completion_tokens,omitempty"`
	Temperature         *float64         `json:"temperature,omitempty"`
	TopP                *float64         `json:"top_p,omitempty"`
	Stop                []string         `json:"stop,omitempty"`
	Stream              bool             `json:"stream,omitempty"`
	StreamOptions       *oaStreamOptions `json:"stream_options,omitempty"`
	ReasoningEffort     string           `json:"reasoning_effort,omitempty"`
	Thinking            *oaThinking      `json:"thinking,omitempty"`
}

// buildOpenAIRequest renders the canonical request in OpenAI format.
func buildOpenAIRequest(cfg ProviderConfig, req *ChatRequest, model string, stream, includeStreamOptions bool) oaRequest {
	q := cfg.Quirks
	out := oaRequest{
		Model:  model,
		Stop:   req.Stop,
		Stream: stream,
	}
	// OpenAI o-series/gpt-5 models reject max_tokens in favor of
	// max_completion_tokens; everyone else keeps the classic parameter.
	if modelUsesCompletionTokens(model) {
		out.MaxCompletionTokens = req.MaxTokens
	} else {
		out.MaxTokens = req.MaxTokens
	}

	// One OpenAI system message per SystemBlock and per in-band system
	// role. Concatenating would collapse prompt-tiering (stable base +
	// volatile memory) and bust prefix cache on every memory refresh.
	msgs := make([]oaMessage, 0, len(req.Messages)+len(req.System)+1)
	appendSystem := func(text string) {
		if t := strings.TrimRight(text, "\n"); t != "" {
			s := t
			msgs = append(msgs, oaMessage{Role: "system", Content: &s})
		}
	}
	for _, b := range req.System {
		appendSystem(b.Text)
	}
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			appendSystem(m.Content)
		}
	}
	for _, m := range req.Messages {
		switch m.Role {
		case RoleSystem:
			// already folded into the system message
			continue
		case RoleTool:
			c := m.Content
			msgs = append(msgs, oaMessage{Role: "tool", Content: &c, ToolCallID: m.ToolCallID})
		case RoleAssistant:
			om := oaMessage{Role: "assistant"}
			c := m.Content
			om.Content = &c
			om.ReasoningContent = m.ReasoningContent
			for _, tc := range m.ToolCalls {
				om.ToolCalls = append(om.ToolCalls, oaToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: oaFunction{
						Name:      tc.Name,
						Arguments: tc.Arguments,
					},
				})
			}
			msgs = append(msgs, om)
		default: // user
			c := m.Content
			msgs = append(msgs, oaMessage{Role: "user", Content: &c})
		}
	}
	out.Messages = msgs

	for _, t := range req.Tools {
		fn, _ := json.Marshal(oaToolFn(t))
		out.Tools = append(out.Tools, oaToolDef{Type: "function", Function: fn})
	}

	// Temperature: 0 → omit (provider default); negative → explicit 0.
	// Models that only accept the default temperature never get the field.
	if req.Temperature != 0 && !modelForbidsTemperature(model) {
		t := req.Temperature
		if t < 0 {
			t = 0
		}
		out.Temperature = &t
	}
	if req.TopP != 0 && !modelForbidsTemperature(model) {
		p := req.TopP
		if p < 0 {
			p = 0
		}
		out.TopP = &p
	}

	if stream && includeStreamOptions {
		out.StreamOptions = &oaStreamOptions{IncludeUsage: true}
	}

	switch {
	case q.ThinkingObject:
		// Anthropic-style thinking object (DeepSeek, GLM).
		switch req.Thinking {
		case "enabled":
			out.Thinking = &oaThinking{Type: "enabled"}
		case "disabled":
			if q.forceThinkingModel(model) {
				// GLM-5.3+ rejects disabled outright; the documented
				// migration is enabled + reasoning_effort low.
				out.Thinking = &oaThinking{Type: "enabled"}
				if q.ReasoningEffort {
					out.ReasoningEffort = "low"
				}
			} else {
				out.Thinking = &oaThinking{Type: "disabled"}
			}
		case "low", "medium", "high", "max":
			out.Thinking = &oaThinking{Type: "enabled"}
			if q.ReasoningEffort {
				// GLM has no "medium" effort level; odek's medium maps to high.
				if req.Thinking == "medium" {
					out.ReasoningEffort = "high"
				} else {
					out.ReasoningEffort = req.Thinking
				}
			}
		}
	case q.ReasoningEffort:
		switch req.Thinking {
		case "enabled":
			out.ReasoningEffort = "medium"
		case "low", "medium", "high":
			out.ReasoningEffort = req.Thinking
			// "disabled" and "" → omit (provider default)
		}
	default:
		// Provider accepts neither field (Kimi, plain gateways).
	}

	return out
}

// reasoningEffortNonePatched returns a copy of the request with
// reasoning_effort pinned to "none" (learn-once: the provider rejected
// reasoning_effort combined with function tools).
func reasoningEffortNonePatched(r oaRequest) oaRequest {
	r.ReasoningEffort = "none"
	if r.Thinking != nil {
		r.Thinking = nil // effort and thinking object never combine here
	}
	return r
}

// ── response ─────────────────────────────────────────────────────────────

type oaRespToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaRespMessage struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ReasoningContent string           `json:"reasoning_content"`
	ToolCalls        []oaRespToolCall `json:"tool_calls"`
}

type oaRespChoice struct {
	Message      oaRespMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type oaUsageDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type oaPromptDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type oaRespUsage struct {
	PromptTokens            int              `json:"prompt_tokens"`
	CompletionTokens        int              `json:"completion_tokens"`
	CompletionTokensDetails oaUsageDetails   `json:"completion_tokens_details"`
	CacheCreationTokens     int              `json:"cache_creation_input_tokens"`
	CacheReadTokens         int              `json:"cache_read_input_tokens"`
	PromptTokensDetails     *oaPromptDetails `json:"prompt_tokens_details"`
	PromptCacheHitTokens    int              `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens   int              `json:"prompt_cache_miss_tokens"`
}

// usageFromOpenAI maps a chat-completions usage object onto canonical
// Usage with odek applyUsage exclusive normalization: OpenAI
// cached_tokens and DeepSeek hit/miss are subsets of prompt_tokens and
// are subtracted so PromptTokens is uncached-only; Anthropic-named
// cache fields are exclusive and are not subtracted. Guards keep
// hostile payloads from driving PromptTokens negative.
func usageFromOpenAI(u *oaRespUsage) Usage {
	if u == nil {
		return Usage{}
	}
	out := Usage{
		PromptTokens:        u.PromptTokens,
		CompletionTokens:    u.CompletionTokens,
		ReasoningTokens:     u.CompletionTokensDetails.ReasoningTokens,
		CacheCreationTokens: u.CacheCreationTokens,
		CacheReadTokens:     u.CacheReadTokens,
	}
	if u.PromptTokensDetails != nil {
		out.CachedTokens = u.PromptTokensDetails.CachedTokens
		out.CacheReported = true
	}
	if u.CacheCreationTokens > 0 || u.CacheReadTokens > 0 {
		out.CacheReported = true
	}
	// DeepSeek native fields: a hit is prompt content read from cache; a
	// miss is newly processed content that DeepSeek then caches for
	// future requests, i.e. a cache write.
	if u.PromptCacheHitTokens > 0 || u.PromptCacheMissTokens > 0 {
		out.CacheReadTokens += u.PromptCacheHitTokens
		out.CacheCreationTokens += u.PromptCacheMissTokens
		out.CacheReported = true
	}
	if u.PromptTokensDetails != nil && u.PromptTokensDetails.CachedTokens > 0 &&
		u.PromptTokensDetails.CachedTokens <= out.PromptTokens {
		out.PromptTokens -= u.PromptTokensDetails.CachedTokens
		out.CacheReadTokens += u.PromptTokensDetails.CachedTokens
	}
	if total := u.PromptCacheHitTokens + u.PromptCacheMissTokens; total > 0 && total <= out.PromptTokens {
		out.PromptTokens -= total
	}
	return out
}

type oaResponse struct {
	Choices []oaRespChoice `json:"choices"`
	Usage   *oaRespUsage   `json:"usage"`
	Error   *oaErrorBody   `json:"error"`
}

type oaErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

// mapOpenAIFinishReason maps OpenAI finish_reason to canonical values.
func mapOpenAIFinishReason(s string) string {
	switch s {
	case "stop":
		return FinishStop
	case "length":
		return FinishLength
	case "tool_calls", "function_call":
		return FinishToolCalls
	case "content_filter":
		return FinishContentFilter
	case "":
		return ""
	default:
		// Non-canonical reasons never leak into the public vocabulary.
		return ""
	}
}

// parseOpenAIResponse parses a buffered chat-completions response body.
func parseOpenAIResponse(body []byte) (*ChatResult, error) {
	var r oaResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("llm: parse response: %w", err)
	}
	if r.Error != nil && r.Error.Message != "" {
		return nil, fmt.Errorf("llm: provider error: %s", r.Error.Message)
	}
	if len(r.Choices) == 0 {
		return nil, fmt.Errorf("llm: response has no choices")
	}
	ch := r.Choices[0]
	res := &ChatResult{
		Content:          ch.Message.Content,
		ReasoningContent: ch.Message.ReasoningContent,
		FinishReason:     mapOpenAIFinishReason(ch.FinishReason),
	}
	for _, tc := range ch.Message.ToolCalls {
		res.ToolCalls = append(res.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}
	if r.Usage != nil {
		res.Usage = usageFromOpenAI(r.Usage)
	}
	return res, nil
}

// ── streaming ────────────────────────────────────────────────────────────

type oaStreamDelta struct {
	Content          string             `json:"content"`
	ReasoningContent string             `json:"reasoning_content"`
	ToolCalls        []oaStreamToolCall `json:"tool_calls"`
}

type oaStreamToolCall struct {
	Index    *int   `json:"index"` // OpenAI sends it; default 0 when absent
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaStreamChoice struct {
	Delta        oaStreamDelta `json:"delta"`
	FinishReason string        `json:"finish_reason"`
}

type oaStreamChunk struct {
	Choices []oaStreamChoice `json:"choices"`
	Usage   *oaRespUsage     `json:"usage"`
}

// mapOpenAIStreamEvent folds one SSE data payload into the accumulator and
// returns the canonical deltas to emit. done reports a terminal event.
func mapOpenAIStreamEvent(data []byte, acc *streamAccum) (deltas []Delta, done bool, err error) {
	var c oaStreamChunk
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, false, fmt.Errorf("llm: parse stream chunk: %w", err)
	}
	if c.Usage != nil {
		acc.usage = usageFromOpenAI(c.Usage)
	}
	for _, ch := range c.Choices {
		d := ch.Delta
		if d.ReasoningContent != "" {
			acc.reasoning.WriteString(d.ReasoningContent)
			deltas = append(deltas, Delta{Kind: DeltaReasoning, Text: d.ReasoningContent})
		}
		if d.Content != "" {
			acc.content.WriteString(d.Content)
			deltas = append(deltas, Delta{Kind: DeltaContent, Text: d.Content})
		}
		for _, tc := range d.ToolCalls {
			idx := 0
			if tc.Index != nil {
				idx = *tc.Index
			}
			call := acc.call(idx)
			if tc.ID != "" {
				call.id = tc.ID
			}
			if tc.Function.Name != "" {
				call.name = tc.Function.Name
			}
			accDelta := Delta{Kind: DeltaToolArgs, Text: tc.Function.Arguments, ToolIndex: idx, ToolID: call.id, ToolName: call.name}
			call.args.WriteString(tc.Function.Arguments)
			deltas = append(deltas, accDelta)
		}
		if ch.FinishReason != "" {
			acc.finishReason = mapOpenAIFinishReason(ch.FinishReason)
		}
	}
	return deltas, false, nil
}
