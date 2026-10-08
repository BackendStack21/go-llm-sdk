package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	neturl "net/url"
	"strings"
	"time"
)

// Anthropic Messages API format. Canonical requests translate to the
// top-level "system" field, content blocks, and tool_use/tool_result blocks;
// responses translate back. Extended thinking maps to the canonical
// Thinking/ThinkingBudget fields.

// ── request ──────────────────────────────────────────────────────────────

type anCacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

type anSysBlock struct {
	Type         string          `json:"type"` // "text"
	Text         string          `json:"text"`
	CacheControl *anCacheControl `json:"cache_control,omitempty"`
}

type anBlock struct {
	Type   string         `json:"type"` // "text" | "tool_use" | "tool_result"
	Text   string         `json:"text,omitempty"`
	Source *anImageSource `json:"source,omitempty"`
	// CacheControl is set on user text blocks when Message.Cache is true.
	CacheControl *anCacheControl `json:"cache_control,omitempty"`
	// thinking / redacted_thinking (replayed assistant turns; they go
	// FIRST, in response order, and carry the provider signature or the
	// opaque redacted data)
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result (string content form)
	ToolUseID string `json:"tool_use_id,omitempty"`
	Result    string `json:"content,omitempty"`
}

type anImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type anMessage struct {
	Role    string    `json:"role"` // "user" | "assistant"
	Content []anBlock `json:"content"`
}

type anTool struct {
	Name         string          `json:"name"`
	Description  string          `json:"description,omitempty"`
	InputSchema  json.RawMessage `json:"input_schema,omitempty"`
	CacheControl *anCacheControl `json:"cache_control,omitempty"`
}

type anThinking struct {
	Type         string `json:"type"` // "enabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type anRequest struct {
	Model       string        `json:"model"`
	MaxTokens   int           `json:"max_tokens"`
	Messages    []anMessage   `json:"messages"`
	System      []anSysBlock  `json:"system,omitempty"`
	Tools       []anTool      `json:"tools,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	Stop        []string      `json:"stop_sequences,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Thinking    *anThinking   `json:"thinking,omitempty"`
	ToolChoice  *anToolChoice `json:"tool_choice,omitempty"`
}

const anthropicDefaultMaxTokens = 8192

// anthropicMinThinkingBudget is Anthropic's minimum budget_tokens.
const anthropicMinThinkingBudget = 1024

// anthropicThinkingBudget maps canonical thinking levels to budgets.
// Anthropic requires budget_tokens >= 1024.
func anthropicThinkingBudget(level string, explicit int) (int, bool) {
	var budget int
	switch level {
	case "enabled":
		budget = 5000
	case "low":
		budget = 1024
	case "medium":
		budget = 8192
	case "high", "max":
		budget = 16384
	default: // "", "disabled"
		return 0, false
	}
	if explicit > 0 {
		budget = maxInt(explicit, anthropicMinThinkingBudget)
	}
	return budget, true
}

// anthropicThinkingReplay renders the signed reasoning of a replayed
// assistant turn. ThinkingBlocks (every thinking and redacted_thinking block,
// in response order) supersede the legacy single ReasoningContent +
// ThinkingSignature pair, which is validated here and rendered by the caller.
func anthropicThinkingReplay(m Message, i int) ([]anBlock, error) {
	if len(m.ThinkingBlocks) == 0 {
		if m.ReasoningContent != "" && m.ThinkingSignature == "" {
			return nil, &ConfigError{Msg: fmt.Sprintf("message %d: Anthropic thinking replay requires ThinkingSignature", i)}
		}
		return nil, nil
	}
	blocks := make([]anBlock, 0, len(m.ThinkingBlocks))
	for j, tb := range m.ThinkingBlocks {
		switch {
		case tb.Redacted != "":
			blocks = append(blocks, anBlock{Type: "redacted_thinking", Data: tb.Redacted})
		case tb.Signature == "":
			return nil, &ConfigError{Msg: fmt.Sprintf("message %d thinking block %d: Anthropic thinking replay requires a Signature", i, j)}
		default:
			blocks = append(blocks, anBlock{Type: "thinking", Thinking: tb.Text, Signature: tb.Signature})
		}
	}
	return blocks, nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// buildAnthropicRequest renders the canonical request in Anthropic format.
func buildAnthropicRequest(req *ChatRequest, model string, stream bool) ([]byte, error) {
	out := anRequest{
		Model:     model,
		MaxTokens: req.MaxTokens,
		Stop:      req.Stop,
		Stream:    stream,
	}

	// System: canonical blocks (with cache markers) + in-band system
	// messages become the top-level system field.
	for _, b := range req.System {
		sb := anSysBlock{Type: "text", Text: b.Text}
		if b.Cache {
			sb.CacheControl = &anCacheControl{Type: "ephemeral"}
		}
		out.System = append(out.System, sb)
	}

	if req.Temperature != 0 {
		t := req.Temperature
		if t < 0 {
			t = 0
		}
		out.Temperature = &t
	}
	if req.TopP != 0 {
		p := req.TopP
		if p < 0 {
			p = 0
		}
		out.TopP = &p
	}
	if budget, ok := anthropicThinkingBudget(req.Thinking, req.ThinkingBudget); ok {
		// Anthropic requires max_tokens > budget_tokens, and the budget
		// counts against max_tokens. An unset MaxTokens leaves the usual
		// visible-output room on top of the budget; an explicit MaxTokens
		// is a hard cap, so a preset level is clamped below it and an
		// explicit ThinkingBudget that cannot fit fails fast.
		switch {
		case out.MaxTokens <= 0:
			out.MaxTokens = budget + anthropicDefaultMaxTokens
		case budget >= out.MaxTokens && req.ThinkingBudget > 0:
			return nil, &ConfigError{Msg: fmt.Sprintf("Anthropic ThinkingBudget %d must be below MaxTokens %d", budget, out.MaxTokens)}
		case budget >= out.MaxTokens:
			budget = out.MaxTokens - 1
			if budget < anthropicMinThinkingBudget {
				return nil, &ConfigError{Msg: fmt.Sprintf("Anthropic thinking needs MaxTokens above %d, got %d", anthropicMinThinkingBudget, out.MaxTokens)}
			}
		}
		out.Thinking = &anThinking{Type: "enabled", BudgetTokens: budget}
		// Extended thinking rejects modified sampling; the provider default
		// is the only accepted value.
		out.Temperature, out.TopP = nil, nil
	}
	if out.MaxTokens <= 0 {
		out.MaxTokens = anthropicDefaultMaxTokens
	}
	for _, t := range req.Tools {
		schema := t.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		tool := anTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		}
		if t.Cache {
			tool.CacheControl = &anCacheControl{Type: "ephemeral"}
		}
		out.Tools = append(out.Tools, tool)
	}
	if err := anthropicToolControls(req, &out, out.Thinking != nil); err != nil {
		return nil, err
	}

	// Messages. Hard edges:
	//   - consecutive tool messages merge into ONE user message carrying
	//     multiple tool_result blocks;
	//   - empty assistant turns synthesize a placeholder text block
	//     (Anthropic rejects empty content arrays).
	for i := 0; i < len(req.Messages); i++ {
		m := req.Messages[i]
		switch m.Role {
		case RoleSystem:
			out.System = append(out.System, anSysBlock{Type: "text", Text: m.Content})
		case RoleUser:
			blocks := make([]anBlock, 0, maxInt(1, len(m.Parts)))
			if len(m.Parts) == 0 {
				blocks = append(blocks, anBlock{Type: "text", Text: m.Content})
			} else {
				for _, p := range m.Parts {
					if p.Type == ContentPartImage {
						blocks = append(blocks, anBlock{Type: "image", Source: &anImageSource{Type: "base64", MediaType: wireMIME(p.MIMEType), Data: base64.StdEncoding.EncodeToString(p.Image)}})
					} else {
						blocks = append(blocks, anBlock{Type: "text", Text: p.Text})
					}
				}
			}
			if m.Cache && len(blocks) > 0 && blocks[0].Type == "text" {
				blocks[0].CacheControl = &anCacheControl{Type: "ephemeral"}
			}
			out.Messages = append(out.Messages, anMessage{Role: "user", Content: blocks})
		case RoleAssistant:
			blocks, err := anthropicThinkingReplay(m, i)
			if err != nil {
				return nil, err
			}
			if len(m.ThinkingBlocks) == 0 && m.ReasoningContent != "" {
				// Anthropic requires a replayed thinking block to be the
				// FIRST block and to carry its signature; extended-thinking
				// tool loops are otherwise rejected mid-conversation.
				blocks = append(blocks, anBlock{Type: "thinking", Thinking: m.ReasoningContent, Signature: m.ThinkingSignature})
			}
			if m.Content != "" {
				blocks = append(blocks, anBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				in := json.RawMessage(tc.Arguments)
				if len(in) == 0 {
					in = json.RawMessage("{}")
				}
				blocks = append(blocks, anBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Name,
					Input: in,
				})
			}
			if len(blocks) == 0 {
				blocks = []anBlock{{Type: "text", Text: " "}} // placeholder
			}
			out.Messages = append(out.Messages, anMessage{Role: "assistant", Content: blocks})
		case RoleTool:
			// Merge the run of consecutive tool messages.
			var results []anBlock
			for ; i < len(req.Messages) && req.Messages[i].Role == RoleTool; i++ {
				tm := req.Messages[i]
				results = append(results, anBlock{
					Type:      "tool_result",
					ToolUseID: tm.ToolCallID,
					Result:    tm.Content,
				})
			}
			i-- // outer loop increment
			out.Messages = append(out.Messages, anMessage{Role: "user", Content: results})
		}
	}

	return json.Marshal(out)
}

// ── response ─────────────────────────────────────────────────────────────

type anRespBlock struct {
	Type      string          `json:"type"` // "text" | "thinking" | "tool_use"
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	Data      string          `json:"data"` // redacted_thinking payload
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type anUsage struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	CacheCreationTokens int `json:"cache_creation_input_tokens"`
	CacheReadTokens     int `json:"cache_read_input_tokens"`
}

type anResponse struct {
	Content    []anRespBlock `json:"content"`
	StopReason string        `json:"stop_reason"`
	Usage      anUsage       `json:"usage"`
}

// usageFromAnthropic maps Messages API usage onto canonical Usage.
// Anthropic reports cache volumes exclusively — input_tokens is already
// uncached-only — so PromptTokens is left alone.
func usageFromAnthropic(u anUsage) Usage {
	out := Usage{
		PromptTokens:        u.InputTokens,
		CompletionTokens:    u.OutputTokens,
		CacheCreationTokens: u.CacheCreationTokens,
		CacheReadTokens:     u.CacheReadTokens,
	}
	if u.CacheCreationTokens > 0 || u.CacheReadTokens > 0 {
		out.CacheReported = true
	}
	return out
}

// mapAnthropicStopReason maps stop_reason to canonical values.
func mapAnthropicStopReason(s string) string {
	switch s {
	case "end_turn", "stop_sequence", "":
		if s == "" {
			return ""
		}
		return FinishStop
	case "max_tokens":
		return FinishLength
	case "tool_use":
		return FinishToolCalls
	case "refusal":
		return FinishContentFilter
	default:
		// Provider-specific reasons (pause_turn, model_context_window_exceeded,
		// ...) stay out of the canonical vocabulary.
		return ""
	}
}

// parseAnthropicResponse parses a buffered Messages API response body.
func parseAnthropicResponse(body []byte) (*ChatResult, error) {
	var r anResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("llm: parse response: %w", err)
	}
	if len(r.Content) == 0 && r.StopReason == "" {
		return nil, fmt.Errorf("llm: response has no content blocks")
	}
	res := &ChatResult{FinishReason: mapAnthropicStopReason(r.StopReason)}
	var content, thinking []string
	for _, b := range r.Content {
		switch b.Type {
		case "text":
			content = append(content, b.Text)
		case "thinking":
			thinking = append(thinking, b.Thinking)
			res.ThinkingBlocks = append(res.ThinkingBlocks, ThinkingBlock{Text: b.Thinking, Signature: b.Signature})
			if b.Signature != "" {
				// Legacy single-signature field: the last block's.
				res.ThinkingSignature = b.Signature
			}
		case "redacted_thinking":
			// Opaque, encrypted reasoning: never surfaced as text, but it
			// must be replayed verbatim for tool loops to stay valid.
			res.ThinkingBlocks = append(res.ThinkingBlocks, ThinkingBlock{Redacted: b.Data})
		case "tool_use":
			args := string(b.Input)
			if args == "" {
				args = "{}"
			}
			res.ToolCalls = append(res.ToolCalls, ToolCall{ID: b.ID, Name: b.Name, Arguments: args})
		}
	}
	res.Content = strings.Join(content, "")
	res.ReasoningContent = strings.Join(thinking, "")
	res.Usage = usageFromAnthropic(r.Usage)
	return res, nil
}

// ── streaming ────────────────────────────────────────────────────────────

type anStreamEvent struct {
	Type string `json:"type"`
	// message_start
	Message struct {
		Usage anUsage `json:"usage"`
	} `json:"message"`
	// content_block_start / content_block_stop
	Index        int         `json:"index"`
	ContentBlock anRespBlock `json:"content_block"`
	// content_block_delta AND message_delta (stop_reason): the wire keeps
	// delta and usage as top-level siblings, not nested under a
	// "message_delta" key.
	Delta struct {
		Type        string `json:"type"` // "text_delta" | "thinking_delta" | "signature_delta" | "input_json_delta"
		StopReason  string `json:"stop_reason"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
	// message_delta usage (top-level sibling of delta)
	Usage anUsage `json:"usage"` // message_delta: cumulative counts
	// error
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// mapAnthropicStreamEvent folds one Anthropic SSE payload into acc.
func mapAnthropicStreamEvent(data []byte, acc *streamAccum) ([]Delta, bool, error) {
	var ev anStreamEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, false, fmt.Errorf("llm: parse stream event: %w", err)
	}
	var deltas []Delta
	switch ev.Type {
	case "message_start":
		acc.usage = usageFromAnthropic(ev.Message.Usage)
	case "content_block_start":
		switch ev.ContentBlock.Type {
		case "tool_use":
			if acc.jsonTool != "" && acc.jsonBlock < 0 && ev.ContentBlock.Name == acc.jsonTool {
				// JSON-mode tool: its input streams as content.
				acc.jsonBlock = ev.Index
				break
			}
			c := acc.call(ev.Index)
			c.id, c.name = ev.ContentBlock.ID, ev.ContentBlock.Name
			deltas = append(deltas, Delta{
				Kind:      DeltaToolArgs,
				ToolIndex: c.pos,
				ToolID:    c.id,
				ToolName:  c.name,
			})
		case "thinking":
			acc.block(ev.Index)
		case "redacted_thinking":
			acc.block(ev.Index).redacted = ev.ContentBlock.Data
		}
	case "content_block_delta":
		switch ev.Delta.Type {
		case "text_delta":
			acc.content.WriteString(ev.Delta.Text)
			deltas = append(deltas, Delta{Kind: DeltaContent, Text: ev.Delta.Text})
		case "thinking_delta":
			acc.reasoning.WriteString(ev.Delta.Thinking)
			acc.block(ev.Index).text.WriteString(ev.Delta.Thinking)
			deltas = append(deltas, Delta{Kind: DeltaReasoning, Text: ev.Delta.Thinking})
		case "signature_delta":
			// Signatures belong to their own block; concatenating them
			// across blocks would produce an invalid signature.
			b := acc.block(ev.Index)
			b.sig += ev.Delta.Signature
			acc.thinkingSignature = b.sig
		case "input_json_delta":
			if acc.jsonTool != "" && ev.Index == acc.jsonBlock {
				acc.content.WriteString(ev.Delta.PartialJSON)
				deltas = append(deltas, Delta{Kind: DeltaContent, Text: ev.Delta.PartialJSON})
				break
			}
			c := acc.call(ev.Index)
			c.args.WriteString(ev.Delta.PartialJSON)
			deltas = append(deltas, Delta{
				Kind:      DeltaToolArgs,
				Text:      ev.Delta.PartialJSON,
				ToolIndex: c.pos,
				ToolID:    c.id,
				ToolName:  c.name,
			})
		}
	case "message_delta":
		if ev.Delta.StopReason != "" {
			acc.finishReason = mapAnthropicStopReason(ev.Delta.StopReason)
			acc.sawFinish = true
			if acc.jsonBlock >= 0 && acc.finishReason == FinishToolCalls && len(acc.calls) == 0 {
				acc.finishReason = FinishStop // the JSON tool is the answer, not a tool turn
			}
		}
		// message_delta usage is cumulative. Output is always present; input
		// and cache volumes may only arrive here, so non-zero values win.
		acc.usage.CompletionTokens = ev.Usage.OutputTokens
		if ev.Usage.InputTokens > 0 {
			acc.usage.PromptTokens = ev.Usage.InputTokens
		}
		if ev.Usage.CacheReadTokens > 0 {
			acc.usage.CacheReadTokens = ev.Usage.CacheReadTokens
			acc.usage.CacheReported = true
		}
		if ev.Usage.CacheCreationTokens > 0 {
			acc.usage.CacheCreationTokens = ev.Usage.CacheCreationTokens
			acc.usage.CacheReported = true
		}
	case "message_stop":
		return deltas, true, nil
	case "error":
		msg := "provider stream error"
		if ev.Error != nil && ev.Error.Message != "" {
			msg = ev.Error.Message
		}
		return deltas, false, fmt.Errorf("llm: %s", msg)
	case "ping", "content_block_stop":
		// keepalive / no-op
	}
	return deltas, false, nil
}

// ── model discovery ──────────────────────────────────────────────────────

type anModelEntry struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name"`
	CreatedAt   string `json:"created_at"`
}

type anModelsPage struct {
	Data    []anModelEntry `json:"data"`
	HasMore bool           `json:"has_more"`
	LastID  string         `json:"last_id"`
}

// listModelsAnthropic pages through GET /v1/models.
func listModelsAnthropic(ctx context.Context, pc *providerClient) ([]Model, error) {
	var out []Model
	pageID := ""
	for page := 0; page < maxModelPages; page++ {
		url := pc.base + "/v1/models?limit=100"
		if pageID != "" {
			url += "&after_id=" + neturl.QueryEscape(pageID)
		}
		data, _, err := pc.get(ctx, url)
		if err != nil {
			return nil, err
		}
		var p anModelsPage
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, terminal(fmt.Errorf("llm: parse models response: %w", err))
		}
		for _, m := range p.Data {
			mm := Model{ID: m.ID, DisplayName: m.DisplayName}
			if t, err := time.Parse(time.RFC3339, m.CreatedAt); err == nil {
				mm.CreatedAt = t
			}
			out = append(out, mm)
		}
		if !p.HasMore || p.LastID == "" {
			return out, nil
		}
		pageID = p.LastID
	}
	return nil, fmt.Errorf("%w (%d pages)", ErrModelListTruncated, maxModelPages)
}
