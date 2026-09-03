package llm

import (
	"context"
	"encoding/json"
	"fmt"
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
	Type string `json:"type"` // "text" | "tool_use" | "tool_result"
	Text string `json:"text,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result (string content form)
	ToolUseID string `json:"tool_use_id,omitempty"`
	Result    string `json:"content,omitempty"`
}

type anMessage struct {
	Role    string    `json:"role"` // "user" | "assistant"
	Content []anBlock `json:"content"`
}

type anTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type anThinking struct {
	Type         string `json:"type"` // "enabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type anRequest struct {
	Model       string       `json:"model"`
	MaxTokens   int          `json:"max_tokens"`
	Messages    []anMessage  `json:"messages"`
	System      []anSysBlock `json:"system,omitempty"`
	Tools       []anTool     `json:"tools,omitempty"`
	Temperature *float64     `json:"temperature,omitempty"`
	Stream      bool         `json:"stream,omitempty"`
	Thinking    *anThinking  `json:"thinking,omitempty"`
}

const anthropicDefaultMaxTokens = 8192

// anthropicThinkingBudget maps canonical thinking levels to budgets.
// Anthropic requires budget_tokens >= 1024.
func anthropicThinkingBudget(level string, explicit int) (int, bool) {
	switch level {
	case "enabled":
		if explicit > 0 {
			return maxInt(explicit, 1024), true
		}
		return 5000, true
	case "low":
		return 1024, true
	case "medium":
		return 8192, true
	case "high":
		return 16384, true
	default: // "", "disabled"
		return 0, false
	}
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
	if out.MaxTokens <= 0 {
		out.MaxTokens = anthropicDefaultMaxTokens
	}
	if budget, ok := anthropicThinkingBudget(req.Thinking, req.ThinkingBudget); ok {
		out.Thinking = &anThinking{Type: "enabled", BudgetTokens: budget}
	}
	for _, t := range req.Tools {
		schema := t.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		out.Tools = append(out.Tools, anTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		})
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
			out.Messages = append(out.Messages, anMessage{
				Role:    "user",
				Content: []anBlock{{Type: "text", Text: m.Content}},
			})
		case RoleAssistant:
			var blocks []anBlock
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
	Type     string          `json:"type"` // "text" | "thinking" | "tool_use"
	Text     string          `json:"text"`
	Thinking string          `json:"thinking"`
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Input    json.RawMessage `json:"input"`
}

type anResponse struct {
	Content    []anRespBlock `json:"content"`
	StopReason string        `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
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
		return s
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
	res.Usage = Usage{
		PromptTokens:     r.Usage.InputTokens,
		CompletionTokens: r.Usage.OutputTokens,
	}
	return res, nil
}

// ── streaming ────────────────────────────────────────────────────────────

type anStreamEvent struct {
	Type string `json:"type"`
	// message_start
	Message struct {
		Usage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message"`
	// content_block_start / content_block_stop
	Index        int         `json:"index"`
	ContentBlock anRespBlock `json:"content_block"`
	// content_block_delta AND message_delta (stop_reason): the wire keeps
	// delta and usage as top-level siblings, not nested under a
	// "message_delta" key.
	Delta struct {
		Type        string `json:"type"` // "text_delta" | "thinking_delta" | "input_json_delta"
		StopReason  string `json:"stop_reason"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
	// message_delta usage (top-level sibling of delta)
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
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
		acc.usage.PromptTokens = ev.Message.Usage.InputTokens
	case "content_block_start":
		if ev.ContentBlock.Type == "tool_use" {
			c := acc.call(ev.Index)
			c.id, c.name = ev.ContentBlock.ID, ev.ContentBlock.Name
			deltas = append(deltas, Delta{
				Kind:      DeltaToolArgs,
				ToolIndex: ev.Index,
				ToolID:    c.id,
				ToolName:  c.name,
			})
		}
	case "content_block_delta":
		switch ev.Delta.Type {
		case "text_delta":
			acc.content.WriteString(ev.Delta.Text)
			deltas = append(deltas, Delta{Kind: DeltaContent, Text: ev.Delta.Text})
		case "thinking_delta":
			acc.reasoning.WriteString(ev.Delta.Thinking)
			deltas = append(deltas, Delta{Kind: DeltaReasoning, Text: ev.Delta.Thinking})
		case "input_json_delta":
			c := acc.call(ev.Index)
			c.args.WriteString(ev.Delta.PartialJSON)
			deltas = append(deltas, Delta{
				Kind:      DeltaToolArgs,
				Text:      ev.Delta.PartialJSON,
				ToolIndex: ev.Index,
				ToolID:    c.id,
				ToolName:  c.name,
			})
		}
	case "message_delta":
		if ev.Delta.StopReason != "" {
			acc.finishReason = mapAnthropicStopReason(ev.Delta.StopReason)
		}
		acc.usage.CompletionTokens = ev.Usage.OutputTokens
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
	for page := 0; page < 10; page++ {
		url := pc.base + "/v1/models?limit=100"
		if pageID != "" {
			url += "&after_id=" + pageID
		}
		data, _, err := pc.get(ctx, url)
		if err != nil {
			return nil, err
		}
		var p anModelsPage
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, fmt.Errorf("llm: parse models response: %w", err)
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
	return out, nil
}
