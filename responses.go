package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// OpenAI Responses API (/v1/responses). GPT-5.6 (and later GPT-5.4/5.5 with
// an explicit effort) rejects function tools combined with a non-none
// reasoning_effort on /v1/chat/completions. The documented path is this
// endpoint, which also returns a reasoning summary the Chat Completions
// path never exposes.

// ── request ──────────────────────────────────────────────────────────────

type rsReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type rsTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type rsEasyMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type rsReasoningItem struct {
	Type             string          `json:"type"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`
	Summary          []rsSummaryText `json:"summary"`
}

type rsSummaryText struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type rsFunctionCallItem struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type rsFunctionOutputItem struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

type rsRequest struct {
	Model           string       `json:"model"`
	Instructions    string       `json:"instructions,omitempty"`
	Input           []any        `json:"input"`
	Tools           []rsTool     `json:"tools,omitempty"`
	MaxOutputTokens int          `json:"max_output_tokens,omitempty"`
	Temperature     *float64     `json:"temperature,omitempty"`
	TopP            *float64     `json:"top_p,omitempty"`
	Stream          bool         `json:"stream,omitempty"`
	Store           *bool        `json:"store,omitempty"`
	Include         []string     `json:"include,omitempty"`
	Reasoning       *rsReasoning `json:"reasoning,omitempty"`
}

func boolPtr(v bool) *bool { return &v }

// chatCompletionsRejectsReasoningWithTools reports models whose Chat
// Completions endpoint returns 400 for function tools plus any non-none
// reasoning effort (including the provider default of medium).
func chatCompletionsRejectsReasoningWithTools(model string) bool {
	m := strings.ToLower(model)
	return strings.HasPrefix(m, "gpt-5.6") ||
		strings.HasPrefix(m, "gpt-5.7") ||
		strings.HasPrefix(m, "gpt-6")
}

func chatCompletionsRejectsExplicitReasoningWithTools(model string) bool {
	if chatCompletionsRejectsReasoningWithTools(model) {
		return true
	}
	m := strings.ToLower(model)
	return strings.HasPrefix(m, "gpt-5.4") || strings.HasPrefix(m, "gpt-5.5")
}

// useResponsesAPI decides whether this OpenAI-format call must go to
// POST /responses instead of /chat/completions.
func useResponsesAPI(learn *learnOnce, format Format, model string, req *ChatRequest) bool {
	if format != FormatOpenAI || req == nil || len(req.Tools) == 0 {
		return false
	}
	if learn != nil && learn.forceResponses.Load() {
		return true
	}
	if learn != nil && learn.forceNoneEffort.Load() {
		return false
	}
	if req.Thinking == "disabled" {
		return false
	}
	if chatCompletionsRejectsReasoningWithTools(model) {
		return true
	}
	return req.Thinking != "" && chatCompletionsRejectsExplicitReasoningWithTools(model)
}

func isResponsesURL(url string) bool {
	return strings.HasSuffix(strings.TrimRight(url, "/"), "/responses")
}

func responsesEffort(thinking string) string {
	switch thinking {
	case "enabled":
		return "medium"
	case "low", "medium", "high", "max":
		return thinking
	case "disabled":
		return "none"
	default:
		return "medium"
	}
}

// buildResponsesRequest renders a ChatRequest as a Responses API body.
func buildResponsesRequest(req *ChatRequest, model string, stream bool) rsRequest {
	instructions, input := buildResponsesInput(req)
	out := rsRequest{
		Model:        model,
		Instructions: instructions,
		Input:        input,
		Stream:       stream,
		Store:        boolPtr(false),
		Include:      []string{"reasoning.encrypted_content"},
	}
	if req.MaxTokens > 0 {
		out.MaxOutputTokens = req.MaxTokens
	}
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
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, rsTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}
	effort := responsesEffort(req.Thinking)
	rsn := &rsReasoning{Effort: effort}
	if effort != "none" {
		rsn.Summary = "auto"
	}
	out.Reasoning = rsn
	return out
}

func buildResponsesInput(req *ChatRequest) (instructions string, input []any) {
	var sys []string
	appendSys := func(text string) {
		if t := strings.TrimRight(text, "\n"); t != "" {
			sys = append(sys, t)
		}
	}
	for _, b := range req.System {
		appendSys(b.Text)
	}
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			appendSys(m.Content)
		}
	}
	instructions = strings.Join(sys, "\n\n")

	for _, m := range req.Messages {
		switch m.Role {
		case RoleSystem:
			continue
		case RoleUser:
			input = append(input, rsEasyMessage{Role: "user", Content: m.Content})
		case RoleAssistant:
			if m.ThinkingSignature != "" {
				sum := make([]rsSummaryText, 0)
				if m.ReasoningContent != "" {
					sum = append(sum, rsSummaryText{Type: "summary_text", Text: m.ReasoningContent})
				}
				input = append(input, rsReasoningItem{
					Type:             "reasoning",
					EncryptedContent: m.ThinkingSignature,
					Summary:          sum,
				})
			}
			for _, tc := range m.ToolCalls {
				input = append(input, rsFunctionCallItem{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Name,
					Arguments: tc.Arguments,
				})
			}
			if m.Content != "" {
				input = append(input, rsEasyMessage{Role: "assistant", Content: m.Content})
			}
		case RoleTool:
			input = append(input, rsFunctionOutputItem{
				Type:   "function_call_output",
				CallID: m.ToolCallID,
				Output: m.Content,
			})
		}
	}
	if input == nil {
		input = []any{}
	}
	return instructions, input
}

// ── response ─────────────────────────────────────────────────────────────

type rsContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type rsOutputItem struct {
	Type             string          `json:"type"`
	ID               string          `json:"id"`
	Role             string          `json:"role"`
	CallID           string          `json:"call_id"`
	Name             string          `json:"name"`
	Arguments        string          `json:"arguments"`
	EncryptedContent string          `json:"encrypted_content"`
	Summary          []rsSummaryText `json:"summary"`
	Content          []rsContentPart `json:"content"`
}

type rsUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

type rsErrorBody struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

type rsResponse struct {
	Status            string `json:"status"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Output []rsOutputItem `json:"output"`
	Usage  *rsUsage       `json:"usage"`
	Error  *rsErrorBody   `json:"error"`
}

func usageFromResponses(u *rsUsage) Usage {
	if u == nil {
		return Usage{}
	}
	out := Usage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
	}
	if u.OutputTokensDetails != nil {
		out.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
	}
	if u.InputTokensDetails != nil {
		out.CachedTokens = u.InputTokensDetails.CachedTokens
		out.CacheReported = true
		if u.InputTokensDetails.CachedTokens > 0 && u.InputTokensDetails.CachedTokens <= out.PromptTokens {
			out.PromptTokens -= u.InputTokensDetails.CachedTokens
			out.CacheReadTokens += u.InputTokensDetails.CachedTokens
		}
	}
	return out
}

func chatResultFromResponses(r *rsResponse) *ChatResult {
	res := &ChatResult{}
	var summaries []string
	for _, item := range r.Output {
		switch item.Type {
		case "reasoning":
			if item.EncryptedContent != "" {
				res.ThinkingSignature = item.EncryptedContent
			}
			for _, s := range item.Summary {
				if s.Text != "" {
					summaries = append(summaries, s.Text)
				}
			}
		case "message":
			for _, p := range item.Content {
				if p.Type == "output_text" && p.Text != "" {
					res.Content += p.Text
				}
			}
		case "function_call":
			res.ToolCalls = append(res.ToolCalls, ToolCall{
				ID:        item.CallID,
				Name:      item.Name,
				Arguments: item.Arguments,
			})
		}
	}
	res.ReasoningContent = strings.Join(summaries, "\n")
	res.FinishReason = responsesFinishReason(r, len(res.ToolCalls) > 0)
	if r.Usage != nil {
		res.Usage = usageFromResponses(r.Usage)
	}
	return res
}

func responsesFinishReason(r *rsResponse, hasTools bool) string {
	switch r.Status {
	case "incomplete":
		if r.IncompleteDetails != nil && strings.Contains(strings.ToLower(r.IncompleteDetails.Reason), "max_output") {
			return FinishLength
		}
		if hasTools {
			return FinishToolCalls
		}
		return FinishLength
	case "failed", "cancelled":
		return ""
	default:
		if hasTools {
			return FinishToolCalls
		}
		return FinishStop
	}
}

func parseResponsesAPI(body []byte) (*ChatResult, error) {
	var r rsResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("llm: parse responses: %w", err)
	}
	if r.Error != nil && r.Error.Message != "" {
		return nil, fmt.Errorf("llm: provider error: %s", r.Error.Message)
	}
	if r.Status == "failed" {
		msg := "responses failed"
		if r.Error != nil && r.Error.Message != "" {
			msg = r.Error.Message
		}
		return nil, fmt.Errorf("llm: provider error: %s", msg)
	}
	return chatResultFromResponses(&r), nil
}

// ── streaming ────────────────────────────────────────────────────────────

type rsStreamEvent struct {
	Type        string        `json:"type"`
	Delta       string        `json:"delta"`
	OutputIndex int           `json:"output_index"`
	Item        *rsOutputItem `json:"item"`
	Response    *rsResponse   `json:"response"`
}

func mapResponsesStreamEvent(data []byte, acc *streamAccum) (deltas []Delta, done bool, err error) {
	var ev rsStreamEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, false, fmt.Errorf("llm: parse responses stream: %w", err)
	}
	switch ev.Type {
	case "response.reasoning_summary_text.delta":
		if ev.Delta != "" {
			acc.reasoning.WriteString(ev.Delta)
			deltas = append(deltas, Delta{Kind: DeltaReasoning, Text: ev.Delta})
		}
	case "response.output_text.delta":
		if ev.Delta != "" {
			acc.content.WriteString(ev.Delta)
			deltas = append(deltas, Delta{Kind: DeltaContent, Text: ev.Delta})
		}
	case "response.function_call_arguments.delta":
		call := acc.call(ev.OutputIndex)
		call.args.WriteString(ev.Delta)
		deltas = append(deltas, Delta{
			Kind:      DeltaToolArgs,
			Text:      ev.Delta,
			ToolIndex: ev.OutputIndex,
			ToolID:    call.id,
			ToolName:  call.name,
		})
	case "response.output_item.added":
		if ev.Item != nil && ev.Item.Type == "function_call" {
			call := acc.call(ev.OutputIndex)
			if ev.Item.CallID != "" {
				call.id = ev.Item.CallID
			}
			if ev.Item.Name != "" {
				call.name = ev.Item.Name
			}
		}
	case "response.output_item.done":
		if ev.Item != nil && ev.Item.Type == "reasoning" && ev.Item.EncryptedContent != "" {
			acc.thinkingSignature = ev.Item.EncryptedContent
		}
	case "response.completed", "response.incomplete":
		if ev.Response != nil {
			if ev.Response.Usage != nil {
				acc.usage = usageFromResponses(ev.Response.Usage)
			}
			hasTools := len(acc.calls) > 0
			acc.finishReason = responsesFinishReason(ev.Response, hasTools)
			if acc.thinkingSignature == "" {
				for _, item := range ev.Response.Output {
					if item.Type == "reasoning" && item.EncryptedContent != "" {
						acc.thinkingSignature = item.EncryptedContent
						break
					}
				}
			}
		} else if acc.finishReason == "" {
			if len(acc.calls) > 0 {
				acc.finishReason = FinishToolCalls
			} else if ev.Type == "response.incomplete" {
				acc.finishReason = FinishLength
			} else {
				acc.finishReason = FinishStop
			}
		}
		return deltas, true, nil
	case "response.failed":
		msg := "responses failed"
		if ev.Response != nil && ev.Response.Error != nil && ev.Response.Error.Message != "" {
			msg = ev.Response.Error.Message
		}
		return nil, true, fmt.Errorf("llm: provider error: %s", msg)
	}
	return deltas, false, nil
}
