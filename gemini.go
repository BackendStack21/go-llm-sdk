package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Google Gemini generateContent format. Canonical requests translate to
// contents/parts with systemInstruction, functionDeclarations, and
// generationConfig.thinkingConfig; the model rides the URL path, not the
// body. Gemini reports no tool-call ids — the SDK synthesizes call_<n>.

// ── request ──────────────────────────────────────────────────────────────

type gmFnCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type gmFnResp struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type gmPart struct {
	Text             string    `json:"text,omitempty"`
	Thought          bool      `json:"thought,omitempty"`
	FunctionCall     *gmFnCall `json:"functionCall,omitempty"`
	FunctionResponse *gmFnResp `json:"functionResponse,omitempty"`
}

type gmContent struct {
	Role  string   `json:"role,omitempty"` // "user" | "model"
	Parts []gmPart `json:"parts"`
}

type gmFnDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type gmToolGroup struct {
	FunctionDeclarations []gmFnDecl `json:"functionDeclarations"`
}

type gmThinkCfg struct {
	ThinkingBudget  int  `json:"thinkingBudget"` // 0 = off, -1 = dynamic
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
}

type gmGenCfg struct {
	MaxOutputTokens int         `json:"maxOutputTokens,omitempty"`
	Temperature     *float64    `json:"temperature,omitempty"`
	ThinkingConfig  *gmThinkCfg `json:"thinkingConfig,omitempty"`
}

type gmRequest struct {
	SystemInstruction *gmContent    `json:"systemInstruction,omitempty"`
	Contents          []gmContent   `json:"contents"`
	Tools             []gmToolGroup `json:"tools,omitempty"`
	GenerationConfig  *gmGenCfg     `json:"generationConfig,omitempty"`
}

// geminiThinkingConfig maps canonical thinking to thinkingConfig.
// Budgets: low 1024, medium 8192, high 24576; -1 = dynamic (provider-decided).
func geminiThinkingConfig(level string, explicit int) *gmThinkCfg {
	switch level {
	case "enabled":
		b := -1
		if explicit > 0 {
			b = explicit
		}
		return &gmThinkCfg{ThinkingBudget: b, IncludeThoughts: true}
	case "disabled":
		return &gmThinkCfg{ThinkingBudget: 0}
	case "low":
		return &gmThinkCfg{ThinkingBudget: 1024, IncludeThoughts: true}
	case "medium":
		return &gmThinkCfg{ThinkingBudget: 8192, IncludeThoughts: true}
	case "high":
		return &gmThinkCfg{ThinkingBudget: 24576, IncludeThoughts: true}
	default: // ""
		return nil
	}
}

// wrapToolResponse ensures functionResponse.response is a JSON object:
// pass through valid object JSON, wrap everything else under "result".
func wrapToolResponse(content string) json.RawMessage {
	t := strings.TrimSpace(content)
	if strings.HasPrefix(t, "{") && json.Valid([]byte(t)) {
		return json.RawMessage(t)
	}
	return json.RawMessage(fmt.Sprintf(`{"result":%s}`, mustJSONString(content)))
}

func mustJSONString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// buildGeminiRequest renders the canonical request in Gemini format.
func buildGeminiRequest(req *ChatRequest, model string, stream bool) ([]byte, error) {
	_ = model // rides the URL path
	_ = stream

	out := gmRequest{}

	// System: canonical blocks + in-band system messages → systemInstruction.
	var sysParts []gmPart
	for _, b := range req.System {
		sysParts = append(sysParts, gmPart{Text: b.Text})
	}
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			sysParts = append(sysParts, gmPart{Text: m.Content})
		}
	}
	if len(sysParts) > 0 {
		out.SystemInstruction = &gmContent{Parts: sysParts}
	}

	toolNameByID := make(map[string]string)
	for i := 0; i < len(req.Messages); i++ {
		m := req.Messages[i]
		switch m.Role {
		case RoleSystem:
			// folded into systemInstruction
		case RoleUser:
			out.Contents = append(out.Contents, gmContent{
				Role:  "user",
				Parts: []gmPart{{Text: m.Content}},
			})
		case RoleAssistant:
			for _, tc := range m.ToolCalls {
				if tc.ID != "" {
					toolNameByID[tc.ID] = tc.Name
				}
			}
			parts := []gmPart{}
			if m.Content != "" {
				parts = append(parts, gmPart{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				args := json.RawMessage(tc.Arguments)
				if len(args) == 0 {
					args = json.RawMessage("{}")
				}
				parts = append(parts, gmPart{FunctionCall: &gmFnCall{Name: tc.Name, Args: args}})
			}
			if len(parts) == 0 {
				parts = []gmPart{{Text: " "}} // Gemini rejects empty parts
			}
			out.Contents = append(out.Contents, gmContent{Role: "model", Parts: parts})
		case RoleTool:
			// Merge the run of consecutive tool messages into one user
			// content with functionResponse parts.
			var parts []gmPart
			for ; i < len(req.Messages) && req.Messages[i].Role == RoleTool; i++ {
				tm := req.Messages[i]
				name := tm.ToolName
				if name == "" {
					// Consumers ported from the OpenAI format set only
					// ToolCallID; recover the function name from the
					// assistant tool_call it answers.
					name = toolNameByID[tm.ToolCallID]
				}
				if name == "" {
					return nil, &ConfigError{Msg: "tool result for \"" + tm.ToolCallID + "\" has no ToolName and no matching assistant tool_call"}
				}
				parts = append(parts, gmPart{
					FunctionResponse: &gmFnResp{
						Name:     name,
						Response: wrapToolResponse(tm.Content),
					},
				})
			}
			i--
			out.Contents = append(out.Contents, gmContent{Role: "user", Parts: parts})
		}
	}

	if len(req.Tools) > 0 {
		g := gmToolGroup{}
		for _, t := range req.Tools {
			g.FunctionDeclarations = append(g.FunctionDeclarations, gmFnDecl(t))
		}
		out.Tools = []gmToolGroup{g}
	}

	cfg := gmGenCfg{MaxOutputTokens: req.MaxTokens}
	if req.Temperature != 0 {
		t := req.Temperature
		if t < 0 {
			t = 0
		}
		cfg.Temperature = &t
	}
	if tc := geminiThinkingConfig(req.Thinking, req.ThinkingBudget); tc != nil {
		cfg.ThinkingConfig = tc
	}
	if cfg.MaxOutputTokens != 0 || cfg.Temperature != nil || cfg.ThinkingConfig != nil {
		out.GenerationConfig = &cfg
	}
	return json.Marshal(out)
}

// ── response ─────────────────────────────────────────────────────────────

type gmRespPart struct {
	Text         string    `json:"text"`
	Thought      bool      `json:"thought"`
	FunctionCall *gmFnCall `json:"functionCall"`
}

type gmCandidate struct {
	Content      gmRespContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type gmRespContent struct {
	Role  string       `json:"role"`
	Parts []gmRespPart `json:"parts"`
}

type gmUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount"`
}

type gmResponse struct {
	Candidates    []gmCandidate `json:"candidates"`
	UsageMetadata gmUsage       `json:"usageMetadata"`
}

// mapGeminiFinishReason maps finishReason to canonical values.
func mapGeminiFinishReason(s string) string {
	switch s {
	case "STOP":
		return FinishStop
	case "MAX_TOKENS":
		return FinishLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII":
		return FinishContentFilter
	case "":
		return ""
	default:
		// Provider-specific reasons stay out of the canonical vocabulary.
		return ""
	}
}

func mapGeminiUsage(u gmUsage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: u.CandidatesTokenCount,
		ReasoningTokens:  u.ThoughtsTokenCount,
	}
}

// foldGeminiParts folds response parts into acc, emitting deltas for new
// fragments. Used by both buffered and streaming parsing.
func foldGeminiParts(parts []gmRespPart, acc *streamAccum) []Delta {
	var deltas []Delta
	for _, p := range parts {
		switch {
		case p.FunctionCall != nil:
			idx := len(acc.calls)
			c := acc.call(idx)
			c.id = fmt.Sprintf("call_%d", idx)
			c.name = p.FunctionCall.Name
			args := string(p.FunctionCall.Args)
			if args == "" {
				args = "{}"
			}
			c.args.WriteString(args)
			deltas = append(deltas, Delta{
				Kind:      DeltaToolArgs,
				Text:      args,
				ToolIndex: idx,
				ToolID:    c.id,
				ToolName:  c.name,
			})
		case p.Text != "":
			if p.Thought {
				acc.reasoning.WriteString(p.Text)
				deltas = append(deltas, Delta{Kind: DeltaReasoning, Text: p.Text})
			} else {
				acc.content.WriteString(p.Text)
				deltas = append(deltas, Delta{Kind: DeltaContent, Text: p.Text})
			}
		}
	}
	return deltas
}

// parseGeminiResponse parses a buffered generateContent response body.
func parseGeminiResponse(body []byte) (*ChatResult, error) {
	var r gmResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, fmt.Errorf("llm: parse response: %w", err)
	}
	if len(r.Candidates) == 0 {
		return nil, fmt.Errorf("llm: response has no candidates")
	}
	acc := newStreamAccum()
	c := r.Candidates[0]
	foldGeminiParts(c.Content.Parts, acc)
	res := acc.result()
	res.FinishReason = mapGeminiFinishReason(c.FinishReason)
	res.Usage = mapGeminiUsage(r.UsageMetadata)
	return res, nil
}

// ── streaming ────────────────────────────────────────────────────────────

type gmStreamChunk struct {
	Candidates    []gmCandidate `json:"candidates"`
	UsageMetadata *gmUsage      `json:"usageMetadata"` // nil = chunk carries no usage update
}

// mapGeminiStreamEvent folds one Gemini SSE chunk into acc. Chunks carry
// full response shapes; usage metadata is cumulative (last wins). There is
// no [DONE] sentinel — the stream simply ends.
func mapGeminiStreamEvent(data []byte, acc *streamAccum) ([]Delta, bool, error) {
	var c gmStreamChunk
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, false, fmt.Errorf("llm: parse stream chunk: %w", err)
	}
	if c.UsageMetadata != nil {
		acc.usage = mapGeminiUsage(*c.UsageMetadata)
	}
	if len(c.Candidates) == 0 {
		return nil, false, nil
	}
	cand := c.Candidates[0]
	deltas := foldGeminiParts(cand.Content.Parts, acc)
	if cand.FinishReason != "" {
		acc.finishReason = mapGeminiFinishReason(cand.FinishReason)
	}
	return deltas, false, nil
}

// ── model discovery ──────────────────────────────────────────────────────

type gmModelEntry struct {
	Name             string   `json:"name"` // "models/<id>"
	DisplayName      string   `json:"displayName"`
	InputTokenLimit  int      `json:"inputTokenLimit"`
	OutputTokenLimit int      `json:"outputTokenLimit"`
	SupportedMethods []string `json:"supportedGenerationMethods"`
}

type gmModelsPage struct {
	Models        []gmModelEntry `json:"models"`
	NextPageToken string         `json:"nextPageToken"`
}

// listModelsGemini pages through GET /v1beta/models. Gemini is the richest
// source: it reports input/output token limits per model.
func listModelsGemini(ctx context.Context, pc *providerClient) ([]Model, error) {
	var out []Model
	pageToken := ""
	for page := 0; page < 10; page++ {
		url := pc.base + "/v1beta/models?pageSize=100"
		if pageToken != "" {
			url += "&pageToken=" + pageToken
		}
		data, _, err := pc.get(ctx, url)
		if err != nil {
			return nil, err
		}
		var p gmModelsPage
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, fmt.Errorf("llm: parse models response: %w", err)
		}
		for _, m := range p.Models {
			out = append(out, Model{
				ID:              strings.TrimPrefix(m.Name, "models/"),
				DisplayName:     m.DisplayName,
				ContextWindow:   m.InputTokenLimit,
				MaxOutputTokens: m.OutputTokenLimit,
				Capabilities:    m.SupportedMethods,
			})
		}
		if p.NextPageToken == "" {
			return out, nil
		}
		pageToken = p.NextPageToken
	}
	return out, nil
}
