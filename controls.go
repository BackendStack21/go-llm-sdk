package llm

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Request controls (ToolChoice, ParallelToolCalls, ResponseFormat, Seed)
// are validated once at the SDK boundary, then mapped per format by the
// request builders. Contradictory or malformed controls fail fast with a
// ConfigError instead of a provider 400.

// schemaNameRE is the OpenAI json_schema name contract, applied to every
// format so one request stays portable.
var schemaNameRE = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// defaultSchemaName names a json_schema response format without a Name.
const defaultSchemaName = "response"

func validateRequestControls(req *ChatRequest) error {
	if tc := req.ToolChoice; tc != nil {
		if len(req.Tools) == 0 {
			return &ConfigError{Msg: "ToolChoice requires Tools"}
		}
		switch tc.Mode {
		case ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired:
			if tc.Name != "" {
				return &ConfigError{Msg: fmt.Sprintf("ToolChoice mode %q takes no Name", tc.Mode)}
			}
		case ToolChoiceTool:
			if tc.Name == "" {
				return &ConfigError{Msg: "ToolChoice mode \"tool\" requires a Name"}
			}
			if !hasTool(req.Tools, tc.Name) {
				return &ConfigError{Msg: fmt.Sprintf("ToolChoice names unknown tool %q", tc.Name)}
			}
		default:
			return &ConfigError{Msg: fmt.Sprintf("unknown ToolChoice mode %q", tc.Mode)}
		}
	}
	if rf := req.ResponseFormat; rf != nil {
		switch rf.Type {
		case "", ResponseText, ResponseJSONObject:
			if len(rf.Schema) > 0 {
				return &ConfigError{Msg: fmt.Sprintf("ResponseFormat %q takes no Schema", rf.Type)}
			}
		case ResponseJSONSchema:
			t := strings.TrimSpace(string(rf.Schema))
			if !strings.HasPrefix(t, "{") || !json.Valid([]byte(t)) {
				return &ConfigError{Msg: "ResponseFormat json_schema requires a JSON Schema object"}
			}
		default:
			return &ConfigError{Msg: fmt.Sprintf("unknown ResponseFormat type %q", rf.Type)}
		}
		if rf.Name != "" && !schemaNameRE.MatchString(rf.Name) {
			return &ConfigError{Msg: fmt.Sprintf("ResponseFormat Name %q must match [A-Za-z0-9_-]{1,64}", rf.Name)}
		}
	}
	return nil
}

func hasTool(tools []ToolDef, name string) bool {
	for _, t := range tools {
		if t.Name == name {
			return true
		}
	}
	return false
}

// jsonMode reports whether rf asks for JSON output.
func (rf *ResponseFormat) jsonMode() bool {
	return rf != nil && (rf.Type == ResponseJSONObject || rf.Type == ResponseJSONSchema)
}

// schemaName is the json_schema name sent on the wire.
func (rf *ResponseFormat) schemaName() string {
	if rf.Name != "" {
		return rf.Name
	}
	return defaultSchemaName
}

// ── OpenAI chat completions ──────────────────────────────────────────────

type oaJSONSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict,omitempty"`
}

type oaResponseFormat struct {
	Type       string        `json:"type"`
	JSONSchema *oaJSONSchema `json:"json_schema,omitempty"`
}

// openAIToolChoice renders tool_choice ("auto" | "none" | "required" |
// {"type":"function","function":{"name":…}}).
func openAIToolChoice(tc *ToolChoice) any {
	if tc == nil {
		return nil
	}
	if tc.Mode == ToolChoiceTool {
		return map[string]any{"type": "function", "function": map[string]string{"name": tc.Name}}
	}
	return string(tc.Mode)
}

func openAIResponseFormat(rf *ResponseFormat) *oaResponseFormat {
	switch {
	case !rf.jsonMode():
		return nil
	case rf.Type == ResponseJSONObject:
		return &oaResponseFormat{Type: "json_object"}
	default:
		return &oaResponseFormat{Type: "json_schema", JSONSchema: &oaJSONSchema{Name: rf.schemaName(), Schema: rf.Schema, Strict: rf.Strict}}
	}
}

// parallelToolCalls is sent only alongside tools (OpenAI rejects it
// otherwise).
func parallelToolCalls(req *ChatRequest) *bool {
	if len(req.Tools) == 0 {
		return nil
	}
	return req.ParallelToolCalls
}

// ── OpenAI Responses ─────────────────────────────────────────────────────

type rsTextFormat struct {
	Type   string          `json:"type"`
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict bool            `json:"strict,omitempty"`
}

type rsText struct {
	Format rsTextFormat `json:"format"`
}

func responsesToolChoice(tc *ToolChoice) any {
	if tc == nil {
		return nil
	}
	if tc.Mode == ToolChoiceTool {
		return map[string]string{"type": "function", "name": tc.Name}
	}
	return string(tc.Mode)
}

func responsesText(rf *ResponseFormat) *rsText {
	switch {
	case !rf.jsonMode():
		return nil
	case rf.Type == ResponseJSONObject:
		return &rsText{Format: rsTextFormat{Type: "json_object"}}
	default:
		return &rsText{Format: rsTextFormat{Type: "json_schema", Name: rf.schemaName(), Schema: rf.Schema, Strict: rf.Strict}}
	}
}

// ── Anthropic ────────────────────────────────────────────────────────────

type anToolChoice struct {
	Type                   string `json:"type"` // auto | any | tool | none
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

// anthropicJSONToolDefault names the synthetic JSON-mode tool when
// ResponseFormat.Name is empty.
const anthropicJSONToolDefault = "json_response"

// anthropicJSONToolName returns the synthetic tool that carries a JSON
// response on Anthropic ("" when JSON mode is off). Anthropic has no
// portable response_format, so JSON mode forces a tool whose input schema
// is the requested schema and folds its input back into Content.
func anthropicJSONToolName(req *ChatRequest) string {
	if req == nil || !req.ResponseFormat.jsonMode() {
		return ""
	}
	if req.ResponseFormat.Name != "" {
		return req.ResponseFormat.Name
	}
	return anthropicJSONToolDefault
}

// anthropicToolControls appends the JSON-mode tool when requested and
// renders tool_choice. thinking reports extended thinking, which rejects
// forced tool use.
func anthropicToolControls(req *ChatRequest, out *anRequest, thinking bool) error {
	var choice *anToolChoice
	if name := anthropicJSONToolName(req); name != "" {
		switch {
		case req.ToolChoice != nil:
			return &ConfigError{Msg: "Anthropic JSON ResponseFormat forces its own tool; it cannot be combined with ToolChoice"}
		case thinking:
			return &ConfigError{Msg: "Anthropic JSON ResponseFormat forces tool use, which extended thinking rejects"}
		case len(req.Tools) > 0:
			// The forced JSON tool would make every user tool uncallable.
			return &ConfigError{Msg: "Anthropic JSON ResponseFormat forces its own tool; it cannot be combined with Tools"}
		case !objectSchema(req.ResponseFormat.Schema):
			return &ConfigError{Msg: "Anthropic JSON ResponseFormat needs a top-level \"type\": \"object\" schema"}
		}
		schema := req.ResponseFormat.Schema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		out.Tools = append(out.Tools, anTool{
			Name:        name,
			Description: "Return the final answer as JSON matching this schema.",
			InputSchema: schema,
		})
		choice = &anToolChoice{Type: "tool", Name: name}
	} else if tc := req.ToolChoice; tc != nil {
		switch tc.Mode {
		case ToolChoiceAuto:
			choice = &anToolChoice{Type: "auto"}
		case ToolChoiceNone:
			choice = &anToolChoice{Type: "none"}
		case ToolChoiceRequired:
			choice = &anToolChoice{Type: "any"}
		default: // ToolChoiceTool (validated)
			choice = &anToolChoice{Type: "tool", Name: tc.Name}
		}
		if thinking && (choice.Type == "any" || choice.Type == "tool") {
			return &ConfigError{Msg: "Anthropic extended thinking rejects forced tool use (ToolChoice required/tool)"}
		}
	}
	if p := req.ParallelToolCalls; p != nil && !*p && len(out.Tools) > 0 && (choice == nil || choice.Type != "none") {
		if choice == nil {
			choice = &anToolChoice{Type: "auto"}
		}
		choice.DisableParallelToolUse = true
	}
	out.ToolChoice = choice
	return nil
}

// objectSchema reports whether a schema is absent (json_object) or declares
// a top-level object type — the only shape Anthropic accepts as a tool's
// input_schema.
func objectSchema(schema json.RawMessage) bool {
	if len(schema) == 0 {
		return true
	}
	var s struct {
		Type any `json:"type"`
	}
	return json.Unmarshal(schema, &s) == nil && s.Type == "object"
}

// foldAnthropicJSON moves the JSON-mode tool call back into Content: the
// caller asked for JSON text, not a tool call. Other tool calls stay.
func foldAnthropicJSON(res *ChatResult, name string) {
	if res == nil || name == "" {
		return
	}
	var kept []ToolCall
	folded := false
	for _, tc := range res.ToolCalls {
		if !folded && tc.Name == name {
			res.Content += tc.Arguments
			folded = true
			continue
		}
		kept = append(kept, tc)
	}
	if !folded {
		return
	}
	res.ToolCalls = kept
	if res.FinishReason == FinishToolCalls && len(kept) == 0 {
		res.FinishReason = FinishStop
	}
}

// ── Gemini ───────────────────────────────────────────────────────────────

type gmFnCallingCfg struct {
	Mode                 string   `json:"mode"` // AUTO | NONE | ANY
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type gmToolConfig struct {
	FunctionCallingConfig gmFnCallingCfg `json:"functionCallingConfig"`
}

func geminiToolConfig(tc *ToolChoice) *gmToolConfig {
	if tc == nil {
		return nil
	}
	switch tc.Mode {
	case ToolChoiceNone:
		return &gmToolConfig{FunctionCallingConfig: gmFnCallingCfg{Mode: "NONE"}}
	case ToolChoiceRequired:
		return &gmToolConfig{FunctionCallingConfig: gmFnCallingCfg{Mode: "ANY"}}
	case ToolChoiceTool:
		return &gmToolConfig{FunctionCallingConfig: gmFnCallingCfg{Mode: "ANY", AllowedFunctionNames: []string{tc.Name}}}
	default:
		return &gmToolConfig{FunctionCallingConfig: gmFnCallingCfg{Mode: "AUTO"}}
	}
}
