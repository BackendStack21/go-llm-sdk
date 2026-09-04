// Package llm is a multi-provider Go SDK for LLM inference endpoints:
// OpenAI, Google Gemini, DeepSeek, Z.ai, Kimi (Moonshot) and Anthropic,
// plus any custom OpenAI-compatible gateway.
//
// Design highlights:
//
//   - Multiple authenticated endpoints simultaneously, discovered from the
//     environment via <PROVIDER>_API_KEY (aliases supported).
//   - Dynamic model discovery (ListModels) — no static model profile tables.
//   - One canonical request/response shape (OpenAI-compatible); Anthropic
//     and Gemini formats are translated by per-format serializers.
//   - Zero external dependencies: stdlib only.
//   - Streaming with idle watchdog, hard wall-clock deadline, abort-with-
//     partial-result, and retries that never duplicate partial output.
package llm

import (
	"encoding/json"
	"time"
)

// Role enumerates canonical message roles.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Canonical finish reasons.
const (
	FinishStop          = "stop"
	FinishLength        = "length"
	FinishToolCalls     = "tool_calls"
	FinishContentFilter = "content_filter"
)

// ToolCall is a model-requested tool invocation.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // JSON object as a string
}

// Message is one canonical chat message. For RoleTool messages, ToolCallID
// and ToolName identify the call being answered and Content carries the
// tool result. ReasoningContent is provider-reported thinking text
// (deepseek-reasoner, anthropic thinking, gemini thoughts). It is consumer
// metadata; the SDK replays it back only where a provider requires it for
// conversation continuity — Anthropic, and only when ThinkingSignature is
// also set (extended-thinking tool loops mandate the signed thinking block
// as the first block of the replayed assistant turn).
type Message struct {
	Role             Role
	Content          string
	ReasoningContent string
	// ThinkingSignature authenticates ReasoningContent for providers that
	// require thinking to be replayed verbatim (Anthropic signature).
	ThinkingSignature string
	ToolCalls         []ToolCall
	ToolCallID        string
	ToolName          string
}

// SystemBlock is one system-prompt segment. On Anthropic each block maps to
// a system text block (Cache marks it for prompt caching); on OpenAI-format
// providers blocks concatenate into a leading system message; on Gemini they
// become systemInstruction parts.
type SystemBlock struct {
	Text  string
	Cache bool
}

// ToolDef declares a callable tool. Parameters is a JSON Schema object
// (json.RawMessage so callers can pass through marshaled schemas verbatim).
type ToolDef struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// Usage reports token accounting. Fields the provider does not report stay 0.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	ReasoningTokens  int
}

// ChatRequest is the canonical request. Model is filled from the ChatClient
// when empty. Thinking accepts "", "enabled", "disabled", "low", "medium",
// "high" and is translated per provider format. Temperature: 0 means use
// the provider default (field omitted); use a negative value to explicitly
// send 0.
type ChatRequest struct {
	Model          string
	Messages       []Message
	System         []SystemBlock
	Tools          []ToolDef
	Thinking       string
	ThinkingBudget int
	MaxTokens      int
	Temperature    float64
}

// ChatResult is the canonical response for both buffered and streaming calls.
type ChatResult struct {
	Content          string
	ReasoningContent string
	// ThinkingSignature authenticates ReasoningContent (Anthropic extended
	// thinking). Consumers must carry it back on the next assistant Message
	// for tool loops to stay valid.
	ThinkingSignature string
	ToolCalls         []ToolCall
	FinishReason      string
	Usage             Usage
}

// DeltaKind discriminates streamed fragments.
type DeltaKind int

const (
	// DeltaReasoning is a thinking/reasoning fragment, usually before content.
	DeltaReasoning DeltaKind = iota
	// DeltaContent is an assistant text fragment.
	DeltaContent
	// DeltaToolArgs is a tool-call argument fragment (partial JSON). ToolIndex
	// and, on the first fragment of a call, ToolID/ToolName identify the call.
	DeltaToolArgs
)

// Delta is one streamed fragment. Text is the fragment for this event, not
// accumulated output.
type Delta struct {
	Kind      DeltaKind
	Text      string
	ToolIndex int
	ToolID    string
	ToolName  string
}

// Model is one accessible model, as reported by a provider's models endpoint.
// Fields the provider does not report stay zero — the SDK never guesses.
type Model struct {
	ID              string
	DisplayName     string
	CreatedAt       time.Time
	ContextWindow   int // input token limit, 0 = unknown
	MaxOutputTokens int // 0 = unknown
	Capabilities    []string
}
