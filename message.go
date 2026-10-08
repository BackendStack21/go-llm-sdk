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

// ContentPartType identifies the payload carried by ContentPart.
type ContentPartType string

const (
	ContentPartText  ContentPartType = "text"
	ContentPartImage ContentPartType = "image"
)

// ContentPart is one ordered text or inline-image segment in a message.
// Image data is raw bytes and is base64-encoded only by provider serializers.
// A part must contain exactly one supported payload: Text, or Image with a
// valid MIMEType. Inline images are bounded by MaxImageBytes.
type ContentPart struct {
	Type     ContentPartType
	Text     string
	Image    []byte
	MIMEType string
}

const MaxImageBytes = 10 << 20

// TextPart creates an ordered text content part.
func TextPart(text string) ContentPart { return ContentPart{Type: ContentPartText, Text: text} }

// ImagePart creates an ordered inline-image content part.
func ImagePart(mimeType string, data []byte) ContentPart {
	return ContentPart{Type: ContentPartImage, Image: data, MIMEType: mimeType}
}

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
	// Signature is the provider's opaque per-call reasoning signature
	// (Gemini thoughtSignature). Replay it unchanged; Gemini 3 rejects a
	// function-call turn whose signature is missing.
	Signature string
}

// ThinkingBlock is one provider-signed reasoning segment that must be
// replayed verbatim and in order: an Anthropic thinking or
// redacted_thinking block, or an OpenAI Responses reasoning item.
type ThinkingBlock struct {
	Text      string // reasoning text (summary on Responses); empty when redacted
	Signature string // Anthropic signature / Responses encrypted_content
	Redacted  string // Anthropic redacted_thinking data; Text and Signature stay empty
}

// Message is one canonical chat message. For RoleTool messages, ToolCallID
// and ToolName identify the call being answered and Content carries the
// tool result. ReasoningContent is provider-reported thinking text
// (deepseek-reasoner, anthropic thinking, gemini thoughts). The SDK
// replays it where a provider requires conversation continuity:
// OpenAI-format assistant messages echo it as reasoning_content
// (DeepSeek tool loops; with tools present the key is echoed even
// when empty, because DeepSeek rejects a missing key with 400), and
// Anthropic re-serializes a signed thinking block as the first content
// block when ThinkingSignature is also set.
type Message struct {
	Role    Role
	Content string
	// Parts optionally carries ordered text and inline-image content. When
	// empty, Content retains the legacy plain-text representation.
	Parts            []ContentPart
	ReasoningContent string
	// ThinkingSignature authenticates ReasoningContent for providers that
	// require thinking to be replayed verbatim (Anthropic signature, OpenAI
	// Responses encrypted_content).
	ThinkingSignature string
	// ThinkingBlocks replays every signed reasoning segment of a previous
	// assistant turn, in order. When set it supersedes ReasoningContent and
	// ThinkingSignature on Anthropic and Responses requests.
	ThinkingBlocks []ThinkingBlock
	ToolCalls      []ToolCall
	ToolCallID     string
	ToolName       string
	// Cache marks this user message for Anthropic prompt caching
	// (cache_control ephemeral on the text block). Ignored on other
	// formats and on non-user roles.
	Cache bool
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
	// Cache marks this tool for Anthropic prompt caching. Set it on the
	// last tool so the catalog is a stable prefix independent of later
	// system rows. OpenAI-format serializers ignore it.
	Cache bool
}

// Usage reports token accounting with one meaning on every format. Fields
// the provider does not report stay 0.
//
//   - PromptTokens is uncached input only. Inclusive provider counts
//     (OpenAI cached_tokens, DeepSeek prompt_cache_hit_tokens, Gemini
//     cachedContentTokenCount) are subtracted and moved to CacheReadTokens;
//     Anthropic reports cache volumes exclusively and is left alone.
//   - CacheReadTokens / CacheCreationTokens are cache reads and cache
//     writes. A DeepSeek cache miss is ordinary input (billed as such), not
//     a write, so it stays in PromptTokens.
//   - CompletionTokens counts every generated token, reasoning included
//     (Gemini's candidatesTokenCount excludes thoughts, so they are added);
//     ReasoningTokens is the reasoning subset.
//   - CachedTokens is the raw provider-reported cached count, kept for
//     diagnostics. It is already included in CacheReadTokens — never add it.
//
// Sum with InputTokens and TotalTokens rather than by hand.
type Usage struct {
	PromptTokens        int
	CompletionTokens    int
	ReasoningTokens     int
	CacheReadTokens     int
	CacheCreationTokens int
	CachedTokens        int
	CacheReported       bool
}

// InputTokens is the full input volume: uncached prompt plus cache reads
// and writes.
func (u Usage) InputTokens() int {
	return u.PromptTokens + u.CacheReadTokens + u.CacheCreationTokens
}

// TotalTokens is InputTokens plus CompletionTokens.
func (u Usage) TotalTokens() int { return u.InputTokens() + u.CompletionTokens }

// ChatRequest is the canonical request. Model is filled from the ChatClient
// when empty. Thinking accepts "", "enabled", "disabled", "low", "medium",
// "high", "max" and is translated per provider format. Temperature and TopP:
// 0 means use the provider default (field omitted); use a negative value to
// explicitly send 0. Stop maps to each provider's stop-sequence field.
type ChatRequest struct {
	Model          string
	Messages       []Message
	System         []SystemBlock
	Tools          []ToolDef
	Thinking       string
	ThinkingBudget int
	MaxTokens      int
	Temperature    float64
	TopP           float64
	Stop           []string
}

// ChatResult is the canonical response for both buffered and streaming calls.
type ChatResult struct {
	Content          string
	ReasoningContent string
	// ThinkingSignature authenticates ReasoningContent (Anthropic extended
	// thinking). Consumers must carry it back on the next assistant Message
	// for tool loops to stay valid.
	ThinkingSignature string
	// ThinkingBlocks lists every signed reasoning segment in response order
	// (Anthropic thinking/redacted_thinking, Responses reasoning items).
	ThinkingBlocks []ThinkingBlock
	ToolCalls      []ToolCall
	FinishReason   string
	Usage          Usage
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

// AssistantMessage returns the assistant turn to append to the conversation
// before the next request. It carries every replay field (reasoning,
// signatures, thinking blocks, tool calls) so tool loops stay valid on
// every format. Slices are copied.
func (r *ChatResult) AssistantMessage() Message {
	m := Message{Role: RoleAssistant}
	if r == nil {
		return m
	}
	m.Content = r.Content
	m.ReasoningContent = r.ReasoningContent
	m.ThinkingSignature = r.ThinkingSignature
	if len(r.ThinkingBlocks) > 0 {
		m.ThinkingBlocks = append([]ThinkingBlock(nil), r.ThinkingBlocks...)
	}
	if len(r.ToolCalls) > 0 {
		m.ToolCalls = append([]ToolCall(nil), r.ToolCalls...)
	}
	return m
}
