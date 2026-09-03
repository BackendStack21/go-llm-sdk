package llm

import (
	"strings"
)

// Format identifies a wire protocol family. The SDK translates the single
// canonical request shape into exactly one of these.
type Format string

const (
	// FormatOpenAI is the OpenAI chat-completions protocol — also spoken by
	// DeepSeek, Z.ai (GLM), Kimi/Moonshot and most self-hosted gateways.
	FormatOpenAI Format = "openai"
	// FormatAnthropic is the Anthropic Messages API.
	FormatAnthropic Format = "anthropic"
	// FormatGemini is the Google Gemini generateContent API.
	FormatGemini Format = "gemini"
)

// Quirks carries per-provider protocol deviations, resolved at registration
// time. This replaces odek's URL-sniffing: a custom base URL gets its
// format's default quirks unless the caller overrides them explicitly.
type Quirks struct {
	// ThinkingObject: provider accepts the Anthropic-style top-level
	// "thinking" object (Anthropic, DeepSeek, Z.ai GLM).
	ThinkingObject bool
	// ReasoningEffort: provider accepts reasoning_effort (OpenAI, GLM-5.3+).
	ReasoningEffort bool
	// ForceThinking lists model-name prefixes that reject
	// thinking.type=disabled outright (GLM-5.3 always reasons; the
	// documented migration is {type: enabled} + reasoning_effort "low").
	ForceThinking []string
	// AnthropicVersion is the anthropic-version header value required by
	// FormatAnthropic providers ("2023-06-01").
	AnthropicVersion string
}

// ProviderConfig fully describes one inference endpoint. APIKey is resolved
// from EnvKeys (primary first) or set explicitly; it is never included in
// error text, logs, or String() output.
type ProviderConfig struct {
	ID      string
	Format  Format
	BaseURL string
	APIKey  string
	// EnvKeys lists environment variable names to consult, primary first
	// (aliases after). Resolution stops at the first non-empty value.
	EnvKeys []string
	Quirks  Quirks
}

func (c ProviderConfig) String() string {
	var b strings.Builder
	b.WriteString("llm.Provider{ID:")
	b.WriteString(c.ID)
	b.WriteString(" Format:")
	b.WriteString(string(c.Format))
	b.WriteString(" BaseURL:")
	b.WriteString(c.BaseURL)
	b.WriteString(" Authenticated:")
	b.WriteString(strconvBool(c.APIKey != ""))
	b.WriteString("}")
	return b.String()
}

func strconvBool(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// forceThinkingModel reports whether the model must keep thinking enabled.
func (q Quirks) forceThinkingModel(model string) bool {
	m := strings.ToLower(model)
	for _, p := range q.ForceThinking {
		if strings.HasPrefix(m, p) {
			return true
		}
	}
	return false
}

// temperatureForbiddenModels lists model prefixes that reject an explicit
// temperature parameter (only the provider default of 1 is accepted).
// Matching is model-name-based and provider-agnostic: OpenAI-compatible
// proxies serving these model IDs enforce the same constraint.
var temperatureForbiddenModels = []string{"o1", "o3", "o4", "gpt-5", "kimi-for-coding", "k3"}

// modelForbidsTemperature reports whether an explicit temperature must be
// omitted for this model.
func modelForbidsTemperature(model string) bool {
	m := strings.ToLower(model)
	for _, prefix := range temperatureForbiddenModels {
		if strings.HasPrefix(m, prefix) {
			return true
		}
	}
	return false
}

// completionTokenModels lists model prefixes that reject max_tokens in
// favor of max_completion_tokens (OpenAI o-series and gpt-5 families).
// Deliberately narrower than temperatureForbiddenModels: Z.ai/Moonshot
// models documented against max_tokens keep it.
var completionTokenModels = []string{"o1", "o3", "o4", "gpt-5"}

// modelUsesCompletionTokens reports whether this model's token limit must
// be sent as max_completion_tokens.
func modelUsesCompletionTokens(model string) bool {
	m := strings.ToLower(model)
	for _, prefix := range completionTokenModels {
		if strings.HasPrefix(m, prefix) {
			return true
		}
	}
	return false
}

// builtinProviders returns the built-in registry, in stable order.
func builtinProviders() []ProviderConfig {
	return []ProviderConfig{
		{
			ID:      "openai",
			Format:  FormatOpenAI,
			BaseURL: "https://api.openai.com/v1",
			EnvKeys: []string{"OPENAI_API_KEY"},
			Quirks:  Quirks{ReasoningEffort: true},
		},
		{
			ID:      "gemini",
			Format:  FormatGemini,
			BaseURL: "https://generativelanguage.googleapis.com",
			EnvKeys: []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"},
		},
		{
			ID:      "deepseek",
			Format:  FormatOpenAI,
			BaseURL: "https://api.deepseek.com",
			EnvKeys: []string{"DEEPSEEK_API_KEY"},
			Quirks:  Quirks{ThinkingObject: true},
		},
		{
			ID:      "zai",
			Format:  FormatOpenAI,
			BaseURL: "https://api.z.ai/api/paas/v4",
			EnvKeys: []string{"ZAI_API_KEY"},
			Quirks: Quirks{
				ThinkingObject:  true,
				ReasoningEffort: true,
				ForceThinking:   []string{"glm-5.3"},
			},
		},
		{
			ID:      "kimi",
			Format:  FormatOpenAI,
			BaseURL: "https://api.moonshot.ai/v1",
			EnvKeys: []string{"KIMI_API_KEY", "MOONSHOT_API_KEY"},
		},
		{
			ID:      "anthropic",
			Format:  FormatAnthropic,
			BaseURL: "https://api.anthropic.com",
			EnvKeys: []string{"ANTHROPIC_API_KEY"},
			Quirks:  Quirks{ThinkingObject: true, AnthropicVersion: "2023-06-01"},
		},
	}
}

// validateProviderConfig checks a custom provider registration.
func validateProviderConfig(c ProviderConfig) error {
	if c.ID == "" {
		return &ConfigError{Msg: "provider ID is empty"}
	}
	if strings.ContainsAny(c.ID, " \t") {
		return &ConfigError{Msg: "provider ID contains whitespace: " + c.ID}
	}
	switch c.Format {
	case FormatOpenAI, FormatAnthropic, FormatGemini:
	default:
		return &ConfigError{Msg: c.ID + ": unknown format " + string(c.Format)}
	}
	if c.BaseURL == "" {
		return &ConfigError{Msg: c.ID + ": empty base URL"}
	}
	if !strings.HasPrefix(c.BaseURL, "http://") && !strings.HasPrefix(c.BaseURL, "https://") {
		return &ConfigError{Msg: c.ID + ": base URL must start with http:// or https://"}
	}
	return nil
}

// envBaseURLKey returns the env var name that overrides a provider's base
// URL (e.g. "zai" → ZAI_BASE_URL).
func envBaseURLKey(id string) string {
	return strings.ToUpper(id) + "_BASE_URL"
}
