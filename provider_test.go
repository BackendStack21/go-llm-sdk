package llm

import (
	"strings"
	"testing"
)

func TestBuiltinProviders_RegistryFacts(t *testing.T) {
	builtins := builtinProviders()
	if len(builtins) != 6 {
		t.Fatalf("builtins = %d, want 6", len(builtins))
	}
	byID := map[string]ProviderConfig{}
	for _, b := range builtins {
		byID[b.ID] = b
	}
	// Env keys: primary first, aliases after.
	cases := []struct {
		id   string
		want []string
	}{
		{"openai", []string{"OPENAI_API_KEY"}},
		{"gemini", []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}},
		{"deepseek", []string{"DEEPSEEK_API_KEY"}},
		{"zai", []string{"ZAI_API_KEY"}},
		{"kimi", []string{"KIMI_API_KEY", "MOONSHOT_API_KEY"}},
		{"anthropic", []string{"ANTHROPIC_API_KEY"}},
	}
	for _, c := range cases {
		got := strings.Join(byID[c.id].EnvKeys, ",")
		want := strings.Join(c.want, ",")
		if got != want {
			t.Errorf("%s EnvKeys = %q, want %q", c.id, got, want)
		}
	}
	// Quirk pins.
	if !byID["openai"].Quirks.ReasoningEffort || byID["openai"].Quirks.ThinkingObject {
		t.Error("openai quirks: want ReasoningEffort only")
	}
	if !byID["deepseek"].Quirks.ThinkingObject || byID["deepseek"].Quirks.ReasoningEffort {
		t.Error("deepseek quirks: want ThinkingObject only")
	}
	zq := byID["zai"].Quirks
	if !zq.ThinkingObject || !zq.ReasoningEffort || len(zq.ForceThinking) != 1 || zq.ForceThinking[0] != "glm-5.3" {
		t.Errorf("zai quirks = %+v, want ThinkingObject+ReasoningEffort+ForceThinking[glm-5.3]", zq)
	}
	if !byID["anthropic"].Quirks.ThinkingObject || byID["anthropic"].Quirks.AnthropicVersion != "2023-06-01" {
		t.Error("anthropic quirks: want ThinkingObject + version header")
	}
	if byID["anthropic"].Format != FormatAnthropic || byID["gemini"].Format != FormatGemini {
		t.Error("anthropic/gemini format mismatch")
	}
}

func TestQuirksForceThinkingModel(t *testing.T) {
	q := Quirks{ForceThinking: []string{"glm-5.3"}}
	for _, m := range []string{"glm-5.3", "GLM-5.3-Air", "glm-5.3.4"} {
		if !q.forceThinkingModel(m) {
			t.Errorf("forceThinkingModel(%q) = false, want true", m)
		}
	}
	for _, m := range []string{"glm-4.6", "glm-5", "deepseek-v4"} {
		if q.forceThinkingModel(m) {
			t.Errorf("forceThinkingModel(%q) = true, want false", m)
		}
	}
}

func TestModelForbidsTemperature(t *testing.T) {
	for _, m := range []string{"o1", "o3-mini", "o4-high", "gpt-5.6-luna", "kimi-for-coding-v1", "k3-turbo"} {
		if !modelForbidsTemperature(m) {
			t.Errorf("modelForbidsTemperature(%q) = false, want true", m)
		}
	}
	for _, m := range []string{"gpt-4o", "deepseek-v4", "glm-5.3", "claude-sonnet-4"} {
		if modelForbidsTemperature(m) {
			t.Errorf("modelForbidsTemperature(%q) = true, want false", m)
		}
	}
}

func TestValidateProviderConfig(t *testing.T) {
	bad := []ProviderConfig{
		{ID: "", Format: FormatOpenAI, BaseURL: "https://x"},
		{ID: "has space", Format: FormatOpenAI, BaseURL: "https://x"},
		{ID: "x", Format: "bogus", BaseURL: "https://x"},
		{ID: "x", Format: FormatOpenAI, BaseURL: ""},
		{ID: "x", Format: FormatOpenAI, BaseURL: "ftp://x"},
	}
	for i, c := range bad {
		if validateProviderConfig(c) == nil {
			t.Errorf("case %d: expected error for %+v", i, c)
		}
	}
	good := ProviderConfig{ID: "gw", Format: FormatOpenAI, BaseURL: "http://localhost:8080/v1"}
	if err := validateProviderConfig(good); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestProviderConfigString_NoAPIKeyLeak(t *testing.T) {
	cfg := ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: "https://api.openai.com/v1", APIKey: "sk-super-secret"}
	s := cfg.String()
	if strings.Contains(s, "sk-super-secret") {
		t.Fatalf("String() leaks API key: %s", s)
	}
	if !strings.Contains(s, "Authenticated:true") {
		t.Fatalf("String() missing auth flag: %s", s)
	}
}

func TestSDK_WithProvider_CustomGatewayOverrides(t *testing.T) {
	sdk := New(WithProvider("openai",
		WithBaseURL("http://localhost:9999/v1"),
		WithAPIKey("local"),
	))
	p, err := sdk.Provider("openai")
	if err != nil {
		t.Fatal(err)
	}
	cfg := p.Config()
	if cfg.BaseURL != "http://localhost:9999/v1" || cfg.APIKey != "local" {
		t.Fatalf("override not applied: %+v", cfg)
	}
	if !p.Authenticated() {
		t.Fatal("expected authenticated")
	}
}

func TestSDK_WithProvider_InvalidCustomRejected(t *testing.T) {
	sdk := New(WithProvider("broken", WithAPIKey("k"))) // missing Format + BaseURL
	for _, p := range sdk.Providers() {
		if p.ID() == "broken" {
			t.Fatal("invalid custom provider must not appear in Providers()")
		}
	}
	if _, err := sdk.Chat("broken", "m"); err == nil {
		t.Fatal("Chat on invalid provider must fail")
	}
}
