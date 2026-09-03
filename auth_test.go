package llm

import (
	"testing"
)

func TestResolveEnvProvider_PrimaryWinsOverAlias(t *testing.T) {
	env := map[string]string{
		"GEMINI_API_KEY": "primary-key",
		"GOOGLE_API_KEY": "alias-key",
	}
	lookup := func(k string) (string, bool) { v, ok := env[k]; return v, ok }

	cfg := ProviderConfig{ID: "gemini", EnvKeys: []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}}
	got, ok := resolveEnvProvider(cfg, lookup)
	if !ok {
		t.Fatal("expected authenticated")
	}
	if got.APIKey != "primary-key" {
		t.Fatalf("APIKey = %q, want primary-key (primary beats alias)", got.APIKey)
	}
}

func TestResolveEnvProvider_AliasFallback(t *testing.T) {
	env := map[string]string{"MOONSHOT_API_KEY": "alias-key"}
	lookup := func(k string) (string, bool) { v, ok := env[k]; return v, ok }

	cfg := ProviderConfig{ID: "kimi", EnvKeys: []string{"KIMI_API_KEY", "MOONSHOT_API_KEY"}}
	got, ok := resolveEnvProvider(cfg, lookup)
	if !ok {
		t.Fatal("expected authenticated via alias")
	}
	if got.APIKey != "alias-key" {
		t.Fatalf("APIKey = %q, want alias-key", got.APIKey)
	}
}

func TestResolveEnvProvider_NoKeyUnauthenticated(t *testing.T) {
	cfg := ProviderConfig{ID: "openai", EnvKeys: []string{"OPENAI_API_KEY"}}
	if _, ok := resolveEnvProvider(cfg, func(string) (string, bool) { return "", false }); ok {
		t.Fatal("expected ok=false with no env key")
	}
}

func TestResolveEnvProvider_BaseURLOverride(t *testing.T) {
	env := map[string]string{
		"ZAI_API_KEY":  "k",
		"ZAI_BASE_URL": "https://api.z.ai/api/coding/paas/v4",
	}
	lookup := func(k string) (string, bool) { v, ok := env[k]; return v, ok }

	cfg := ProviderConfig{ID: "zai", BaseURL: "https://api.z.ai/api/paas/v4", EnvKeys: []string{"ZAI_API_KEY"}}
	got, ok := resolveEnvProvider(cfg, lookup)
	if !ok {
		t.Fatal("expected authenticated")
	}
	if got.BaseURL != "https://api.z.ai/api/coding/paas/v4" {
		t.Fatalf("BaseURL = %q, want coding-plan override", got.BaseURL)
	}
}

func TestSDK_WithEnv_MultipleAuthenticatedEndpoints(t *testing.T) {
	env := map[string]string{
		"OPENAI_API_KEY":    "a",
		"DEEPSEEK_API_KEY":  "b",
		"ANTHROPIC_API_KEY": "c",
		"GEMINI_API_KEY":    "d",
		"KIMI_API_KEY":      "e",
		"ZAI_API_KEY":       "f",
	}
	sdk := New(WithEnv(func(k string) (string, bool) { v, ok := env[k]; return v, ok }))

	got := sdk.Providers()
	if len(got) != 6 {
		t.Fatalf("Providers() = %d entries, want 6", len(got))
	}
	wantOrder := []string{"openai", "gemini", "deepseek", "zai", "kimi", "anthropic"}
	for i, p := range got {
		if p.ID() != wantOrder[i] {
			t.Errorf("Providers()[%d] = %s, want %s", i, p.ID(), wantOrder[i])
		}
	}
}

func TestSDK_WithEnv_ExplicitKeyBeatsEnv(t *testing.T) {
	env := map[string]string{"DEEPSEEK_API_KEY": "from-env"}
	sdk := New(
		WithEnv(func(k string) (string, bool) { v, ok := env[k]; return v, ok }),
		WithProvider("deepseek", WithAPIKey("explicit")),
	)
	// WithProvider runs after WithEnv here, so it wins by ordering; check
	// the documented pre-env case too.
	env2 := map[string]string{"KIMI_API_KEY": "from-env"}
	sdk2 := New(
		WithProvider("kimi", WithAPIKey("explicit")),
		WithEnv(func(k string) (string, bool) { v, ok := env2[k]; return v, ok }),
	)
	p2, err := sdk2.Provider("kimi")
	if err != nil {
		t.Fatal(err)
	}
	if got := mustConfigT(p2, t).APIKey; got != "explicit" {
		t.Fatalf("kimi key = %q, want explicit (explicit must survive later FromEnv)", got)
	}
	_ = sdk
}

func TestSDK_ChatUnknownProvider(t *testing.T) {
	sdk := New()
	if _, err := sdk.Chat("nope", "m"); err == nil {
		t.Fatal("expected ConfigError for unknown provider")
	} else if _, ok := err.(*ConfigError); !ok {
		t.Fatalf("error type = %T, want *ConfigError", err)
	}
}

func TestSDK_ChatUnauthenticatedProvider(t *testing.T) {
	sdk := New() // no env, no keys
	_, err := sdk.Chat("openai", "gpt-5.6")
	if err == nil {
		t.Fatal("expected ConfigError for unauthenticated provider")
	}
	if _, ok := err.(*ConfigError); !ok {
		t.Fatalf("error type = %T, want *ConfigError", err)
	}
}

// mustConfig is a test helper reading a provider's config.
func mustConfigT(p *Provider, t *testing.T) ProviderConfig {
	t.Helper()
	if p == nil {
		t.Fatal("nil provider")
	}
	return p.cfg
}
