package llm

import (
	"os"
	"strings"
)

// lookupEnv is the default environment reader (os.LookupEnv).
func lookupEnv(key string) (string, bool) { return os.LookupEnv(key) }

// resolveEnvProvider fills APIKey and BaseURL for one provider from the
// environment using lookup. EnvKeys are consulted primary-first; the first
// non-empty value wins (primary beats alias, silently). The base URL can be
// overridden via <ID>_BASE_URL (e.g. ZAI_BASE_URL) for coding-plan
// endpoints and self-hosted gateways. Returns ok=false when no key is set —
// the provider stays unauthenticated and is excluded from Providers().
func resolveEnvProvider(cfg ProviderConfig, lookup func(string) (string, bool)) (ProviderConfig, bool) {
	if lookup == nil {
		return cfg, false
	}
	for _, k := range cfg.EnvKeys {
		if v, ok := lookup(k); ok && strings.TrimSpace(v) != "" {
			cfg.APIKey = strings.TrimSpace(v)
			break
		}
	}
	if cfg.APIKey == "" {
		return cfg, false
	}
	if v, ok := lookup(envBaseURLKey(cfg.ID)); ok && strings.TrimSpace(v) != "" {
		cfg.BaseURL = strings.TrimSpace(v)
	}
	return cfg, true
}
