package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Cache-token exclusive normalization (odek applyUsage parity).
//
// Anthropic reports cache tokens exclusively (input_tokens excludes them).
// OpenAI (prompt_tokens_details.cached_tokens) and DeepSeek
// (prompt_cache_hit_tokens + prompt_cache_miss_tokens = prompt_tokens)
// report them inclusively, as subsets of prompt_tokens.
//
// Usage.PromptTokens must be exclusive ("uncached" input) on every
// provider, with cache volumes carried in CacheReadTokens /
// CacheCreationTokens, so budget enforcement can sum them without
// double-counting. CacheReported is true when any cache field was present.

func TestUsageCache_OpenAICachedTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{
			"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
			"usage": {
				"prompt_tokens": 300,
				"completion_tokens": 30,
				"prompt_tokens_details": {"cached_tokens": 200}
			}
		}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100 (300 prompt − 200 cached; exclusive)", res.Usage.PromptTokens)
	}
	if res.Usage.CacheReadTokens != 200 {
		t.Errorf("CacheReadTokens = %d, want 200 (OpenAI cached_tokens)", res.Usage.CacheReadTokens)
	}
	if res.Usage.CachedTokens != 200 {
		t.Errorf("CachedTokens = %d, want 200 (display field unchanged)", res.Usage.CachedTokens)
	}
	if !res.Usage.CacheReported {
		t.Error("CacheReported = false, want true (prompt_tokens_details present)")
	}
	if res.Usage.CompletionTokens != 30 {
		t.Errorf("CompletionTokens = %d, want 30", res.Usage.CompletionTokens)
	}
}

func TestUsageCache_AnthropicExclusive(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{
			"content": [{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"usage":{
				"input_tokens": 500,
				"output_tokens": 50,
				"cache_creation_input_tokens": 400,
				"cache_read_input_tokens": 100
			}
		}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "anthropic", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Quirks: Quirks{AnthropicVersion: "2023-06-01"}}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	// Anthropic input_tokens is already uncached-only: no subtraction.
	if res.Usage.PromptTokens != 500 {
		t.Errorf("PromptTokens = %d, want 500 (Anthropic is exclusive already)", res.Usage.PromptTokens)
	}
	if res.Usage.CacheCreationTokens != 400 || res.Usage.CacheReadTokens != 100 {
		t.Errorf("cache fields = %d/%d, want 400/100", res.Usage.CacheCreationTokens, res.Usage.CacheReadTokens)
	}
	if !res.Usage.CacheReported {
		t.Error("CacheReported = false, want true (Anthropic cache fields present)")
	}
	if res.Usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", res.Usage.CompletionTokens)
	}
}

func TestUsageCache_DeepSeekHitMiss(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{
			"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
			"usage": {
				"prompt_tokens": 1000,
				"completion_tokens": 40,
				"prompt_cache_hit_tokens": 750,
				"prompt_cache_miss_tokens": 250
			}
		}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "deepseek", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.PromptTokens != 0 {
		t.Errorf("PromptTokens = %d, want 0 (prompt 1000 = hit 750 + miss 250; every token is cache-accounted)", res.Usage.PromptTokens)
	}
	if res.Usage.CacheReadTokens != 750 {
		t.Errorf("CacheReadTokens = %d, want 750", res.Usage.CacheReadTokens)
	}
	if res.Usage.CacheCreationTokens != 250 {
		t.Errorf("CacheCreationTokens = %d, want 250", res.Usage.CacheCreationTokens)
	}
	if !res.Usage.CacheReported {
		t.Error("CacheReported = false, want true (DeepSeek hit/miss present)")
	}
}

func TestUsageCache_HostileCachedTokensNeverNegative(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{
			"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
			"usage": {
				"prompt_tokens": 50,
				"completion_tokens": 5,
				"prompt_tokens_details": {"cached_tokens": 500}
			}
		}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.PromptTokens < 0 {
		t.Errorf("PromptTokens = %d, must never go negative", res.Usage.PromptTokens)
	}
	if res.Usage.PromptTokens != 50 {
		t.Errorf("PromptTokens = %d, want 50 (hostile cached_tokens skipped; no subtraction)", res.Usage.PromptTokens)
	}
	if !res.Usage.CacheReported {
		t.Error("CacheReported = false, want true (details object present)")
	}
}

func TestUsageCache_OpenAIFormatAnthropicCacheFields(t *testing.T) {
	// Some OpenAI-compatible gateways forward Anthropic-named cache fields
	// on the chat-completions usage object. They are exclusive — do not
	// subtract from prompt_tokens.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{
			"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
			"usage": {
				"prompt_tokens": 500,
				"completion_tokens": 50,
				"cache_creation_input_tokens": 400,
				"cache_read_input_tokens": 100
			}
		}`)
	}))
	defer srv.Close()

	cc := newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.Call(context.Background(), &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.PromptTokens != 500 {
		t.Errorf("PromptTokens = %d, want 500 (Anthropic-named fields are exclusive)", res.Usage.PromptTokens)
	}
	if res.Usage.CacheCreationTokens != 400 || res.Usage.CacheReadTokens != 100 {
		t.Errorf("cache fields = %d/%d, want 400/100", res.Usage.CacheCreationTokens, res.Usage.CacheReadTokens)
	}
	if !res.Usage.CacheReported {
		t.Error("CacheReported = false, want true")
	}
}
