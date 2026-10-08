package llm

import "testing"

// One canonical meaning across formats:
//   - PromptTokens: uncached input only
//   - CacheReadTokens / CacheCreationTokens: cache volumes (exclusive)
//   - CompletionTokens: every generated token, reasoning included
//   - ReasoningTokens: subset of CompletionTokens
//   - InputTokens() / TotalTokens(): the provider's full totals
func TestUsage_CanonicalAcrossFormats(t *testing.T) {
	type want struct{ prompt, read, create, completion, reasoning, input, total int }
	check := func(name string, u Usage, w want) {
		t.Helper()
		if u.PromptTokens != w.prompt || u.CacheReadTokens != w.read || u.CacheCreationTokens != w.create ||
			u.CompletionTokens != w.completion || u.ReasoningTokens != w.reasoning ||
			u.InputTokens() != w.input || u.TotalTokens() != w.total {
			t.Errorf("%s: usage = %+v input=%d total=%d, want %+v", name, u, u.InputTokens(), u.TotalTokens(), w)
		}
	}

	// DeepSeek: a cache miss is ordinary (uncached) input, not a cache write.
	check("deepseek", usageFromOpenAI(&oaRespUsage{
		PromptTokens: 1000, CompletionTokens: 40, PromptCacheHitTokens: 750, PromptCacheMissTokens: 250,
	}), want{prompt: 250, read: 750, completion: 40, input: 1000, total: 1040})

	// DeepSeek may echo the hit count as cached_tokens too: never twice.
	check("deepseek+details", usageFromOpenAI(&oaRespUsage{
		PromptTokens: 1000, CompletionTokens: 40, PromptCacheHitTokens: 750, PromptCacheMissTokens: 250,
		PromptTokensDetails: &oaPromptDetails{CachedTokens: 750},
	}), want{prompt: 250, read: 750, completion: 40, input: 1000, total: 1040})

	// OpenAI: cached_tokens is a subset of prompt_tokens.
	check("openai", usageFromOpenAI(&oaRespUsage{
		PromptTokens: 300, CompletionTokens: 30, PromptTokensDetails: &oaPromptDetails{CachedTokens: 200},
		CompletionTokensDetails: oaUsageDetails{ReasoningTokens: 10},
	}), want{prompt: 100, read: 200, completion: 30, reasoning: 10, input: 300, total: 330})

	// Gemini: cachedContentTokenCount is a subset of promptTokenCount and
	// candidatesTokenCount excludes thoughts.
	cached := 30
	g := mapGeminiUsage(gmUsage{PromptTokenCount: 100, CandidatesTokenCount: 10, ThoughtsTokenCount: 5, CachedContentTokenCount: &cached})
	check("gemini", g, want{prompt: 70, read: 30, completion: 15, reasoning: 5, input: 100, total: 115})
	if !g.CacheReported {
		t.Error("gemini: CacheReported must be true when cachedContentTokenCount is present")
	}
	if g := mapGeminiUsage(gmUsage{PromptTokenCount: 5}); g.CacheReported {
		t.Error("gemini: no cache field → CacheReported false")
	}

	// Anthropic reports cache volumes exclusively.
	check("anthropic", usageFromAnthropic(anUsage{InputTokens: 10, OutputTokens: 7, CacheReadTokens: 20, CacheCreationTokens: 5}),
		want{prompt: 10, read: 20, create: 5, completion: 7, input: 35, total: 42})
}

// Gemini usage flows through buffered parsing.
func TestParseGeminiResponse_CachedUsage(t *testing.T) {
	res, err := parseGeminiResponse([]byte(`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":10,"thoughtsTokenCount":5,"cachedContentTokenCount":30}}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.PromptTokens != 70 || res.Usage.CacheReadTokens != 30 || res.Usage.CompletionTokens != 15 {
		t.Errorf("usage = %+v", res.Usage)
	}
}

// Anthropic's message_delta carries cumulative usage (cache volumes may
// only arrive there); non-zero values update the accumulator.
func TestMapAnthropicStreamEvent_MessageDeltaUsage(t *testing.T) {
	acc := newStreamAccum()
	for _, e := range []string{
		`{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7,"input_tokens":10,"cache_read_input_tokens":20,"cache_creation_input_tokens":5}}`,
	} {
		if _, _, err := mapAnthropicStreamEvent([]byte(e), acc); err != nil {
			t.Fatal(err)
		}
	}
	u := acc.result().Usage
	if u.PromptTokens != 10 || u.CompletionTokens != 7 || u.CacheReadTokens != 20 || u.CacheCreationTokens != 5 || !u.CacheReported {
		t.Errorf("usage = %+v", u)
	}
	// A delta that omits input/cache fields keeps message_start's values.
	acc = newStreamAccum()
	_, _, _ = mapAnthropicStreamEvent([]byte(`{"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":3}}}`), acc)
	_, _, _ = mapAnthropicStreamEvent([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`), acc)
	if u := acc.result().Usage; u.PromptTokens != 10 || u.CacheReadTokens != 3 || u.CompletionTokens != 7 {
		t.Errorf("sparse delta usage = %+v", u)
	}
}
