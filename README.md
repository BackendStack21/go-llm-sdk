# go-llm-sdk

[![CI](https://github.com/BackendStack21/go-llm-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/BackendStack21/go-llm-sdk/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/BackendStack21/go-llm-sdk.svg)](https://pkg.go.dev/github.com/BackendStack21/go-llm-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Go](https://img.shields.io/badge/Go-1.25%2B-00ADD8)

Multi-provider Go SDK for LLM inference endpoints — **OpenAI, Google Gemini, DeepSeek, Z.ai, Kimi (Moonshot) and Anthropic**, plus any OpenAI-compatible gateway. Stdlib only; zero external dependencies.

- **Multiple authenticated endpoints at once** — auto-discovered from `<PROVIDER>_API_KEY` environment variables (aliases supported).
- **Dynamic model discovery** — `ListModels` returns what the account can actually access. No static model tables.
- **One canonical API** — OpenAI-shaped requests and responses; Anthropic and Gemini wire formats are translated for you.
- **Portable generation controls** — token limits, temperature, top-p, stop sequences, thinking, and tools map to each provider's native fields.
- **Production streaming** — SSE with an idle watchdog and a hard wall-clock deadline, abort-with-partial-result, retries that never duplicate partial output, premature-close detection, and learn-once fallbacks for providers that reject `stream_options`, streaming, or `reasoning_effort`+tools.
- **Predictable under load** — goroutine-leak-free streaming, race-clean shared state, and a canonical-only error vocabulary (API keys never leak into error text).

## Install

```bash
go get github.com/BackendStack21/go-llm-sdk@v0.2.2
```

Requires Go 1.25+. No dependencies beyond the standard library.

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"

	llm "github.com/BackendStack21/go-llm-sdk"
)

func main() {
	ctx := context.Background()
	sdk := llm.New(llm.FromEnv()) // OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY,
	                             // ZAI_API_KEY, KIMI_API_KEY, ANTHROPIC_API_KEY (+ aliases)

	for _, p := range sdk.Providers() { // authenticated endpoints, registry order
		models, err := p.ListModels(ctx)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s: %d models\n", p.ID(), len(models))
		// models[i] → ID, DisplayName, CreatedAt, ContextWindow, MaxOutputTokens, Capabilities
	}

	chat, err := sdk.Chat("deepseek", "deepseek-chat")
	if err != nil {
		log.Fatal(err)
	}

	res, err := chat.Call(ctx, &llm.ChatRequest{
		System:   []llm.SystemBlock{{Text: "Be terse."}},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello"}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Content)
}
```

Streaming:

```go
res, err = chat.CallStream(ctx, req, func(d llm.Delta) error {
	switch d.Kind {
	case llm.DeltaReasoning: // thinking fragment
	case llm.DeltaContent:   // text fragment
	case llm.DeltaToolArgs:  // tool-call argument fragment (d.ToolID, d.ToolName)
	}
	return nil // or an error to abort — partial result comes back with *StreamAbortedError
})
```

## Providers

| ID | Format | Default base URL | Env var (alias) | Base-URL override |
|---|---|---|---|---|
| `openai` | openai | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| `gemini` | gemini | `https://generativelanguage.googleapis.com` | `GEMINI_API_KEY` (`GOOGLE_API_KEY`) | `GEMINI_BASE_URL` |
| `deepseek` | openai | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| `zai` | openai | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` | `ZAI_BASE_URL` (e.g. coding-plan endpoint) |
| `kimi` | openai | `https://api.moonshot.ai/v1` | `KIMI_API_KEY` (`MOONSHOT_API_KEY`) | `KIMI_BASE_URL` |
| `anthropic` | anthropic | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |

Primary env var beats its alias. Explicit keys (`WithAPIKey`) beat env. Base-URL overrides accept any gateway speaking the provider's format. A bad URL passed to `WithBaseURL` is rejected at wiring time: the provider is marked invalid, `Providers()` omits it, and `Chat` returns a `*ConfigError` — no request is sent.

Custom gateways:

```go
sdk := llm.New(llm.WithProvider("my-gateway",
	llm.WithFormat(llm.FormatOpenAI),
	llm.WithBaseURL("http://localhost:11434/v1"),
	llm.WithAPIKey("local"),
))
```

## Canonical API

Requests and results are provider-neutral. Unknown message roles are rejected at the SDK boundary (never silently dropped or reinterpreted).

```go
type ChatRequest struct {
	Model          string         // optional; ChatClient's model wins when both set
	Messages       []Message      // RoleUser | RoleAssistant | RoleSystem | RoleTool; Message.Cache → Anthropic user-block cache_control
	System         []SystemBlock  // {Text, Cache} — Cache marks Anthropic prompt-cache blocks
	Tools          []ToolDef      // {Name, Description, Parameters json.RawMessage}
	Thinking       string         // "", "enabled", "disabled", "low", "medium", "high", "max"
	ThinkingBudget int            // explicit token budget where the provider supports it
	MaxTokens      int            // routed to max_completion_tokens on o-series/gpt-5
	Temperature    float64        // 0 = provider default; negative = explicit 0
	TopP           float64        // 0 = provider default; negative = explicit 0
	Stop           []string       // provider-native stop / stop_sequences / stopSequences
}

type ChatResult struct {
	Content           string
	ReasoningContent  string      // provider thinking text; replayed as reasoning_content on OpenAI-format assistant turns
	ThinkingSignature string      // Anthropic: replay via Message.ThinkingSignature
	ToolCalls         []ToolCall  // {ID, Name, Arguments}
	FinishReason      string      // stop | length | tool_calls | content_filter | ""
	Usage             Usage       // PromptTokens is uncached-only; cache volumes in CacheReadTokens / CacheCreationTokens / CachedTokens
}
```

Finish reasons are canonical: anything a provider reports outside that vocabulary maps to `""` (unknown) rather than leaking provider-specific strings.

## Streaming semantics

`CallStream` enforces four guarantees, each covered by regression tests:

1. **Idle watchdog** — a stream silent longer than `StreamIdleTimeout()` (120s default; override with `SetStreamIdleTimeout`, positive values only) fails with `ErrIdleTimeout`. Keepalive comments reset it.
2. **Hard wall-clock deadline** — the whole stream is bounded by the per-request timeout (`WithRequestTimeout`, default 120s; per-client via `SetRequestTimeout`).
3. **Retries never duplicate output** — retries happen only before the first emitted delta. A failure after partial output returns the partial `*ChatResult` plus a wrapped error and is never retried.
4. **No silent empty successes** — a provider that closes the stream before its completion signal (before `[DONE]` / `message_stop`) yields a retryable error, not an empty result. Gemini, whose streams legitimately end at EOF, is exempt.

Aborting from the delta handler returns the partial result alongside `*StreamAbortedError` — the parser goroutine is always released, so aborted streams leak nothing.

### Tool-call loop

```go
for {
	res, err := chat.CallStream(ctx, req, func(d llm.Delta) error { return nil })
	if err != nil {
		return err
	}
	if len(res.ToolCalls) == 0 {
		return nil
	}
	req.Messages = append(req.Messages,
		llm.Message{Role: llm.RoleAssistant, Content: res.Content,
			ReasoningContent: res.ReasoningContent, ThinkingSignature: res.ThinkingSignature,
			ToolCalls: res.ToolCalls})
	for _, tc := range res.ToolCalls {
		req.Messages = append(req.Messages,
			llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: tc.Name, Content: execute(tc)})
	}
}
```

On Gemini, a tool result's `ToolName` may be omitted — the SDK recovers the function name from the assistant `ToolCall` it answers, and errors loudly if it cannot.

## Extended thinking

- **Anthropic** — `thinking` blocks are parsed in both buffered and streaming modes. `ChatResult.ThinkingSignature` carries the provider signature; for tool loops, replay it on the assistant message (`Message.ReasoningContent` + `Message.ThinkingSignature`) — the SDK re-serializes it as the first block, as Anthropic's API requires. Omitting it makes extended-thinking tool loops fail mid-conversation.
- **DeepSeek / GLM** — reasoning streams as `DeltaReasoning` fragments and lands in `ReasoningContent`. Assistant-turn replay echoes it as `reasoning_content` (required for DeepSeek/GLM tool loops). GLM maps thinking `medium` → `reasoning_effort` `high` (no medium level) and `max` → `max`.
- **Gemini** — `thought: true` parts map to reasoning deltas; `thinkingConfig` is derived from `Thinking` / `ThinkingBudget`.

## Learn-once fallbacks

When a provider rejects a request pattern, the SDK learns the constraint **once per provider** (shared across every `ChatClient` you mint) and never re-pays the failed round-trip:

| Trigger (provider 400) | Learned fallback |
|---|---|
| Rejects `stream_options` | omit `stream_options` from streaming requests |
| Rejects `reasoning_effort` + tools | pin `reasoning_effort: "none"` |
| Rejects streaming itself | downgrade to buffered calls permanently |
| Answers a streamed request with a non-SSE body | downgrade to buffered calls permanently |

## Retry policy

8 attempts, exponential backoff capped at 30s with ±20% jitter, `Retry-After` (seconds or HTTP-date) honored and capped at 120s, context cancellation honored between and during attempts. Retryable statuses include 408/429/5xx plus Cloudflare 520–524 and Anthropic 529. Persistent 429s surface as `*llm.RateLimitError{Attempts, RetryAfter}` — including when the retry sleep is cut short by a deadline, so the caller never loses the retry signal. A 429 whose body is billing exhaustion (`insufficient_quota`, `exceeded your current quota`, `insufficient balance`, `no resource package`) is not retryable and fails on the first attempt. `RateLimitError` unwraps to `*APIError` for `errors.As` access to `Status`/`Retryable`.

## Timeouts & cancellation

- The pooled transport honors `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` via `http.ProxyFromEnvironment`.
- Buffered calls: per-request timeout on the HTTP client (`WithRequestTimeout`, per-client `SetRequestTimeout` — race-safe, swap is atomic).
- Streaming: the same timeout becomes the hard wall-clock deadline via context; per-attempt SSE reads are additionally bounded by the idle watchdog.
- Every wait (backoff, Retry-After, stream reads) selects on the caller's context — cancellation propagates everywhere, and a cancelled call never misreports as "retry exhausted".
- Response bodies are capped (50 MB chat, 8 MB listings, 1 MiB SSE lines, 4 MiB SSE events) as an OOM bound.

## Error handling

`*ConfigError` (unknown/unauthenticated provider, invalid wiring, unknown role, no model), `*APIError{Provider, Status, Code, Message, Retryable}`, `*RateLimitError{Attempts, RetryAfter}` (unwraps to `*APIError`), `*StreamAbortedError` (returned together with the partial `*ChatResult`). A stream failure after partial output returns the partial `*ChatResult` plus a wrapped error and is never retried; the idle watchdog surfaces as `ErrIdleTimeout` (retried only before the first delta); wall-clock deadlines surface as context deadline errors. Recommended classification:

```go
var abort *llm.StreamAbortedError
var rl *llm.RateLimitError
var ae *llm.APIError
switch {
case errors.As(err, &abort):                 // consumer abort (partial result returned)
case errors.As(err, &rl):                    // back off rl.RetryAfter
case errors.As(err, &ae):                    // provider said no (ae.Status)
case errors.Is(err, llm.ErrIdleTimeout):     // stream went silent
case errors.Is(err, context.DeadlineExceeded): // wall-clock budget spent
}
```

API keys never appear in any error text. Provider error bodies are parsed per format (nested OpenAI envelope, Anthropic `error.type/message`, Gemini `error.status/message`) with a 512-byte raw-body fallback.

## Thread safety

`SDK` and `Provider` are safe for concurrent use. `ChatClient` is safe for concurrent `Call`/`CallStream`; `SetRequestTimeout` is race-safe (atomic swap) but should still be called before the first request so in-flight calls use one timeout. Learn-once state is shared per provider via atomics — monotonic, converging, race-free.

## Model discovery

`ListModels` hits each provider's models endpoint (Anthropic paginates with `after_id`, Gemini with `pageToken`), caches per SDK for 5 minutes (`WithModelCacheTTL(0)` disables, `ForceRefresh()` bypasses), and retries transient failures 3×. Fields the provider does not report stay zero — the SDK never guesses.

## Testing

```bash
make quality    # fmt + vet + tests
make test-race  # race detector
make lint       # golangci-lint (v2 config)
```

Live end-to-end tests against real APIs (tag-gated, never run in CI). The suite covers every provider you have credentials for — currently DeepSeek, Z.ai, and a custom OpenRouter gateway. Each target skips when its key is absent.

```bash
go test -tags e2e -run 'TestE2E' -timeout 15m -v .
```

Credentials come from the environment or a repo-root `.env` file (`KEY=VALUE`); the file is gitignored and its contents are never logged. Override a target's model with `<ID>_E2E_MODEL` (e.g. `DEEPSEEK_E2E_MODEL`). Adding a provider is one `e2eTarget` entry in `e2e_test.go`.

Coverage sits at **97.7%** of statements, including the streaming failure-orchestration paths (deadline, 429, premature close, partial-output) that are usually the blind spot of SDK test suites. The residual ~2% is unreachable defensive code.

## Repo guidance

See [AGENTS.md](AGENTS.md) for the architecture map, invariants, testing conventions, and the odek migration path.

## Status

v0.2.2 — API may shift until the odek integration lands, then v1.0.

## License

[MIT](LICENSE)
