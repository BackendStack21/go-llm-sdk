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
- **Production streaming** — SSE with an idle watchdog and a hard wall-clock deadline, abort-with-partial-result, retries that never duplicate partial output, premature-close detection, and learn-once fallbacks for providers that reject `stream_options`, streaming, or `reasoning_effort`+tools (GPT-5.6+ retries on `/v1/responses` so reasoning stays on).
- **Predictable under load** — goroutine-leak-free streaming, race-clean shared state, and a canonical-only error vocabulary (API keys never leak into error text).

## Install

```bash
go get github.com/BackendStack21/go-llm-sdk@v0.3.2
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
	case llm.DeltaToolArgs:  // tool-call argument fragment (d.ToolID, d.ToolName; d.ToolIndex = position in res.ToolCalls)
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

Extra HTTP headers ride every request a provider makes — chat, streaming, model listing, speech, transcription, embeddings. They are applied last, so they can override the SDK's own; an empty value removes a header. Values are treated like keys: never logged, never in `String()` or errors.

```go
llm.WithProvider("openrouter", llm.WithFormat(llm.FormatOpenAI),
	llm.WithBaseURL("https://openrouter.ai/api/v1"), llm.WithEnvKeys("OPENROUTER_API_KEY"),
	llm.WithHeaders(map[string]string{"HTTP-Referer": "https://myapp.example", "X-Title": "myapp"}))

llm.WithProvider("anthropic", llm.WithHeaders(map[string]string{"anthropic-beta": "interleaved-thinking-2025-05-14"}))

// api-key gateways (e.g. Azure OpenAI): drop the Bearer header, send api-key.
llm.WithProvider("azure", llm.WithFormat(llm.FormatOpenAI), llm.WithBaseURL(azureURL), llm.WithAPIKey(key),
	llm.WithHeaders(map[string]string{"Authorization": "", "api-key": key}))
```

Redirects to another host are never followed (the redirect response surfaces as an `*APIError`): Go strips only `Authorization` across hosts, so `x-api-key`, `x-goog-api-key` and custom headers would otherwise leak to the target. `Provider` and `ProviderConfig` redact the key and header values under every `fmt` verb (`%v`, `%+v`, `%#v`).

For request/response observability (logging, metrics, tracing), wrap the transport: `llm.WithTransport(myRoundTripper)` sees every HTTP exchange the SDK makes.

## Canonical API

Requests and results are provider-neutral. Unknown message roles are rejected at the SDK boundary (never silently dropped or reinterpreted).

User messages and tool results may contain ordered text and inline image parts.
`Content` remains the backwards-compatible plain-text form; use `TextPart` and
`ImagePart` when an image is needed. Parts are user- and tool-role only and
cannot be combined with `Content` on the same message; empty text parts are
rejected. Accepted image
types are png, jpeg, gif, and webp (the informal `image/jpg` alias is accepted
and normalized to `image/jpeg` on the wire). Per-image (`MaxImageBytes`, 10 MiB)
and per-request aggregate (`MaxRequestImageBytes`, 32 MiB) caps are enforced
before any network I/O. All validation failures are typed `*ConfigError`.

```go
req := &llm.ChatRequest{Messages: []llm.Message{{
	Role: llm.RoleUser,
	Parts: []llm.ContentPart{
		llm.TextPart("Describe this image:"),
		llm.ImagePart("image/png", pngBytes),
	},
}}}
```

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
	Stop           []string       // provider-native stop / stop_sequences / stopSequences (not sent on /responses)
	ToolChoice        *ToolChoice     // nil = provider default; see "Request controls"
	ParallelToolCalls *bool           // with Tools: allow/forbid several calls per turn
	ResponseFormat    *ResponseFormat // JSON object / JSON Schema structured output
	Seed              *int            // OpenAI chat completions, Gemini
	Extra             map[string]any  // provider-specific top-level body fields (override the SDK's)
}

type ChatResult struct {
	Content           string
	ReasoningContent  string          // provider thinking text; replay rules per provider in "Extended thinking"
	ThinkingSignature string          // last signature (Anthropic / Responses / Gemini text part)
	ThinkingBlocks    []ThinkingBlock // every signed reasoning segment, in order (Anthropic, Responses)
	ToolCalls         []ToolCall      // {ID, Name, Arguments, Signature (Gemini thoughtSignature)}
	FinishReason      string          // stop | length | tool_calls | content_filter | ""
	Usage             Usage           // see "Usage" below
}
```

Finish reasons are canonical: anything a provider reports outside that vocabulary maps to `""` (unknown) rather than leaking provider-specific strings. A turn that requested tools finishes as `tool_calls` on every format (Gemini reports `STOP` for it; the SDK maps it).

Signatures (`ThinkingSignature`, `ThinkingBlocks`, `ToolCall.Signature`) are provider-specific: replay them only to the provider that produced them — never carry a conversation's signed turns across providers.

`res.AssistantMessage()` returns the assistant turn to append before the next request, carrying every replay field (content, reasoning, signatures, thinking blocks, tool calls) — use it instead of copying fields by hand.

### Usage

One meaning on every format:

- `PromptTokens` — uncached input only. Inclusive provider counts (OpenAI `cached_tokens`, DeepSeek `prompt_cache_hit_tokens`, Gemini `cachedContentTokenCount`) are moved to `CacheReadTokens`.
- `CacheReadTokens` / `CacheCreationTokens` — cache reads and writes. A DeepSeek cache *miss* is ordinary input, not a write, and stays in `PromptTokens`.
- `CompletionTokens` — every generated token, reasoning included (Gemini thoughts are added); `ReasoningTokens` is that subset.
- `CachedTokens` — the raw provider-reported cached count, for diagnostics; already inside `CacheReadTokens`, never add it.
- `InputTokens()` = prompt + cache reads + cache writes; `TotalTokens()` = input + completion. Use these instead of summing by hand.

## Request controls

```go
req := &llm.ChatRequest{
	Messages: msgs,
	Tools:    tools,
	ToolChoice:        llm.ToolChoiceNamed("lookup"),         // or &llm.ToolChoice{Mode: llm.ToolChoiceRequired}
	ParallelToolCalls: &no,
	ResponseFormat:    &llm.ResponseFormat{Type: llm.ResponseJSONSchema, Name: "answer", Schema: schema, Strict: true},
	Seed:              &seed,
	Extra:             map[string]any{"service_tier": "flex"},
}
```

| Control | OpenAI chat / Responses | Anthropic | Gemini |
|---|---|---|---|
| `ToolChoice` auto / none / required / tool | `tool_choice` | `tool_choice` auto / none / any / tool | `toolConfig.functionCallingConfig` AUTO / NONE / ANY (+ `allowedFunctionNames`) |
| `ParallelToolCalls` | `parallel_tool_calls` (only with tools) | `disable_parallel_tool_use` | — (ignored) |
| `ResponseFormat` json_object / json_schema | `response_format` / `text.format` | forced synthetic tool whose input is folded back into `Content` (buffered and streamed as `DeltaContent`) | `responseMimeType` + `responseJsonSchema` |
| `Seed` | `seed` (chat completions only) | — (ignored) | `generationConfig.seed` |
| `Extra` | merged into the body | merged into the body | merged into the body |

Caveats: DeepSeek accepts `json_object` only (`Quirks.NoJSONSchema`; `json_schema` fails fast with a `ConfigError` — set the flag via `WithQuirks` for other gateways with the same limit). On Anthropic, JSON mode forces its own tool, so it cannot be combined with `Tools` and needs a top-level `"type": "object"` schema. Gemini 2.x models reject JSON mode together with function declarations; Gemini 3 accepts it. `Extra` is a shallow, top-level override: an object value replaces the SDK's whole object (e.g. `generationConfig`), and keys go to whichever endpoint the request lands on (including `/responses` for diverted OpenAI calls). It may not replace what the SDK validates (`model`, `messages`/`contents`/`input`, `system`/`systemInstruction`/`instructions`, `tools`) or `stream`. Gemini seeds must fit in 32 bits.

Controls are validated before any network I/O (`*ConfigError`): `ToolChoice` needs `Tools` and a known tool name; `json_schema` needs a JSON object schema; names match `[A-Za-z0-9_-]{1,64}`; `Extra` may not set the reserved keys above. Anthropic rejects forced tool use under extended thinking, so `ToolChoice` required/tool and JSON `ResponseFormat` fail fast with `Thinking` on, and JSON mode cannot be combined with `ToolChoice` (it forces its own tool).

## Streaming semantics

`CallStream` enforces four guarantees, each covered by regression tests:

1. **Idle watchdog** — a stream silent longer than the idle timeout (120s default; per SDK via `WithStreamIdleTimeout`, process-wide default via the race-safe `SetStreamIdleTimeout`; positive values only) fails with `ErrIdleTimeout`. Keepalive comments reset it.
2. **Hard wall-clock deadline** — the whole stream is bounded by the per-request timeout (`WithRequestTimeout`, default 120s; per-client via `SetRequestTimeout`).
3. **Retries never duplicate output** — retries happen only before the first emitted delta. A failure after partial output returns the partial `*ChatResult` plus a wrapped error and is never retried.
4. **No silent truncation** — a provider that closes the stream before any completion signal (`[DONE]`, a `finish_reason`, `message_stop`, a Gemini `finishReason`, `response.completed`) yields an error: retryable before the first delta, the partial result plus a wrapped error after it. Gemini is no longer exempt. An unmapped provider finish reason still counts as completion.

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
	req.Messages = append(req.Messages, res.AssistantMessage()) // every replay field
	for _, tc := range res.ToolCalls {
		out, err := execute(tc)
		req.Messages = append(req.Messages,
			llm.Message{Role: llm.RoleTool, ToolCallID: tc.ID, ToolName: tc.Name, Content: out, IsError: err != nil})
	}
}
```

`IsError` marks a failed tool execution (Anthropic `is_error`, Gemini `{"error": …}`; OpenAI formats have no flag, so describe the failure in `Content`). Tool results may carry image `Parts`: Anthropic and the Responses API take them natively, Gemini sends them as `inlineData` in the same turn, and Chat Completions (whose tool messages are text-only) receives them in a user message right after the tool results.

On Gemini, a tool result's `ToolName` may be omitted — the SDK recovers the function name from the assistant `ToolCall` it answers, and errors loudly if it cannot.

## Extended thinking

- **OpenAI GPT-5.6+** — function tools plus reasoning cannot ride Chat Completions (`reasoning_effort` 400s). Those calls go to `POST /v1/responses` with `reasoning.effort` and `reasoning.summary=auto`; summaries land in `ReasoningContent`, every reasoning item lands in `ThinkingBlocks` (encrypted content as `Signature`) and is replayed in order; the legacy `ThinkingSignature` carries the last one. `Thinking: disabled` stays on Chat Completions with `reasoning_effort: none`. Other OpenAI models keep `reasoning_effort` on Chat Completions.
- **Anthropic** — every `thinking` and `redacted_thinking` block is parsed in both buffered and streaming modes into `ChatResult.ThinkingBlocks`, each with its own signature (redacted blocks keep their opaque `data`). For tool loops, replay them via `Message.ThinkingBlocks` (or `res.AssistantMessage()`): the SDK re-serializes them first, in order, as Anthropic's API requires. The legacy single pair (`ReasoningContent` + `ThinkingSignature`) still works for one-block turns. Unsigned thinking replay is rejected locally with `ConfigError`. With thinking on, `max_tokens` always exceeds `budget_tokens` (unset `MaxTokens` → budget + 8192; a preset larger than half an explicit `MaxTokens` is clamped to half of it — never below 1024 — so the answer keeps room; an explicit `ThinkingBudget` that cannot fit is a `ConfigError`), and `temperature` / `top_p` are omitted (extended thinking rejects them).
- **DeepSeek** — reasoning streams as `DeltaReasoning` fragments and lands in `ReasoningContent`. Assistant-turn replay echoes it as `reasoning_content`, which DeepSeek requires for tool loops. On a chat-completions request that carries tools, the key is echoed for **every** assistant turn, even where the provider returned no reasoning for that turn (empty value instead of a dropped key) — including turns that made no tool call themselves: DeepSeek documents that a tool-bearing request whose assistant turns omit `reasoning_content` returns 400 for *every* later request in the loop, so one elided turn would otherwise poison the rest of the session. Whether DeepSeek accepts a present-but-empty value is measured against the live API by `TestE2EDeepSeekEmptyReasoningEcho` (tag-gated), not assumed by the suite.
- **zai / GLM** — same reasoning field shape and `DeltaReasoning` streaming; GLM maps thinking `medium` → `reasoning_effort` `high` (no medium level) and `max` → `max`. GLM does **not** get the empty echo: no z.ai documentation confirms the same replay requirement, so `Quirks.EchoReasoningWithTools` is off and `zai` bodies are byte-identical to the pre-fix shape. A custom provider registered under its own id inherits **no** quirks — set `WithQuirks` explicitly if it needs the echo.
- **Gemini** — a prompt blocked by safety (`promptFeedback.blockReason`, no candidates) completes with `content_filter` instead of erroring. `thought: true` parts map to reasoning deltas; `thinkingConfig` is derived from `Thinking` / `ThinkingBudget`. `thoughtSignature`s are captured per function call (`ToolCall.Signature`) and on text parts (`ThinkingSignature`) and replayed on the same parts — Gemini 3 rejects a function-call turn whose signature is missing.
- **OpenAI-format gateways (OpenRouter, LiteLLM, vLLM)** — responses are accepted in every documented reasoning shape: `reasoning_content` (LiteLLM standardized), the `reasoning` string alias (OpenRouter), and the typed `reasoning_details` array (OpenRouter; `reasoning.text`/`reasoning.summary` fold in order, `reasoning.encrypted` entries are skipped). Precedence: `reasoning_content`, then `reasoning`, then the `reasoning_details` fold. Gateways that require an explicit opt-in get it via `Quirks.IncludeReasoning`, which sends the OpenRouter-documented legacy `include_reasoning: true` (equivalent to the canonical `reasoning: {}`) next to `reasoning_effort`. The flag is off by default — strict OpenAI-format endpoints reject unknown parameters.

`ThinkingBudget`, when positive, overrides the selected non-disabled thinking preset (Anthropic enforces its 1024-token minimum). Canonical `max` selects the highest portable preset: OpenAI `high`, Gemini 24576, Anthropic 16384; GLM retains its native `max`.

## Learn-once fallbacks

When a provider rejects a request pattern, the SDK learns the constraint **once per provider** (shared across every `ChatClient` you mint) and never re-pays the failed round-trip:

| Trigger (provider 400) | Learned fallback |
|---|---|
| Rejects `stream_options` | omit `stream_options` from streaming requests |
| Names `/v1/responses` as the tools+reasoning path | retry on `POST /responses` (keeps reasoning on) |
| Rejects `reasoning_effort` + tools (legacy) | pin `reasoning_effort: "none"` |
| Rejects streaming itself | downgrade to buffered calls permanently |
| Answers a streamed request with a non-SSE body | consume that JSON response directly, then use buffered calls permanently (no duplicate generation) |

Every engagement is observable: `WithLearnObserver(func(LearnEvent))` registers a per-SDK callback (taking precedence over the process-wide `SetLearnObserver`) fired once per engaged fallback, with the kind (`buffered`, `responses`, `none_effort`, `drop_stream_options`), provider id, triggering HTTP status, and the provider's error message. Nil (default) disables observation. The callback runs on the request goroutine — keep it fast and never call back into the SDK from it.

## Retry policy

8 attempts, exponential backoff capped at 30s with ±20% jitter (both configurable per SDK with `WithRetryPolicy(llm.RetryPolicy{MaxAttempts, MaxBackoff})`; `MaxAttempts: 1` disables retries), `Retry-After` (seconds or HTTP-date) honored and capped at 120s, context cancellation honored between and during attempts. Retryable statuses include 408/429/5xx plus Cloudflare 520–524 and Anthropic 529. Persistent 429s surface as `*llm.RateLimitError{Attempts, RetryAfter}` — including when the retry sleep is cut short by a deadline, so the caller never loses the retry signal. A 429 whose body is billing exhaustion (`insufficient_quota`, `exceeded your current quota`, `insufficient balance`, `no resource package`) is not retryable and fails on the first attempt. `RateLimitError` unwraps to `*APIError` for `errors.As` access to `Status`/`Retryable`.

## Timeouts & cancellation

- The pooled transport honors `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` via `http.ProxyFromEnvironment`.
- The request timeout (`WithRequestTimeout`, per-client `SetRequestTimeout` — race-safe, swap is atomic) is the wall-clock budget of the **whole call, retries included**, for buffered chat, streaming, speech, transcription and embeddings alike — a failing provider can never hold a caller for timeout × attempts.
- Streaming: per-attempt SSE reads are additionally bounded by the idle watchdog.
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

## Text-to-speech

`Speak` synthesizes speech via a provider's TTS endpoint. Supported wire formats: OpenAI-compatible (`POST {base}/audio/speech`, binary audio) and Gemini (`generateContent` with `AUDIO` response modality, base64 `inlineData`). Other formats return a `ConfigError`.

```go
res, err := sdk.Speak(ctx, "openai", "tts-1", llm.SpeakRequest{
    Text:  "Hello from go-llm-sdk",
    Voice: "alloy",           // required — no local default (never guessed)
    Format: "mp3",            // OpenAI-compat response_format (default "mp3")
    Speed: 1.0,               // optional, OpenAI-compat only
})
// res.Audio    — raw audio bytes exactly as the provider returned them
// res.Model    — the model that produced the audio
// res.MIMEType — the provider's Content-Type, or audio/mpeg when omitted
```

The SDK never transcodes: it returns the provider's bytes plus the MIME type the provider declared, falling back to the wire format's well-known default (`audio/mpeg` on the OpenAI path; Gemini's `inlineData.mimeType` is always present). A 2xx JSON error envelope from a gateway surfaces as a typed `*APIError` — JSON is never returned as audio. Requests carry the same retry ladder and error taxonomy as chat; empty `Text` or `Voice` fail fast with a `ConfigError`.

## Speech-to-text

`Transcribe` converts audio bytes to text via the provider's transcription endpoint. v1 supports the OpenAI-compatible wire format (`POST {base}/audio/transcriptions`, multipart/form-data); other formats return a `ConfigError`. The SDK never touches the filesystem — callers own the audio bytes.

```go
res, err := sdk.Transcribe(ctx, "openai", "whisper-1", llm.TranscribeRequest{
    Audio:    audio,            // required, non-empty, ≤25MB
    Filename: "probe.mp3",      // file part name (default "audio.wav")
    MIMEType: "audio/mpeg",     // audio part content type (default application/octet-stream)
    Language: "en",             // optional ISO-639-1 hint
    Prompt:   "context words",  // optional conditioning text
})
// res.Text        — recognized text
// res.Model       — model that produced the transcription
// res.Language, res.DurationSec — provider-reported, zero when omitted
```

Requests carry the same retry ladder and error taxonomy as chat. Oversized audio (>25MB), empty `Audio`/`Model`, and control characters in `Filename`/`MIMEType` (header injection) fail fast with a `ConfigError` before any network I/O. A 2xx body that is not JSON surfaces as a typed `*APIError`. Fields the provider does not report stay zero — the SDK never guesses.

## Embeddings

`Embed` turns texts into vectors via OpenAI-compatible `POST {base}/embeddings` or Gemini `batchEmbedContents`; Anthropic (no endpoint) returns a `ConfigError`.

```go
res, err := sdk.Embed(ctx, "openai", "text-embedding-3-small", llm.EmbedRequest{
    Inputs:     []string{"first", "second"}, // required, each non-empty
    Dimensions: 256,                         // optional (0 = model default)
})
// res.Embeddings[i] is the vector for Inputs[i]; res.Usage.PromptTokens when reported
```

Vectors come back in input order (the SDK sorts by the provider's `index` — or trusts response order when a gateway omits it — and rejects a count or index mismatch with a typed `*APIError`). Large inputs are split into provider-sized batches (Gemini 100, OpenAI 2048) under one whole-call budget; a `models/` prefix on Gemini model ids is accepted. Same retry ladder, budget, headers and error taxonomy as chat.

## Thread safety

`SDK` and `Provider` are safe for concurrent use; `SetStreamIdleTimeout` and `SetLearnObserver` may be called while requests run. `ChatClient` is safe for concurrent `Call`/`CallStream`; `SetRequestTimeout` is race-safe (atomic swap) but should still be called before the first request so in-flight calls use one timeout. Learn-once state is shared per provider via atomics — monotonic, converging, race-free.

## Model discovery

`ListModels` hits each provider's models endpoint (Anthropic paginates with `after_id`, Gemini with `pageToken`), caches per SDK for 5 minutes (`WithModelCacheTTL(0)` disables, `ForceRefresh()` bypasses), and retries transient failures 3× (a malformed body is not retried). Concurrent cache misses share one upstream fetch. A listing longer than 100 pages returns `ErrModelListTruncated` — never a silently truncated list. Fields the provider does not report stay zero — the SDK never guesses.

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

v0.3.2 — API may shift until v1.0.

## License

[MIT](LICENSE)
