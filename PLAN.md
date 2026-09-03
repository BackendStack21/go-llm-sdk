# go-llm-sdk — Design & Implementation Plan

Externalize odek's `internal/llm` into a standalone, multi-provider Go SDK.
Module: `github.com/BackendStack21/go-llm-sdk` · Go 1.25 · **zero external dependencies** (stdlib only).

> Staging note: built in `odek/.plans/go-llm-sdk/` because the authoring session could not
> write to the target repo directly; this file is the in-repo design record.

---

## 1. Goal

A single SDK that talks to **OpenAI, Google Gemini, DeepSeek, Z.ai, Kimi (Moonshot) and Anthropic**
inference endpoints, with:

1. **Multiple authenticated endpoints simultaneously**, auto-discovered from the environment via
   `<PROVIDER>_API_KEY` (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`,
   `KIM_API_KEY`, `ANTHROPIC_API_KEY`).
2. **Dynamic model discovery on the fly** — `ListModels(ctx)` hits each provider's models endpoint
   and returns what the account can actually access. **No static model profile tables** (replaces
   odek's `KnownProfiles` / `ModelProfile`).
3. **Drop-in replacement path for odek** — the API must cover everything odek's agent loop needs:
   buffered + streaming chat, tool calling, reasoning/thinking content, system prompts, retries.

## 2. Non-goals (this workstream)

- **No changes to odek.** The swap `internal/llm` → `go-llm-sdk` is a separate follow-up (§9).
- Chat completions + model discovery only. No embeddings, images, audio, files, batches.
- No ORM-ish DSL — thin, explicit types close to the wire.
- No key management/secrets storage — env + explicit config only; never logged.

## 3. Current state (source inventory — port, don't reinvent)

From `odek/internal/llm` (battle-tested, ~1,500 lines + 250+ tests):

| Concern | Source | Behavior to keep |
|---|---|---|
| OpenAI-compatible client | `client.go` | Pooled transport (port `internal/transport`), 50 MB response cap, URL-sniffed quirks (Anthropic thinking object + top-level `system`; DeepSeek `thinking`; Z.ai GLM `thinking`+`reasoning_effort`), learn-once fallbacks (`forceNoneEffort`, `dropStreamOptions`, `forceBuffered`) |
| Retries | `client.go` | 8 attempts total, exponential backoff capped, `Retry-After` honored, typed `RateLimitError` (satisfies the ≥5-retries requirement) |
| Streaming | `stream.go` | SSE parsing, `Delta{Kind: Reasoning\|Content\|ToolArgs}`, idle watchdog, hard wall-clock deadline, retry only before first delta, `StreamAbortedError`, buffered fallback |
| Model discovery | `models.go` | `GET /models`, multi-field context parsing (`context_length`/`max_context`/`max_input_tokens`), per-endpoint cache — **generalize into `ListModels`** |
| Static profiles | `odek.go KnownProfiles` | **Deleted concept** — replaced by dynamic discovery |

## 4. Architecture

### 4.1 Package layout

```
go-llm-sdk/
  llm.go            SDK entrypoint: New, options, multi-provider aggregate
  provider.go       Provider registry: IDs, API formats, default base URLs, env keys
  auth.go           <PROVIDER>_API_KEY env resolution + aliases
  message.go        Canonical request/response types (OpenAI-shaped)
  chat.go           Buffered chat (canonical) + per-format serialization
  stream.go         Unified streaming (canonical deltas) + SSE readers
  models.go         ListModels: dynamic discovery + unified Model metadata + optional TTL cache
  retry.go          Retry/backoff/RateLimitError (shared across formats)
  transport.go      Pooled http.Client (ported from internal/transport)
  errors.go         Typed errors
  format/
    openai/         OpenAI-compat serializer (passthrough; quirks per provider)
    anthropic/      Canonical ⇄ Anthropic messages translate + SSE
    gemini/         Canonical ⇄ Gemini generateContent translate + SSE
  ..._test.go       Hermetic httptest-based tests per package
```

### 4.2 Provider registry (built-in)

| ID | Format | Default base URL | Env var (alias) |
|---|---|---|---|
| `openai` | openai | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `gemini` | gemini | `https://generativelanguage.googleapis.com` | `GEMINI_API_KEY` (`GOOGLE_API_KEY`) |
| `deepseek` | openai | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `zai` | openai | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `kimi` | openai | `https://api.moonshot.ai/v1` | `KIMI_API_KEY` (`MOONSHOT_API_KEY`) |
| `anthropic` | anthropic | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |

- Custom endpoints: `Register("my-gateway", Format: openai, BaseURL: …, EnvKey: …)` — any
  OpenAI-compatible gateway works without registry surgery.
- `Format` is one of `openai | anthropic | gemini`. **Sniffing moves from URL-strings to explicit
  registry flags resolved at registration time** — a custom base URL gets that format's default
  quirks unless the caller overrides `Quirks` explicitly (proxied Anthropic-via-gateway works by
  registering `Format: anthropic`).

**Per-provider quirk flags** (replaces odek's URL sniffing):

| Provider | ThinkingObject | ReasoningEffort | ForceThinking¹ | Extra headers | Notes |
|---|---|---|---|---|---|
| openai | – | ✓ | – | – | `reasoning_effort` only; rejects unknown top-level fields |
| gemini | native `thinkingConfig` | – | – | `x-goog-api-key` | model in URL path, not body |
| deepseek | ✓ | – | – | – | `reasoning_content` in responses |
| zai | ✓ | ✓ (GLM-5.3+) | ✓ | – | forced-thinking model subset rejects `thinking:disabled` |
| kimi | – | – | – | – | OpenAI passthrough |
| anthropic | ✓ (budget) | – | – | `anthropic-version: 2023-06-01` | top-level `system`, `max_tokens` required |

¹ `ForceThinking` = model-name prefixes that must not be sent `thinking:disabled`.
All formats additionally carry the three learn-once fallbacks (§3) at runtime, per provider
instance, exactly as odek does today.

### 4.3 Canonical message format

OpenAI-compatible shape is the canonical type system (odek's loop already speaks it):
`Message{Role, Content, ReasoningContent, ToolCalls, ToolCallID}`, `ToolDef`,
`ChatRequest{Model, Messages, Tools, Thinking, ThinkingBudget, MaxTokens, Temperature, SystemBlocks}`,
`ChatResult{Content, ReasoningContent, ToolCalls, FinishReason, Usage}`.
Adapters translate to/from Anthropic (`system` top-level, content blocks, `tool_use`/`tool_result`,
`thinking` blocks) and Gemini (`contents`/`parts`, `functionDeclarations`, `functionCall`/`functionResponse`,
`thinkingConfig`).

Thinking control maps per format: `reasoning_effort` (openai) / `thinking:{type,budget_tokens}`
(anthropic, deepseek, GLM) / `generationConfig.thinkingConfig` (gemini; budget -1 = dynamic,
0 = off; effort levels low/medium/high map to budget tiers on formats that only take budgets).

### 4.3.1 Known translation hard edges (must be covered by golden tests)

- **Anthropic**: `max_tokens` required (SDK default 8192 when unset); content blocks may not be
  empty (synthesize a placeholder text block for empty assistant turns); consecutive canonical
  `tool` messages merge into ONE user message with multiple `tool_result` blocks; parallel tool
  calls = multiple `tool_use` blocks in one assistant message; usage split across SSE
  `message_start` (input) + `message_delta` (output).
- **Gemini**: system prompts go to `systemInstruction`; `functionResponse` parts ride `role:"user"`
  per current API docs; model in URL path (`/v1beta/models/{m}:generateContent`); no tool-call IDs
  — SDK synthesizes `call_<n>` ids; `finishReason` mapping (STOP→stop, MAX_TOKENS→length,
  SAFETY/RECISATION→content_filter); reasoning = parts with `thought:true` + `includeThoughts`.
- **Canonical**: multi-part content (images/audio) is **out of scope for v0** and documented as a
  limitation — `Content` is plain text; if a provider returns multi-part, text parts concatenate.

## 4.4 Dynamic model discovery — `ListModels`

```go
type Model struct {
    ID             string
    CreatedAt      time.Time   // when the API reports it
    ContextWindow  int         // tokens, 0 = unknown
    MaxOutputTokens int        // tokens, 0 = unknown
    Capabilities   []string    // e.g. "tools", "vision" — only what the API reports
}

ListModels(ctx) ([]Model, error)
```

- openai format: `GET /models` → `data[].id` + best-effort context fields (`context_length`,
  `max_context`, `max_input_tokens`). Not paginated in practice; tolerate a `models[]` wrapper.
- anthropic: `GET /v1/models` (`x-api-key` + `anthropic-version`) → id/display_name/created_at;
  paginate via `has_more`/`page_id`.
- gemini: `GET /v1beta/models?pageSize=1000` → includes `inputTokenLimit`/`outputTokenLimit` —
  the richest source; paginate via `nextPageToken` loop (bounded).
- **Normalization**: IDs returned as the API spells them (gemini returns `models/<id>` — trimmed
  to `<id>`; gemini exposes both base and versioned ids — no guessing, no dedup).
- **Error taxonomy**: transport vs `APIError{Provider, Status, Code, Message, Retryable}`; a
  failed listing is an error, **never a static fallback list**.
- Client-side cache **per SDK instance**, keyed per provider (id+baseURL+key hash), default TTL
  5 min; `WithModelCacheTTL(0)` disables; `ListModels(ctx, ForceRefresh)` bypasses.
- Context window for models whose API omits it: **report 0/unknown — never a static guess.**
  Consumers that need a bound (odek's trim engine) may pass an operator override.

### 4.5 Streaming (unified)

One `Delta` stream for all three formats: `{Kind: Reasoning|Content|ToolArgs, Text, ToolIndex, ToolName}`
plus assembled `ChatResult` at the end — including `Usage` (delivered in the final chunk when the
provider supports in-stream usage; anthropic splits it across `message_start`/`message_delta`).
Per-format SSE readers translate provider events (Anthropic `content_block_delta`, Gemini
`alt=sse` chunks, OpenAI deltas) into canonical deltas. Ported contract: idle watchdog (120s
default, package var), hard wall-clock deadline, **abort via handler error returns the partial
result alongside `*StreamAbortedError`**, retry only before the first delta (partial output is
never duplicated), keepalive comments reset the idle timer, buffered fallback when a provider
rejects streaming.

### 4.6 Auth

- `FromEnv()`: scan all registered providers' env vars (+ aliases) → configure every provider that
  has a key. Multiple keys = multiple live endpoints, usable concurrently.
- **Alias precedence**: `GEMINI_API_KEY` beats `GOOGLE_API_KEY`; `KIMI_API_KEY` beats
  `MOONSHOT_API_KEY`; primary wins silently, no error when both are set.
- **Base-URL override**: `<PROVIDER>_BASE_URL` (e.g. `ZAI_BASE_URL`) overrides the default —
  how the coding-plan endpoint and self-hosted gateways slot in without code changes.
- Explicit: `WithProvider("anthropic", WithAPIKey(…), WithBaseURL(…))`.
- Keys live only in the client struct; **never in error strings, `String()`, `RateLimitError`,
  or any typed error** — redaction guarantee extends to the whole error taxonomy.

### 4.7 Error taxonomy

| Error | When | Retryable |
|---|---|---|
| `*APIError{Provider, Status, Code, Message}` | non-2xx after retries exhausted | `Retryable` field; 408/429/5xx were retried, 4xx not |
| `*RateLimitError` (embeds `APIError`, adds `Attempts`, `RetryAfter`) | persistent 429 | final failure carries last `Retry-After` |
| `*StreamAbortedError{Reason}` | consumer delta handler returned error | no — returned WITH the partial `ChatResult` |
| `*ConfigError` | unknown provider id, missing key for a configured provider | no |

Retry policy (shared, all formats): 8 attempts, exponential backoff `1<<n` s capped at 30 s,
±20% jitter, `Retry-After` (seconds or HTTP-date) overrides backoff, ctx cancellation honored
between attempts, streaming retries only pre-first-delta.

## 5. Public API sketch

```go
sdk := llm.New(llm.FromEnv())            // every <PROVIDER>_API_KEY found; <PROVIDER>_BASE_URL honored

for _, p := range sdk.Providers() {      // authenticated endpoints, registry order
    models, _ := p.ListModels(ctx)        // dynamic; p.ListModels(ctx, llm.ForceRefresh) bypasses cache
    _ = models                            // []llm.Model{ID, ContextWindow, MaxOutputTokens, CreatedAt, Capabilities}
}

chat := sdk.Chat("deepseek", "deepseek-v4-flash")   // provider id + model id; ConfigError if unknown
chat.SetRequestTimeout(180 * time.Second)           // default 120s; RequestTimeout() reads it back
res, err := chat.Call(ctx, req)                     // buffered → *ChatResult
res, err = chat.CallStream(ctx, req, func(d llm.Delta) error {
    if d.Kind == llm.DeltaReasoning { ... }          // abort: return err → *StreamAbortedError + partial res
    return nil
})
```

## 6. Milestones

| # | Milestone | Contents |
|---|---|---|
| M1 | Scaffold | go.mod, LICENSE (MIT, house style), README, .gitignore, Makefile (fmt/vet/test quality gate), GitHub Actions `ci.yml` (build + test + `-race`) |
| M2 | Core types + registry | message.go, errors.go, provider.go, auth.go, transport.go + unit tests (env resolution incl. aliases, registry, custom providers) |
| M3 | OpenAI-format engine | Port of client/stream/retry + dynamic ListModels; httptest coverage: buffered, SSE, reasoning deltas, tool calls, retries, 429/Retry-After, learn-once fallbacks |
| M4 | Anthropic adapter | Translate canonical ⇄ native + SSE events + `GET /v1/models`; golden-JSON translation tests |
| M5 | Gemini adapter | Translate canonical ⇄ native + SSE + `GET /v1beta/models` (token limits); golden-JSON translation tests |
| M6 | Aggregate surface + docs | `New/FromEnv/Chat` facade, model cache TTL, README with provider matrix + examples |
| M7 | Verify | `go build ./...`, `go vet`, `go test ./... -race -count=1` green; CI green after first push |

TDD: every milestone lands RED tests first, then implementation (house convention).

## 7. Testing strategy

- **Hermetic only**: `httptest.Server` per adapter; no network, no real keys.
- Golden-JSON tests pin canonical⇄native translations (Anthropic system blocks, tool_use,
  thinking blocks; Gemini function calls, thinkingConfig).
- SSE fixtures as raw strings (multiline data blocks, keepalive comments, partial JSON).
- Retry/rate-limit tests with short-circuited backoff knobs (package vars, like odek).
- `-race -count=1` in CI and locally.

## 8. Repo conventions

- Branch `main` (already initialized with remote `git@github.com:BackendStack21/go-llm-sdk.git`);
  work happens on `feat/multi-provider-sdk`, squash-merge via PR once CI passes (house convention).
- Commits: conventional, small, per-milestone.
- PLAN.md stays in-repo as the design record.

## 9. odek migration path (follow-up, explicitly out of scope)

1. odek adds `github.com/BackendStack21/go-llm-sdk` dep; `internal/llm` becomes a thin shim.
2. `KnownProfiles`/`LookupProfile` deleted; loop engine reads context windows from `ListModels`
   (with operator override for unknown models).
3. Internal tests move to the SDK; shim removed.

## 10. Decisions & assumptions (challenge in polish passes)

1. Canonical format = OpenAI shape (not a neutral third format) — odek compat outweighs neutrality.
2. Provider IDs `openai|gemini|deepseek|zai|kimi|anthropic`; env keys as §4.2 (aliases: GOOGLE_API_KEY, MOONSHOT_API_KEY).
3. Z.ai env var is `ZAI_API_KEY` (no dots in env names).
4. Gemini uses its **native** API, not the OpenAI-compat layer (thinking + token limits + full tool fidelity).
5. Anthropic adapter is native too (top-level `system`, prompt-cache-friendly blocks) — matches odek's current special-casing.
6. No static fallback metadata: unknown context window stays unknown (0) + optional consumer override.
7. v0.x until odek migration lands; then v1.0.
8. Stdlib only — SSE parsing, JSON, HTTP pooling hand-rolled, exactly like odek today.
