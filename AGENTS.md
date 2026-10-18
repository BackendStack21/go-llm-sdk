# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Repository

**go-llm-sdk** — multi-provider Go SDK for LLM inference endpoints: OpenAI, Google Gemini, DeepSeek, Z.ai, Kimi (Moonshot), Anthropic, plus any OpenAI-compatible gateway. Module `github.com/BackendStack21/go-llm-sdk`. Go 1.25+, **stdlib only — zero external dependencies**. Flat single package (`package llm`) at the repo root.

## Commands

```bash
make quality                                   # fmt + vet + tests
make test-race                                 # go test -race -count=1
make lint                                      # golangci-lint (v2 config)
go test -count=1 -timeout 120s -race ./...     # full race suite
go test -tags e2e -run 'TestE2E' -timeout 15m -v .   # LIVE provider e2e (see below)
```

- Always run tests with `-count=1` and an explicit `-timeout` (house rule: no unbounded runs).
- Coverage sits at ~99.0% of statements (unit suite). The residual is documented unreachable defensive code — do not pad with fake tests to move the number.

## Architecture (flat package)

| File | Concern |
|---|---|
| `llm.go` | `SDK` entrypoint: `New`, options (`WithProvider`, `FromEnv`, …), `Chat`, `Provider` (model cache, shared learn-once state) |
| `message.go` | Canonical types: `ChatRequest`, `Message`, `ChatResult`, `Delta`, `Usage`, `ToolDef` |
| `content.go` | Content-part boundary validation (text, image, image URL, document; MIME case-insensitive, canonical lower-case on the wire), aggregate inline cap, attachment helpers |
| `chat.go` | `providerClient`: retry orchestration (buffered + streaming), error classification, learn-once consumption, SSE pump wiring, `httpError` parsing |
| `openai.go` / `gemini.go` / `anthropic.go` | Per-format request builders, response/stream mappers, model listing |
| `responses.go` | OpenAI Responses API (`/v1/responses`) for GPT-5.6+ tools+reasoning |
| `tts.go` | Text-to-speech: `Speak`/`SpeakRequest`/`SpeakResult` — OpenAI-compat `/audio/speech` + Gemini AUDIO modality, no transcoding |
| `controls.go` | Request controls: `ToolChoice`, `ResponseFormat` (Anthropic JSON is native `output_config.format`; the forced-tool emulation + `foldAnthropicJSON` run only under `Quirks.AnthropicToolJSON`, off for the built-in provider), `ParallelToolCalls`, per-format mapping and boundary validation |
| `observe.go` | `AttemptEvent` + `WithAttemptObserver`: per-attempt record filled by `doObserved` in the HTTP helpers and emitted from the retry ladders (`withRetryN`, `callStream`) with the delay about to be waited; key-masked errors; panics recovered |
| `meta.go` | `ResponseMeta`/`RateLimit` and `parseResponseMeta` (request-id + rate-limit headers from the successful attempt); per-request `Headers` validation (reserved auth/framing names rejected, no values in errors) and ctx carriage |
| `tools.go` | `RunTools`: buffered tool-call loop (`ToolHandler`, `ToolLoopOptions`, `ErrMaxToolTurns`); handler errors/panics/unknown tools become `IsError` results, `Parallel` keeps call order |
| `embed.go` | Embeddings: `Embed`/`EmbedRequest`/`EmbedResult` — OpenAI-compat `/embeddings` (requests `encoding_format: base64`, decodes float32 LE, accepts float arrays) + Gemini `batchEmbedContents` |
| `stt.go` | Speech-to-text: `Transcribe`/`TranscribeRequest`/`TranscribeResult` — OpenAI-compat `/audio/transcriptions` (multipart), 25MB input cap |
| `sse.go` | SSE parser (abort-safe via `done` channel) + idle-watchdog pump |
| `retry.go` | `RetryPolicy` + the single buffered ladder `withRetry` (chat, tts, stt, embed); backoff/jitter/`Retry-After`/`retrySleep` (default 8 attempts, cap 30s) |
| `provider.go` | Built-in registry, quirks flags, config validation |
| `models.go` | `ListModels` orchestration (shared `withRetryN` ladder: `WithRetryPolicy` attempts, default 3, Retry-After, request-timeout budget; deep-copied cache, empty lists cached) |
| `auth.go`, `transport.go`, `errors.go` | Env resolution, pooled HTTP clients, typed errors |

The canonical type system is OpenAI-shaped; `gemini.go`/`anthropic.go` translate both directions. Provider quirks are **explicit registry flags** — never URL sniffing.

## Invariants (breaking any of these is a defect)

1. **Canonical finish reasons** are `stop | length | tool_calls | content_filter | ""`. Unmapped provider stop reasons map to `""` on every format — provider-specific strings never leak. A tool-call turn that ends normally (provider stop/`STOP`/end, or the format's tool signal) finishes `tool_calls` on every format (Gemini's `STOP` and OpenAI-format `stop` are mapped); a `length`, `content_filter` or unmapped reason keeps that value (or `""`) even with calls present, since the calls may be incomplete.
   Refusal text surfaces in `ChatResult.Refusal` / `DeltaRefusal` (never in `Content`); a refusal on `stop`/unmapped finishes `content_filter`.
2. **API keys never appear** in error text, `String()`, or any typed error.
3. **Streaming**: retries only before the first emitted delta; a failure after partial output returns the partial `*ChatResult` + wrapped error and is **never retried**; a premature close (no completion signal — on every format, Gemini included; an unmapped finish reason still counts as one) is an error, never a silent empty or partial success; the parser goroutine is always released (abort-safe `done` protocol).
   A successful non-SSE response is parsed directly with the usual body cap and format mapper. It learns buffered mode for future requests, but never discards the current generation or makes a replacement request; parse/read errors remain terminal for that response.
4. **Unknown message roles are rejected** at the SDK boundary (`ConfigError`) — never dropped or reinterpreted per format.
5. **Unknown provider data stays unknown** (zero values) — no static guesses, no fallback model tables.
6. **Learn-once fallback state is per-`Provider`**, monotonic, atomic, shared by every `ChatClient` — never move it back to per-client.
7. Anthropic extended-thinking rounds trip via `ThinkingBlocks` (every `thinking` / `redacted_thinking` block, each with its own signature, in order) or the legacy single `ReasoningContent` + `ThinkingSignature` pair — replay is signature-gated and the thinking blocks go **first**. Signatures are per block, never concatenated. Gemini `thoughtSignature`s round-trip per part (`ToolCall.Signature`, `ThinkingSignature`).
8. **`Delta.ToolIndex` is the call's position in `ChatResult.ToolCalls`** on every format — never a provider content-block or output-item index.
9. **`Usage` has one meaning everywhere**: `PromptTokens` uncached-only, cache reads/writes in their fields (a DeepSeek miss is uncached input, not a write), `CompletionTokens` includes reasoning, `CachedTokens` is a diagnostic subset never summed.
10. **The request timeout is a whole-call budget**, retries included, on every entry point (buffered, streaming, speech, transcription, embeddings).
11. **Meta fields are provider-reported or zero, never guessed**; headers come from the successful attempt only. Attempt-observer panics are recovered and discarded; events never contain the API key.
12. **Anthropic JSON mode is native `output_config.format`** unless the explicit registry quirk `Quirks.AnthropicToolJSON` is on (off for the built-in provider, set via `WithQuirks`); the quirk restores the pre-M4 forced-tool emulation byte-for-byte, including its ConfigErrors (also under any thinking mode). Never infer it from URL or model name.

## Testing conventions

- **RED-first TDD**: failing test first, then the fix. Table-driven tests; `httptest` servers for hermetic coverage; `newTestClient` helper pins `backoffUnit` to 1ms — restore package vars in `t.Cleanup`.
- Timing knobs for tests: the package var `backoffUnit` (use `fastBackoff(t)`), and the process-wide idle default (atomic; use `setIdleForTest(t, d)`). Operators override the idle watchdog per SDK with `WithStreamIdleTimeout`, or process-wide with the race-safe `SetStreamIdleTimeout` (positive values only). Retry shape is per SDK via `WithRetryPolicy`.
- The e2e suite (`e2e_test.go`) is behind a `e2e` build tag and hits **live APIs**. It must never lose that tag. Keys come from env or a gitignored `.env`; contents are never logged; tests skip when a key is absent. Adding a provider = one `e2eTarget` entry; models overridable via `<ID>_E2E_MODEL`.
- Live-provider behavior (e.g. DeepSeek eliding `reasoning_content`) is **not an SDK contract** — probe softly, assert only what the SDK guarantees (call success, parsing, canonical finish). Model answer correctness is never an assertion. Elision is absorbed by an explicit provider quirk rather than hope: on a chat-completions request that carries tools, providers flagged `Quirks.EchoReasoningWithTools` always get the `reasoning_content` key, empty included — including on assistant turns that made no tool call themselves (a request diverted to `/responses` never reaches that builder). That flag is off for every provider except `deepseek`, and whether DeepSeek *accepts* a present-but-empty value is measured live by the tag-gated `TestE2EDeepSeekEmptyReasoningEcho` probe — not asserted by the unit suite.
- Timer hygiene: since Go 1.23 no drain-before-`Reset` is needed for `time.Timer`. Note: on some dev machines `time.After` + select-default spin loops have hung — prefer deadline loops in tests.

## Repo conventions

- Branches: `feat/` (features) and `fix/` (fixes). **PR → CI green → squash-merge** — never push straight to `main`.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`).
- Docs live in `README.md` (public contract) and this file (repo guidance). Keep both in sync with behavior in the same commit.
- CI: build+test matrix (ubuntu/macOS with race+coverage, windows build/vet) + golangci-lint v2 — keep it green; lint failures fail the PR.

## odek migration (follow-up workstream)

1. odek adds this module as a dependency; its `internal/llm` becomes a thin shim.
2. odek's static `KnownProfiles` are deleted; context windows come from `ListModels` (operator override for unknown models).
3. odek's internal LLM tests move here; the shim is removed.
