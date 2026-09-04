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
- Coverage sits at ~97.7% of statements. The residual is documented unreachable defensive code — do not pad with fake tests to move the number.

## Architecture (flat package)

| File | Concern |
|---|---|
| `llm.go` | `SDK` entrypoint: `New`, options (`WithProvider`, `FromEnv`, …), `Chat`, `Provider` (model cache, shared learn-once state) |
| `message.go` | Canonical types: `ChatRequest`, `Message`, `ChatResult`, `Delta`, `Usage`, `ToolDef` |
| `chat.go` | `providerClient`: retry orchestration (buffered + streaming), error classification, learn-once consumption, SSE pump wiring, `httpError` parsing |
| `openai.go` / `gemini.go` / `anthropic.go` | Per-format request builders, response/stream mappers, model listing |
| `sse.go` | SSE parser (abort-safe via `done` channel) + idle-watchdog pump |
| `retry.go` | Backoff/jitter/`Retry-After`/`retrySleep` (8 attempts, cap 30s) |
| `provider.go` | Built-in registry, quirks flags, config validation |
| `models.go` | `ListModels` orchestration (retries, caps, cache backing) |
| `auth.go`, `transport.go`, `errors.go` | Env resolution, pooled HTTP clients, typed errors |

The canonical type system is OpenAI-shaped; `gemini.go`/`anthropic.go` translate both directions. Provider quirks are **explicit registry flags** — never URL sniffing.

## Invariants (breaking any of these is a defect)

1. **Canonical finish reasons** are `stop | length | tool_calls | content_filter | ""`. Unmapped provider stop reasons map to `""` on every format — provider-specific strings never leak.
2. **API keys never appear** in error text, `String()`, or any typed error.
3. **Streaming**: retries only before the first emitted delta; a failure after partial output returns the partial `*ChatResult` + wrapped error and is **never retried**; a premature close (no completion signal) is an error, never a silent empty success; the parser goroutine is always released (abort-safe `done` protocol).
4. **Unknown message roles are rejected** at the SDK boundary (`ConfigError`) — never dropped or reinterpreted per format.
5. **Unknown provider data stays unknown** (zero values) — no static guesses, no fallback model tables.
6. **Learn-once fallback state is per-`Provider`**, monotonic, atomic, shared by every `ChatClient` — never move it back to per-client.
7. Anthropic extended-thinking rounds trip via `Message.ThinkingSignature` / `ChatResult.ThinkingSignature` — replay is signature-gated and the thinking block goes **first**.

## Testing conventions

- **RED-first TDD**: failing test first, then the fix. Table-driven tests; `httptest` servers for hermetic coverage; `newTestClient` helper pins `backoffUnit` to 1ms — restore package vars in `t.Cleanup`.
- Two timing knobs are package vars for tests: `backoffUnit`, `streamIdleTimeout`.
- The e2e suite (`e2e_test.go`) is behind a `e2e` build tag and hits **live APIs**. It must never lose that tag. Keys come from env or a gitignored `.env`; contents are never logged; tests skip when a key is absent. Adding a provider = one `e2eTarget` entry; models overridable via `<ID>_E2E_MODEL`.
- Live-provider behavior (e.g. DeepSeek eliding `reasoning_content`) is **not an SDK contract** — probe softly, assert only what the SDK guarantees (call success, parsing, canonical finish). Model answer correctness is never an assertion.
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
