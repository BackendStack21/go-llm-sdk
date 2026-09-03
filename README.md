# go-llm-sdk

Multi-provider Go SDK for LLM inference endpoints — **OpenAI, Google Gemini, DeepSeek, Z.ai, Kimi (Moonshot) and Anthropic**, plus any custom OpenAI-compatible gateway. Stdlib only, zero external dependencies.

- **Multiple authenticated endpoints at once** — auto-discovered from `<PROVIDER>_API_KEY` environment variables (aliases supported).
- **Dynamic model discovery** — `ListModels` returns what the account can actually access, on the fly. No static model tables, ever.
- **One canonical API** — OpenAI-shaped requests/responses; Anthropic and Gemini wire formats are translated for you.
- **Production streaming semantics** (ported from [odek](https://github.com/BackendStack21/odek)'s battle-tested client): SSE with idle watchdog + hard wall-clock deadline, abort-with-partial-result, retries that never duplicate partial output, and learn-once fallbacks for providers that reject `stream_options`, streaming, or `reasoning_effort`+tools.

## Quickstart

```go
sdk := llm.New(llm.FromEnv()) // reads OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY,
                              // ZAI_API_KEY, KIMI_API_KEY, ANTHROPIC_API_KEY (+ aliases)

for _, p := range sdk.Providers() { // authenticated endpoints, registry order
    models, err := p.ListModels(ctx)
    // → []llm.Model{ID, DisplayName, CreatedAt, ContextWindow, MaxOutputTokens, Capabilities}
}

chat, err := sdk.Chat("deepseek", "deepseek-v4-flash")

res, err := chat.Call(ctx, &llm.ChatRequest{
    System:   []llm.SystemBlock{{Text: "Be terse."}},
    Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello"}},
    Tools:    []llm.ToolDef{{Name: "get_weather", Parameters: schema}},
})

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

Primary env var beats its alias. Explicit keys (`WithAPIKey`) beat env. Base-URL overrides accept any gateway speaking the provider's format.

Custom gateways:

```go
sdk := llm.New(llm.WithProvider("my-gateway",
    llm.WithFormat(llm.FormatOpenAI),
    llm.WithBaseURL("http://localhost:11434/v1"),
    llm.WithAPIKey("local"),
))
```

## Provider quirks, handled

The registry carries explicit quirk flags (no URL sniffing): Anthropic/DeepSeek/GLM thinking objects, OpenAI/GLM `reasoning_effort`, GLM-5.3 forced-thinking models (`disabled` → `enabled` + `reasoning_effort: low`), the `anthropic-version` header, Gemini `systemInstruction`/`thinkingConfig`, and temperature-forbidding models (`o1/o3/o4/gpt-5*/kimi-for-coding*/k3*` never receive an explicit temperature). Learn-once fallbacks fix provider rejections at runtime without repeated failed round-trips.

## Retry policy

8 attempts, exponential backoff capped at 30s with ±20% jitter, `Retry-After` (seconds or HTTP-date) honored, context cancellation between attempts. Persistent 429s surface as `*llm.RateLimitError{Attempts, RetryAfter}`. Streaming retries happen only before the first emitted delta.

## Errors

`*ConfigError` (unknown/unauthenticated provider), `*APIError{Provider, Status, Code, Message, Retryable}`, `*RateLimitError`, `*StreamAbortedError` (returned together with the partial `*ChatResult`). API keys never appear in any error text.

## Design record

See [PLAN.md](PLAN.md) for the architecture and the odek migration path.

## Status

v0 — API may shift until the odek integration lands, then v1.0.

## License

MIT — see [LICENSE](LICENSE).
