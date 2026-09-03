package llm

import (
	"context"
	"net/http"
	"sync"
	"time"
)

// SDK is a configured set of LLM endpoints. Build one with New and options:
//
//	sdk := llm.New(llm.FromEnv())
//
// Providers are authenticated when an API key resolves from the
// environment or is set explicitly; every authenticated provider is usable
// concurrently through the SDK's shared connection pool.
type SDK struct {
	mu        sync.RWMutex
	providers map[string]*Provider
	order     []string
	timeout   time.Duration
	cacheTTL  time.Duration
	rt        http.RoundTripper
}

// Option configures an SDK at construction time.
type Option func(*SDK)

// ProviderOption configures one provider entry.
type ProviderOption func(*ProviderConfig)

// New builds an SDK with the built-in provider registry and the given
// options. Options apply in order; later WithProvider calls for the same
// id override earlier ones.
func New(opts ...Option) *SDK {
	s := &SDK{
		providers: make(map[string]*Provider),
		timeout:   DefaultTimeout,
		cacheTTL:  5 * time.Minute,
		rt:        newPooledTransport(),
	}
	for _, cfg := range builtinProviders() {
		s.put(cfg)
	}
	for _, o := range opts {
		if o != nil {
			o(s)
		}
	}
	return s
}

func (s *SDK) put(cfg ProviderConfig) {
	if _, exists := s.providers[cfg.ID]; !exists {
		s.order = append(s.order, cfg.ID)
	}
	s.providers[cfg.ID] = &Provider{cfg: cfg, sdk: s}
}

func (s *SDK) get(id string) (*Provider, bool) {
	p, ok := s.providers[id]
	return p, ok
}

// WithEnv resolves API keys and base-URL overrides from the environment
// using lookup — the testable twin of FromEnv.
func WithEnv(lookup func(string) (string, bool)) Option {
	return func(s *SDK) {
		if lookup == nil {
			return
		}
		for _, id := range s.order {
			p := s.providers[id]
			if p.cfg.APIKey != "" {
				continue // explicit key wins over env
			}
			if cfg, ok := resolveEnvProvider(p.cfg, lookup); ok {
				p.cfg = cfg
			}
		}
	}
}

// FromEnv resolves <PROVIDER>_API_KEY (aliases included, primary first)
// and optional <PROVIDER>_BASE_URL overrides via os.LookupEnv. Providers
// without a key stay registered but unauthenticated.
func FromEnv() Option { return WithEnv(lookupEnv) }

// WithProvider configures one provider — a built-in id (override) or a new
// custom provider (requires WithFormat and WithBaseURL, plus auth).
func WithProvider(id string, popts ...ProviderOption) Option {
	return func(s *SDK) {
		existing, exists := s.providers[id]
		cfg := ProviderConfig{ID: id}
		if exists {
			cfg = existing.cfg
		}
		for _, po := range popts {
			if po != nil {
				po(&cfg)
			}
		}
		s.put(cfg)
		if !exists {
			// Custom entry: validate before it can poison requests.
			if p, ok := s.get(id); ok {
				p.invalid = validateProviderConfig(p.cfg) != nil
			}
		}
	}
}

// WithAPIKey sets an explicit API key (overrides env).
func WithAPIKey(key string) ProviderOption {
	return func(c *ProviderConfig) { c.APIKey = key }
}

// WithBaseURL overrides the provider's default base URL.
func WithBaseURL(url string) ProviderOption {
	return func(c *ProviderConfig) { c.BaseURL = url }
}

// WithFormat sets the wire format (custom providers).
func WithFormat(f Format) ProviderOption {
	return func(c *ProviderConfig) { c.Format = f }
}

// WithQuirks overrides protocol quirk flags.
func WithQuirks(q Quirks) ProviderOption {
	return func(c *ProviderConfig) { c.Quirks = q }
}

// WithEnvKeys sets env var names consulted for this provider's key,
// primary first.
func WithEnvKeys(keys ...string) ProviderOption {
	return func(c *ProviderConfig) { c.EnvKeys = keys }
}

// WithRequestTimeout sets the default per-request timeout for all chat
// clients (default 120s). Streaming calls use it as the hard wall-clock
// deadline.
func WithRequestTimeout(d time.Duration) Option {
	return func(s *SDK) {
		if d > 0 {
			s.timeout = d
		}
	}
}

// WithModelCacheTTL sets the ListModels cache TTL. Zero disables caching —
// every call hits the provider's models endpoint.
func WithModelCacheTTL(d time.Duration) Option {
	return func(s *SDK) { s.cacheTTL = d }
}

// WithTransport replaces the SDK's pooled HTTP transport (tests, proxies).
func WithTransport(rt http.RoundTripper) Option {
	return func(s *SDK) {
		if rt != nil {
			s.rt = rt
		}
	}
}

// Providers returns the authenticated providers in registry order.
func (s *SDK) Providers() []*Provider {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []*Provider
	for _, id := range s.order {
		if p := s.providers[id]; p != nil && p.Authenticated() && !p.invalid {
			out = append(out, p)
		}
	}
	return out
}

// Provider looks a provider up by id. Unknown ids return a ConfigError;
// known-but-unauthenticated providers are returned with
// Authenticated() == false.
func (s *SDK) Provider(id string) (*Provider, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p, ok := s.providers[id]
	if !ok {
		return nil, &ConfigError{Msg: "unknown provider \"" + id + "\""}
	}
	return p, nil
}

// Chat returns a chat client bound to a provider and model.
func (s *SDK) Chat(providerID, model string) (*ChatClient, error) {
	p, err := s.Provider(providerID)
	if err != nil {
		return nil, err
	}
	if !p.Authenticated() {
		return nil, &ConfigError{Msg: providerID + " has no API key (set " + providerID + "_API_KEY or use WithAPIKey)"}
	}
	if p.invalid {
		return nil, &ConfigError{Msg: providerID + " has an invalid configuration"}
	}
	return &ChatClient{
		pc:     newProviderClient(p.cfg, newBufferedHTTP(p.sdk.rt, p.sdk.timeout), newStreamHTTP(p.sdk.rt)),
		model:  model,
		parent: p,
	}, nil
}

// ── Provider ─────────────────────────────────────────────────────────────

// Provider is one configured endpoint with dynamic model discovery.
type Provider struct {
	cfg     ProviderConfig
	sdk     *SDK
	invalid bool

	clientOnce sync.Once
	listClient *providerClient

	mu       sync.Mutex
	cached   []Model
	cachedAt time.Time
}

// ID returns the provider's registry id.
func (p *Provider) ID() string { return p.cfg.ID }

// Authenticated reports whether an API key resolved.
func (p *Provider) Authenticated() bool { return p.cfg.APIKey != "" }

// Config returns the provider configuration. The copy carries the API key:
// treat it as a secret; it is never logged by the SDK itself.
func (p *Provider) Config() ProviderConfig { return p.cfg }

// ListOption tweaks ListModels.
type ListOption func(*listOpts)

type listOpts struct{ force bool }

// ForceRefresh bypasses the model cache for this call and refreshes it.
func ForceRefresh() ListOption { return func(o *listOpts) { o.force = true } }

// ListModels discovers the models accessible to this provider's account,
// on the fly — no static tables. Results are cached per SDK instance for
// the cache TTL (default 5 min; WithModelCacheTTL(0) disables). Fields the
// provider does not report stay zero.
func (p *Provider) ListModels(ctx context.Context, opts ...ListOption) ([]Model, error) {
	var lo listOpts
	for _, o := range opts {
		if o != nil {
			o(&lo)
		}
	}
	ttl := p.sdk.cacheTTL
	if !lo.force && ttl > 0 {
		p.mu.Lock()
		if p.cached != nil && time.Since(p.cachedAt) < ttl {
			out := make([]Model, len(p.cached))
			copy(out, p.cached)
			p.mu.Unlock()
			return out, nil
		}
		p.mu.Unlock()
	}
	if !p.Authenticated() {
		return nil, &ConfigError{Msg: p.cfg.ID + " has no API key"}
	}
	p.clientOnce.Do(func() {
		p.listClient = newProviderClient(p.cfg, newBufferedHTTP(p.sdk.rt, 30*time.Second), newStreamHTTP(p.sdk.rt))
	})
	models, err := p.listClient.listModels(ctx)
	if err != nil {
		return nil, err
	}
	p.mu.Lock()
	p.cached, p.cachedAt = models, time.Now()
	p.mu.Unlock()
	out := make([]Model, len(models))
	copy(out, models)
	return out, nil
}

// ── ChatClient ───────────────────────────────────────────────────────────

// ChatClient runs chat completions against one provider+model pair. Each
// client carries its own learn-once fallbacks and request timeout; it is
// safe for concurrent use but SetRequestTimeout must be called before the
// first request.
type ChatClient struct {
	pc     *providerClient
	model  string
	parent *Provider
}

// RequestTimeout reports the per-request timeout (streaming wall-clock
// deadline).
func (c *ChatClient) RequestTimeout() time.Duration { return c.pc.requestTimeout() }

// SetRequestTimeout adjusts the per-request timeout. Call before the first
// request.
func (c *ChatClient) SetRequestTimeout(d time.Duration) {
	if d <= 0 {
		return
	}
	c.pc.http = newBufferedHTTP(c.pc.http.Transport, d)
}

// ProviderID returns the bound provider's id.
func (c *ChatClient) ProviderID() string { return c.parent.ID() }

// Model returns the bound model.
func (c *ChatClient) Model() string { return c.model }

// Call runs a buffered chat completion.
func (c *ChatClient) Call(ctx context.Context, req *ChatRequest) (*ChatResult, error) {
	return c.pc.call(ctx, req, c.model)
}

// CallStream runs a streaming chat completion. onDelta receives canonical
// fragments; returning an error from it aborts generation — CallStream then
// returns the partial result alongside *StreamAbortedError.
func (c *ChatClient) CallStream(ctx context.Context, req *ChatRequest, onDelta func(Delta) error) (*ChatResult, error) {
	if onDelta == nil {
		return c.pc.call(ctx, req, c.model)
	}
	return c.pc.callStream(ctx, req, c.model, onDelta)
}
