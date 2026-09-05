package llm

import (
	"net"
	"net/http"
	"time"
)

// Transport tuning defaults, shared by every provider client an SDK builds.
const (
	DefaultTimeout        = 120 * time.Second
	defaultMaxIdleConns   = 20
	defaultMaxIdlePerHost = 10
	defaultIdleTimeout    = 90 * time.Second
	defaultKeepAlive      = 30 * time.Second
	defaultDialTimeout    = 30 * time.Second
)

// newPooledTransport builds a connection-pooled transport. Each SDK builds
// one and shares it across its buffered and streaming clients, so repeated
// calls reuse TCP+TLS connections instead of handshaking every request.
func newPooledTransport() *http.Transport {
	return &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		MaxIdleConns:        defaultMaxIdleConns,
		MaxIdleConnsPerHost: defaultMaxIdlePerHost,
		IdleConnTimeout:     defaultIdleTimeout,
		// API responses are typically already the payload the model sees;
		// disabling compression avoids gzip buffering artifacts on SSE.
		DisableCompression: true,
		DialContext: (&net.Dialer{
			Timeout:   defaultDialTimeout,
			KeepAlive: defaultKeepAlive,
		}).DialContext,
		ForceAttemptHTTP2: true,
	}
}

// newBufferedHTTP returns a client with a whole-request timeout.
func newBufferedHTTP(rt http.RoundTripper, timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	return &http.Client{Timeout: timeout, Transport: rt}
}

// newStreamHTTP returns a client with no whole-request timeout: a
// client-level Timeout would kill an SSE body read mid-stream. Streaming
// calls enforce a hard wall-clock deadline plus an idle watchdog via
// context instead (chat.go).
func newStreamHTTP(rt http.RoundTripper) *http.Client {
	return &http.Client{Timeout: 0, Transport: rt}
}
