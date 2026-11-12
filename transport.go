package llm

import (
	"net"
	"net/http"
	"time"
)

// Transport tuning defaults, shared by every provider client an SDK builds.
const (
	DefaultTimeout        = 120 * time.Second
	defaultMaxIdleConns   = 100
	defaultMaxIdlePerHost = 32
	defaultIdleTimeout    = 90 * time.Second
	defaultKeepAlive      = 30 * time.Second
	defaultDialTimeout    = 30 * time.Second
	// defaultTLSHandshake bounds one TLS handshake so a stalled one fails
	// (and is retried) instead of consuming the whole call budget.
	defaultTLSHandshake = 10 * time.Second
	// HTTP/2 health check: after defaultH2ReadIdle without a frame the
	// transport pings, and closes the connection if no reply arrives within
	// defaultH2PingTimeout. Without it a silently dropped connection (NAT
	// timeout, dead peer) keeps being reused until the kernel gives up.
	defaultH2ReadIdle    = 30 * time.Second
	defaultH2PingTimeout = 15 * time.Second
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
		TLSHandshakeTimeout: defaultTLSHandshake,
		// API responses are typically already the payload the model sees;
		// disabling compression avoids gzip buffering artifacts on SSE.
		DisableCompression: true,
		DialContext: (&net.Dialer{
			Timeout:   defaultDialTimeout,
			KeepAlive: defaultKeepAlive,
		}).DialContext,
		ForceAttemptHTTP2: true,
		// The field's doc comment still reads "does not yet have any
		// effect", but since Go 1.24 the bundled HTTP/2 transport takes its
		// ping settings from here (http2fillNetHTTPTransportConfig).
		HTTP2: &http.HTTP2Config{
			SendPingTimeout: defaultH2ReadIdle,
			PingTimeout:     defaultH2PingTimeout,
		},
	}
}

// newBufferedHTTP returns a client with a whole-request timeout.
func newBufferedHTTP(rt http.RoundTripper, timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	return &http.Client{Timeout: timeout, Transport: rt, CheckRedirect: sameHostRedirect}
}

// newStreamHTTP returns a client with no whole-request timeout: a
// client-level Timeout would kill an SSE body read mid-stream. Streaming
// calls enforce a hard wall-clock deadline plus an idle watchdog via
// context instead (chat.go).
func newStreamHTTP(rt http.RoundTripper) *http.Client {
	return &http.Client{Timeout: 0, Transport: rt, CheckRedirect: sameHostRedirect}
}

// sameHostRedirect refuses to follow a redirect to another host: Go strips
// only Authorization/Cookie across hosts, so x-api-key, x-goog-api-key and
// custom Headers would otherwise reach the redirect target. The redirect
// response itself is returned (and surfaces as an APIError).
func sameHostRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 10 || req.URL.Host != via[0].URL.Host {
		return http.ErrUseLastResponse
	}
	// Never downgrade https -> http: that would resend credentials in
	// plaintext.
	if via[0].URL.Scheme == "https" && req.URL.Scheme != "https" {
		return http.ErrUseLastResponse
	}
	return nil
}
